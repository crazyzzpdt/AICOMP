"""为 v5 提供弱类采样、同步增强、分层微调与双指标权重留存。

由 main.py 导入；使用既有 datasets 划分，增强仅发生在内存中。
模型仍为标准五通道 YOLO26l，训练完成后直接使用现有 predict.py。
"""

# 内置库
import csv
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from collections.abc import Iterator
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace

# 三方库
import cv2
import numpy as np
import torch
from torch.utils.data import Sampler
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, nms
from ultralytics.utils.torch_utils import unwrap_model

# 自己的模块
from 三模态训练 import MultimodalDetectionTrainer, MultimodalYOLODataset


# v4 漏检集中的类别；仅改变训练出现频率，不改官方类别编号。
WEAK_CLASSES: tuple[int, ...] = (3, 4, 5, 7, 8, 9)
# 参考 repeat-factor sampling，低于 20% 图像频率的弱类获得额外采样。
REPEAT_THRESHOLD: float = 0.20
# 每张图每轮至多作为主样本出现三次，防止少数球图主导梯度。
MAX_IMAGE_REPEATS: int = 3
# 每个文件名场景组每轮最多增加八张主样本；分组只是重复限制，不重新划分数据。
MAX_GROUP_EXTRAS: int = 8
# 仅含弱类的图有 35% 概率裁剪，仍保留至少 55% 原始宽高的上下文。
TARGET_CROP_PROBABILITY: float = 0.35
# RGB 单独做轻度色彩变化，数值分别为色相、饱和度、亮度变化幅度。
RGB_HSV_GAINS: tuple[float, float, float] = (0.015, 0.25, 0.20)
# 10% 概率随机缺失一个辅助模态，训练对局部无效深度或红外退化的容忍度。
MODALITY_DROPOUT: float = 0.10
# 预训练骨干学习率为检测头的五分之一；首层保持完整学习率以学习新增模态。
BACKBONE_LR_RATIO: float = 0.20
# 原尺寸裁剪及完整模态收尾改变了配方，不能恢复上一版 v5 的优化器状态。
RECIPE_VERSION: str = "v5_full_nativecrop_polish"


def group_training_images(paths: list[str]) -> list[str]:
    """按文件名前缀与纯数字连续段限制重复，不宣称完全识别真实场景。

    Args:
        paths: 仅训练集的图片路径，顺序与标签一致。

    Returns:
        每张图片对应的采样分组名。
    """
    numeric = sorted({int(Path(path).stem) for path in paths if Path(path).stem.isdecimal()})
    numeric_groups: dict[int, str] = {}
    previous: int = -1000
    start: int = 0
    for number in numeric:
        if number - previous > 50:
            start = number
        numeric_groups[number] = f"numeric_{start}"
        previous = number
    groups: list[str] = []
    for path in paths:
        stem = Path(path).stem
        groups.append(numeric_groups[int(stem)] if stem.isdecimal() else stem.replace("_suppl_", "_").rsplit("_", 1)[0])
    return groups


class WeakClassSampler(Sampler[int]):
    """每轮覆盖所有训练图，并在图像与场景组双重上限内补充弱类。"""

    def __init__(self, dataset: MultimodalYOLODataset, seed: int) -> None:
        """仅根据训练标签建立额外采样池，验证集不参与频率计算。"""
        self.seed: int = seed
        self.epoch: int = 0
        self.size: int = len(dataset)
        categories = [set(label["cls"].astype(int).reshape(-1).tolist()) for label in dataset.labels]
        frequency = {category: sum(category in present for present in categories) / self.size for category in WEAK_CLASSES}
        pools: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for index, (group, present) in enumerate(zip(group_training_images(dataset.im_files), categories)):
            factors = [math.sqrt(REPEAT_THRESHOLD / frequency[category]) for category in present & set(WEAK_CLASSES)]
            extra = max(0.0, min(float(MAX_IMAGE_REPEATS), max(factors, default=1.0)) - 1.0)
            if extra:
                for _ in range(MAX_IMAGE_REPEATS - 1):
                    pools[group].append((index, extra / (MAX_IMAGE_REPEATS - 1)))
        self.pools: list[tuple[list[tuple[int, float]], int]] = [
            (pool, min(MAX_GROUP_EXTRAS, len(pool), math.ceil(sum(weight for _, weight in pool))))
            for _, pool in sorted(pools.items())
        ]
        self.total: int = self.size + sum(count for _, count in self.pools)
        self.frequency: dict[int, float] = frequency

    def __len__(self) -> int:
        """返回固定轮长度，确保学习率、预热与进度条计数一致。"""
        return self.total

    def __iter__(self) -> Iterator[int]:
        """保留全部原图一次，在各组内加权抽取有限附加项后打乱。"""
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(self.size))
        for pool, count in self.pools:
            weights = torch.tensor([weight for _, weight in pool], dtype=torch.double)
            selected = torch.multinomial(weights, count, replacement=False, generator=generator).tolist()
            indices.extend(pool[position][0] for position in selected)
        order = torch.randperm(len(indices), generator=generator).tolist()
        return iter(indices[position] for position in order)


def crop_weak_target(label: dict[str, object], source_image: np.ndarray) -> dict[str, object]:
    """从原尺寸五通道图裁剪弱类上下文，避免先缩小再放大丢失细节。

    Args:
        label: 含缩放图及其坐标系下标注的当前训练样本。
        source_image: 同一样本的原尺寸五通道图，可为只读 NPY 映射。

    Returns:
        同步裁剪后恢复原训练尺寸的样本；无法完整保留目标时返回原样本。

    Note:
        目标完整保留；裁剪边缘的其他类别也保留可见标注，不把它们变成假背景。
        调用方仅在训练阶段使用；缓存图像及原始标签不就地修改。
    """
    categories = label["cls"].reshape(-1).astype(int)
    candidates = np.flatnonzero(categories == 7)
    if not len(candidates):
        candidates = np.flatnonzero(np.isin(categories, WEAK_CLASSES))
    if not len(candidates):
        return label
    output_height, output_width = label["img"].shape[:2]
    height, width = source_image.shape[:2]
    # 框先还原到原图；独立副本保证裁剪不可行时原样本的坐标系不被改变。
    instances = deepcopy(label["instances"])
    instances.convert_bbox("xyxy")
    instances.denormalize(output_width, output_height)
    instances.scale(width / output_width, height / output_height)
    box = instances.bboxes[int(random.choice(candidates))]
    ratio = min(1.0, max(random.uniform(0.55, 0.85), (box[2] - box[0]) / width * 1.4, (box[3] - box[1]) / height * 1.4))
    crop_width, crop_height = min(width, math.ceil(width * ratio)), min(height, math.ceil(height * ratio))
    # 左上角取值须让目标完整留在画面中，同时随机移动上下文。
    low_x, high_x = max(0, math.ceil(box[2] - crop_width)), min(width - crop_width, math.floor(box[0]))
    low_y, high_y = max(0, math.ceil(box[3] - crop_height)), min(height - crop_height, math.floor(box[1]))
    if low_x > high_x or low_y > high_y:
        return label
    left, top = random.randint(low_x, high_x), random.randint(low_y, high_y)
    cropped = source_image[top:top + crop_height, left:left + crop_width]
    label["img"] = cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    instances.add_padding(-left, -top)
    instances.clip(crop_width, crop_height)
    keep = instances.remove_zero_area_boxes()
    label["cls"] = label["cls"][keep]
    instances.scale(output_width / crop_width, output_height / crop_height)
    label["instances"] = instances
    return label


class WeakClassDataset(MultimodalYOLODataset):
    """训练时增加弱类上下文裁剪、RGB 色彩增强与辅助模态随机缺失。"""

    # 跟随框架关闭 Mosaic 的生命周期切换；加载器重建后子进程读取新状态。
    strong_augmentation: bool = True

    def read_crop_source(self, index: int) -> np.ndarray:
        """只读映射原尺寸融合缓存，缺失或无效时从三张官方模态图读取。

        Note:
            不写缓存、不持有整套原图；正常磁盘缓存下只访问裁剪需要的页面。
        """
        if self.cache_is_current(index):
            try:
                image = np.load(self.npy_files[index], mmap_mode="r", allow_pickle=False)
                if image.ndim == 3 and image.shape[2] == 5 and image.dtype == np.uint8:
                    return image
                LOGGER.warning(f"目标裁剪缓存格式异常，改读原始三模态：{self.im_files[index]}")
            except (OSError, ValueError) as error:
                LOGGER.warning(f"目标裁剪缓存不可读，改读原始三模态：{self.im_files[index]}，{error}")
        return self.load_fused_image(index)

    def close_mosaic(self, hyp: SimpleNamespace) -> None:
        """收尾时关闭目标裁剪与辅助模态缺失，保留轻度色彩和几何变化。"""
        self.strong_augmentation = False
        super().close_mosaic(hyp)
        LOGGER.info("完整模态收尾：Mosaic、目标裁剪、随机模态缺失已关闭；弱类采样与轻度增强保留")

    def get_image_and_label(self, index: int) -> dict[str, object]:
        """先复用五通道配对，再对当前训练样本做同步增强。"""
        label = super().get_image_and_label(index)
        if not self.augment:
            return label
        categories = label["cls"].reshape(-1).astype(int)
        if self.strong_augmentation and np.isin(categories, WEAK_CLASSES).any() and random.random() < TARGET_CROP_PROBABILITY:
            source_image = self.read_crop_source(index)
            label = crop_weak_target(label, source_image)
            del source_image
        image = label["img"].copy()
        hsv = cv2.cvtColor(np.ascontiguousarray(image[:, :, :3]), cv2.COLOR_RGB2HSV).astype(np.float32)
        gains = np.random.uniform(-1.0, 1.0, 3) * np.asarray(RGB_HSV_GAINS)
        hsv[:, :, 0] = (hsv[:, :, 0] + gains[0] * 180) % 180
        hsv[:, :, 1:] = np.clip(hsv[:, :, 1:] * (1.0 + gains[1:]), 0, 255)
        image[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if self.strong_augmentation and random.random() < MODALITY_DROPOUT:
            image[:, :, random.choice((3, 4))] = 0
        label["img"] = image
        return label


class SubmissionDetectionValidator(DetectionValidator):
    """按现有提交预测的单标签 NMS 计算指标，减少选权重与提交之间的差异。"""

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """沿用官方匹配及 AP 计算，仅对齐预测的候选类别筛选。"""
        outputs = nms.non_max_suppression(
            preds, self.args.conf, self.args.iou, nc=0, multi_label=False,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det, end2end=self.end2end, rotated=False,
        )
        return [{"bboxes": row[:, :4], "conf": row[:, 4], "cls": row[:, 5], "extra": row[:, 6:]} for row in outputs]


class OptimizedMultimodalTrainer(MultimodalDetectionTrainer):
    """在标准 YOLO26l 上改善弱类学习与迁移保持，保存可追溯训练记录。"""

    def get_model(self, cfg: str | None = None, weights: object = None, verbose: bool = True) -> DetectionModel:
        """沿用类别迁移并核对两个检测分支的球类参数，恢复时检查配方。"""
        model = super().get_model(cfg, weights, verbose)
        source = (weights.get("ema") or weights["model"]) if isinstance(weights, dict) else weights
        if getattr(self, "resume", False):
            if getattr(source, "optimization_recipe", None) != RECIPE_VERSION:
                raise ValueError(f"只能恢复配方 {RECIPE_VERSION} 的检查点；开始新配方时请设置 RESUME_PATH=None")
            model.metric_bests = dict(getattr(source, "metric_bests", {}))
        if source is not None and source.model[0].conv.in_channels == 3:
            source_id = next((key for key, name in source.names.items() if name.strip().lower() == "sports ball"), None)
            if source_id is None or model.names.get(7) != "ball":
                raise ValueError("预训练 sports ball 或比赛 ball=7 对应缺失")
            copied: int = 0
            for branch in ("cv3", "one2one_cv3"):
                for target_layer, source_layer in zip(getattr(model.model[-1], branch), getattr(source.model[-1], branch)):
                    for field in ("weight", "bias"):
                        target_row = getattr(target_layer[-1], field)[7].detach().cpu()
                        source_row = getattr(source_layer[-1], field)[source_id].detach().cpu().to(target_row.dtype)
                        if target_row.shape != source_row.shape or not torch.equal(target_row, source_row):
                            raise ValueError(f"球类预训练参数未正确迁移：{branch}/{field}")
                        copied += 1
            if copied != 12:
                raise ValueError(f"球类迁移核对不完整：期望两个分支、三个尺度的 12 项参数，实际 {copied} 项")
            LOGGER.info(f"ball 迁移核对完成：COCO {source_id} → 比赛 7，两个分支共 {copied} 个参数切片一致")
        model.optimization_recipe = RECIPE_VERSION
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> WeakClassDataset:
        """保持划分与五通道顺序，验证采用与单图预测相同的最小填充。"""
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return WeakClassDataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=mode == "train",
            hyp=copy(self.args), rect=self.args.rect or mode == "val", cache=self.args.cache or None,
            single_cls=self.args.single_cls, stride=stride, pad=0.0, prefix=f"{mode}: ",
            task=self.args.task, classes=self.args.classes, data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = -1, mode: str = "train") -> InfiniteDataLoader:
        """训练使用受限重复采样，验证沿用顺序加载且不增加样本。"""
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank != -1 or self.args.rect:
            raise ValueError("v5 弱类采样要求单卡 device=0、rect=False")
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        sampler = WeakClassSampler(dataset, self.args.seed)
        LOGGER.info(f"弱类采样：{len(dataset)} 张原图全部覆盖，每轮 {len(sampler)} 个主样本；每图至多 {MAX_IMAGE_REPEATS} 次")
        workers = min(self.args.workers, math.ceil(len(dataset) / batch_size))
        return InfiniteDataLoader(
            dataset=dataset, batch_size=min(batch_size, len(dataset)), sampler=sampler,
            num_workers=workers, prefetch_factor=1 if workers else None, pin_memory=False,
            collate_fn=dataset.collate_fn, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.args.seed), drop_last=False,
        )

    def build_optimizer(self, model: torch.nn.Module, name: str = "AdamW", lr: float = 0.001,
                        momentum: float = 0.9, decay: float = 1e-5, iterations: float = 1e5) -> torch.optim.Optimizer:
        """拆分现有衰减分组，让骨干慢学、首层和检测头正常适配。"""
        if name.lower() != "adamw":
            raise ValueError("v5 分层微调配方固定使用 AdamW")
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        native = unwrap_model(model)
        backbone_count = len(native.yaml["backbone"])
        slow_ids = {id(parameter) for module in native.model[1:backbone_count] for parameter in module.parameters()}
        groups = list(optimizer.param_groups)
        optimizer.param_groups = []
        for group in groups:
            for slow in (False, True):
                parameters = [parameter for parameter in group["params"] if (id(parameter) in slow_ids) == slow]
                if parameters:
                    updated = {key: value for key, value in group.items() if key != "params"}
                    updated.update(params=parameters, lr=lr * (BACKBONE_LR_RATIO if slow else 1.0))
                    optimizer.add_param_group(updated)
        LOGGER.info(f"分层学习率：骨干 {lr * BACKBONE_LR_RATIO:g}，首层/颈部/检测头 {lr:g}；预热与余弦衰减保持比例")
        return optimizer

    def get_validator(self) -> SubmissionDetectionValidator:
        """使用单标签 NMS 验证器，标准 best.pt 仍按 mAP50-95 选择。"""
        return SubmissionDetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def _setup_train(self) -> None:
        """框架初始化后记录真实标签统计、数据指纹与 v5 配方。"""
        super()._setup_train()
        self.metric_bests: dict[str, float] = dict(getattr(unwrap_model(self.model), "metric_bests", {}))
        if self.resume:
            self.train_loader.sampler.epoch = self.start_epoch
            self.train_loader.reset()
        statistics: dict[str, object] = {}
        for split, dataset in (("train", self.train_loader.dataset), ("val", self.test_loader.dataset)):
            counts = np.zeros(self.data["nc"], dtype=int)
            image_counts = np.zeros_like(counts)
            digest = hashlib.sha256()
            for label, path in zip(dataset.labels, dataset.label_files):
                categories = label["cls"].astype(int).reshape(-1)
                counts += np.bincount(categories, minlength=len(counts))
                image_counts[np.unique(categories)] += 1
                digest.update(Path(path).name.encode("utf-8"))
                digest.update(Path(path).read_bytes())
            statistics[split] = {"images": len(dataset), "boxes": counts.tolist(), "class_images": image_counts.tolist(), "label_sha256": digest.hexdigest()}
            LOGGER.info(f"{split}: ball={counts[7]} 框/{image_counts[7]} 张，类别编号={self.data['names'][7]}")
        recipe = {"recipe": RECIPE_VERSION, "weak_classes": WEAK_CLASSES, "repeat_threshold": REPEAT_THRESHOLD,
                  "max_image_repeats": MAX_IMAGE_REPEATS, "max_group_extras": MAX_GROUP_EXTRAS,
                  "crop_probability": TARGET_CROP_PROBABILITY, "crop_source": "native_resolution",
                  "polish_start_epoch": self.epochs - self.args.close_mosaic + 1 if self.args.close_mosaic else None,
                  "polish_disables": ["mosaic", "target_crop", "modality_dropout"], "rgb_hsv_gains": RGB_HSV_GAINS,
                  "modality_dropout": MODALITY_DROPOUT, "backbone_lr_ratio": BACKBONE_LR_RATIO,
                  "samples_per_epoch": len(self.train_loader.sampler), "validation_multi_label": False,
                  "validation_pad": 0.0, "statistics": statistics}
        (self.save_dir / "optimization_recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
        for filename in ("main.py", "三模态训练.py", "训练优化.py"):
            shutil.copy2(Path(__file__).parent / filename, self.save_dir / filename)

    def validate(self) -> tuple[dict[str, float] | None, float | None]:
        """利用本轮已有验证结果记录全部类别 AP，不进行额外推理。"""
        metrics, fitness = super().validate()
        if metrics is None:
            return metrics, fitness
        box = self.validator.metrics.box
        positions = {int(category): position for position, category in enumerate(box.ap_class_index)}
        output = self.save_dir / "per_class_metrics.csv"
        new_file = not output.exists()
        with output.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(("epoch", "class_id", "name", "precision", "recall", "AP50", "AP50_95"))
            for category, name in self.data["names"].items():
                position = positions.get(category)
                values = (0.0, 0.0, 0.0, 0.0) if position is None else box.class_result(position)
                writer.writerow((self.epoch + 1, category, name, *map(float, values)))
        ball_position = positions.get(7)
        ball_ap = float(box.ap50[ball_position]) if ball_position is not None else 0.0
        LOGGER.info(f"第 {self.epoch + 1} 轮 ball AP50={ball_ap:.5f}；逐类指标已写入 per_class_metrics.csv")
        return metrics, fitness

    def save_model(self) -> bool:
        """按两种 AP 各留存一个完整检查点；保留框架原有 best/last。"""
        destinations: list[str] = []
        for metric, filename in (("metrics/mAP50(B)", "best_map50.pt"), ("metrics/mAP50-95(B)", "best_map5095.pt")):
            value = float(self.metrics[metric])
            if math.isfinite(value) and value > self.metric_bests.get(metric, -math.inf):
                self.metric_bests[metric] = value
                destinations.append(filename)
        self.ema.ema.metric_bests = dict(self.metric_bests)
        self.ema.ema.optimization_recipe = RECIPE_VERSION
        saved = super().save_model()
        if saved:
            for filename in destinations:
                temporary = self.wdir / (filename + ".tmp")
                shutil.copy2(self.last, temporary)
                temporary.replace(self.wdir / filename)
                LOGGER.info(f"独立最佳权重已保存：{filename}，第 {self.epoch + 1} 轮")
        return saved
