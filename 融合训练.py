"""以原生 YOLO 训练循环执行三分支融合、固定矩形增强与可追溯验证。

main.py 是唯一训练入口；此模块不独立运行，也不启动测试或显存探测。
正式轮末验证复用同一批预测计算分域指标，不附加独立模型评估。
"""

from __future__ import annotations

# 内置库
import csv
import hashlib
import json
import logging
import math
import random
import shutil
import sys
from contextlib import redirect_stdout, redirect_stderr
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

# 三方库
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics.data.augment import Compose
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.torch_utils import unwrap_model

# 自己的模块
from 三模态训练 import MultimodalDetectionTrainer, MultimodalYOLODataset
from 三模态融合 import (FUSION_VERSION, FusionDetectionModel,
                       canvas_shape, clip_canvas_boxes, letterbox_fused, single_label_nms)
from 准备三模态数据集 import CLASS_NAMES, IMAGE_SUFFIXES, file_hash


# 收尾至少完成二十轮后才允许早停；主权重始终按任何真实 AP95 新高保存。
MIN_POLISH_EPOCHS: int = 20


@dataclass(frozen=True)
class FusionRecipe:
    """保存原生 YOLO 参数之外的融合与矩形训练设置。"""

    image_height: int = 1080
    image_width: int = 1920
    backbone_lr: float = 0.00002
    auxiliary_lr: float = 0.0002
    val_batch: int = 1
    min_delta: float = 0.0002
    data_audit: str = "./runs/dataset_cleaning/official_refresh_20260918_214843/manifest.json"


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """按固定列名追加正式训练记录，首次写入表头。"""
    if not rows:
        return
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


class RectangularDataset(MultimodalYOLODataset):
    """原尺寸读取三模态，所有批次输出相同的步长补齐矩形。"""

    def __init__(self, *args: Any, content_hw: tuple[int, int], audit_digest: str, **kwargs: Any) -> None:
        self.content_hw = content_hw
        self.audit_digest = audit_digest
        self.polish = False
        self.hyp = kwargs["hyp"]
        super().__init__(*args, **kwargs)

    def get_cache_hash(self) -> str:
        """将审计指纹加入标签缓存键，防止复用旧配方的标签缓存。"""
        value = f"{super().get_cache_hash()}|{FUSION_VERSION}|{self.audit_digest}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def build_transforms(self, hyp: Any = None) -> Compose:
        """增强与格式化由本类统一完成，避免原生流程重新变成正方形。"""
        return Compose([])

    def close_mosaic(self, hyp: Any) -> None:
        """框架第81轮重启加载器时关闭拼图、尺度和位移。"""
        self.polish = True

    def _sample(self, index: int, content_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple, tuple]:
        """读取原图和归一化标签，应用共用矩形几何变换。"""
        image = self.load_fused_image(index)
        h, w = image.shape[:2]
        label = self.labels[index]
        if label.get("bbox_format", "xywh") != "xywh" or not label.get("normalized", True):
            raise ValueError("矩形数据集要求官方归一化xywh检测标签")
        xywh = label["bboxes"].astype(np.float32).copy()
        boxes = np.empty_like(xywh)
        boxes[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2
        boxes[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2
        boxes *= np.array([w, h, w, h], dtype=np.float32)
        canvas, geometry = letterbox_fused(image, content_hw)
        sx, sy, left, top = geometry
        boxes = boxes * np.array([sx, sy, sx, sy]) + np.array([left, top, left, top])
        return canvas, label["cls"].copy(), boxes.astype(np.float32), geometry, (h, w)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """同步拼图、几何及RGB颜色增强，返回原生YOLO损失所需字段。"""
        mosaic = self.augment and not self.polish and random.random() < self.hyp.mosaic
        if mosaic:
            out_h, out_w = canvas_shape(self.content_hw)
            tile_h, tile_w = out_h // 2, out_w // 2
            image = np.zeros((out_h, out_w, 5), dtype=np.uint8)
            image[:, :, :3] = 114
            categories, coordinates = [], []
            for slot, member in enumerate([index] + random.choices(range(len(self)), k=3)):
                # 子图也保留16:9内容比例；固定四块拼图避免3840平方中间画布。
                tile, cls, boxes, _, _ = self._sample(member, (self.content_hw[0] // 2, self.content_hw[1] // 2))
                y, x = slot // 2 * tile_h, slot % 2 * tile_w
                image[y:y + tile_h, x:x + tile_w] = tile
                coordinates.append(boxes + np.array([x, y, x, y]))
                categories.append(cls)
            cls, boxes = np.concatenate(categories), np.concatenate(coordinates).astype(np.float32)
            original_hw, geometry = (out_h, out_w), (1.0, 1.0, 0, 0)
        else:
            image, cls, boxes, geometry, original_hw = self._sample(index, self.content_hw)
        h, w = image.shape[:2]
        if self.augment:
            if not self.polish and (self.hyp.scale or self.hyp.translate):
                scale = random.uniform(1 - self.hyp.scale, 1 + self.hyp.scale)
                dx = w * ((1 - scale) / 2 + random.uniform(-self.hyp.translate, self.hyp.translate))
                dy = h * ((1 - scale) / 2 + random.uniform(-self.hyp.translate, self.hyp.translate))
                matrix = np.array([[scale, 0, dx], [0, scale, dy]], dtype=np.float32)
                image = np.stack([cv2.warpAffine(image[:, :, c], matrix, (w, h), flags=cv2.INTER_LINEAR,
                                                borderValue=114 if c < 3 else 0) for c in range(5)], axis=-1)
                boxes = boxes * scale + np.array([dx, dy, dx, dy])
                old_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
                bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
                keep = (bw > 1) & (bh > 1) & (bw * bh > old_area * 0.1)
                boxes, cls = boxes[keep], cls[keep]
            if random.random() < self.hyp.fliplr:
                image = image[:, ::-1].copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            hsv = cv2.cvtColor(np.ascontiguousarray(image[:, :, :3]), cv2.COLOR_RGB2HSV).astype(np.float32)
            gains = np.random.uniform(-1, 1, 3) * np.array([self.hyp.hsv_h, self.hyp.hsv_s, self.hyp.hsv_v])
            hsv[:, :, 0] = (hsv[:, :, 0] + gains[0] * 180) % 180
            hsv[:, :, 1:] = np.clip(hsv[:, :, 1:] * (1 + gains[1:]), 0, 255)
            image[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        normalized = np.empty_like(boxes, dtype=np.float32)
        normalized[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        normalized[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        normalized /= np.array([w, h, w, h], dtype=np.float32)
        sx, sy, left, top = geometry
        return {"img": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                "cls": torch.from_numpy(cls.astype(np.float32)), "bboxes": torch.from_numpy(normalized),
                "batch_idx": torch.zeros(len(cls)), "im_file": self.im_files[index],
                "ori_shape": original_hw, "resized_shape": (h, w), "ratio_pad": ((sx, sy), (left, top))}


class RectangularValidator(DetectionValidator):
    """使用固定矩形、FP32和与提交一致的单标签NMS。"""

    def __call__(self, trainer: Any = None, model: Any = None, **kwargs: Any) -> dict[str, float]:
        if trainer is None:
            raise ValueError("本验证器只随正式训练运行，不启动额外独立评估")
        self.epoch = trainer.epoch + 1
        # 原生验证会收窄此开关；每轮重置，确保收尾和最后一轮能生成曲线。
        self.args.plots = trainer.args.plots
        amp = trainer.amp
        trainer.amp = False
        try:
            return super().__call__(trainer, model, **kwargs)
        finally:
            trainer.amp = amp

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.domains: list[str] = []

    def postprocess(self, preds: Any) -> list[dict[str, torch.Tensor]]:
        rows = single_label_nms(preds, self.args.conf, self.args.iou, self.args.max_det)
        return [{"bboxes": row[:, :4], "conf": row[:, 4], "cls": row[:, 5], "extra": row[:, 6:]} for row in rows]

    def update_metrics(self, preds: list[dict], batch: dict) -> None:
        for prediction, original_hw, ratio_pad in zip(preds, batch["ori_shape"], batch["ratio_pad"], strict=True):
            (sx, sy), (left, top) = ratio_pad
            boxes, keep = clip_canvas_boxes(prediction["bboxes"], (sx, sy, left, top), original_hw)
            for key in prediction:
                prediction[key] = prediction[key][keep]
            prediction["bboxes"] = boxes
        super().update_metrics(preds, batch)
        self.domains.extend("png" if Path(path).suffix.lower() == ".png" else "jpg" for path in batch["im_file"])

    def get_stats(self) -> dict[str, float]:
        """从本轮已匹配的预测复用计算分域AP，不额外运行网络。"""
        rows: list[dict[str, Any]] = []
        for domain in ("png", "jpg"):
            selected = [i for i, value in enumerate(self.domains) if value == domain]
            if not selected:
                continue
            metrics = DetMetrics(names=self.names)
            for key in metrics.stats:
                metrics.stats[key] = [self.metrics.stats[key][i] for i in selected]
            metrics.process()
            rows.append({"epoch": self.epoch, "domain": domain, "class_id": -1, "name": "all",
                         "images": len(selected), "AP50": metrics.box.map50, "AP50_95": metrics.box.map})
            for position, category in enumerate(metrics.ap_class_index):
                rows.append({"epoch": self.epoch, "domain": domain, "class_id": int(category),
                             "name": self.names[int(category)], "images": len(selected),
                             "AP50": metrics.box.ap50[position], "AP50_95": metrics.box.ap[position]})
        append_csv(self.save_dir / "domain_metrics.csv", rows)
        return super().get_stats()


class PolishEarlyStopping:
    """允许至少二十轮完整画面收尾，耐心计数与真实最佳保存分离。"""

    def __init__(self, polish_start: int, patience: int, min_delta: float) -> None:
        self.polish_start, self.patience, self.min_delta = polish_start, patience, min_delta
        self.best = -math.inf
        self.wait = 0
        self.possible_stop = False

    def __call__(self, epoch: int, fitness: float) -> bool:
        if epoch < self.polish_start:
            return False
        if fitness > self.best + self.min_delta:
            self.best, self.wait = fitness, 0
        else:
            self.wait += 1
        allowed = epoch >= self.polish_start + MIN_POLISH_EPOCHS - 1
        self.possible_stop = allowed and self.wait >= self.patience - 1
        return allowed and self.wait >= self.patience


class LogStream:
    """把正式训练的进度条复制到日志，保留原终端输出。"""

    def __init__(self, console: TextIO, file: TextIO) -> None:
        self.console, self.file = console, file

    def write(self, value: str) -> int:
        self.console.write(value)
        return self.file.write(value)

    def flush(self) -> None:
        self.console.flush()
        self.file.flush()

    def isatty(self) -> bool:
        return self.console.isatty()


class FusionDetectionTrainer(MultimodalDetectionTrainer):
    """沿用原生YOLO训练循环，显式配置融合结构、矩形数据与指标记录。"""

    def __init__(self, *args: Any, recipe: FusionRecipe, **kwargs: Any) -> None:
        self.recipe = recipe
        self.requested = dict(kwargs.get("overrides") or {})
        self.resume_metadata: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)
        if self.args.imgsz != recipe.image_width or self.args.rect or self.args.multi_scale:
            raise ValueError("imgsz须等于内容宽度；固定矩形由自定义数据集提供，rect和multi_scale须关闭")
        if (recipe.image_height, recipe.image_width) != (1080, 1920):
            raise ValueError("首版矩形拼图限定1920×1080内容窗口，避免子图步长取整不一致")
        if self.args.batch < 1 or self.args.nbs % self.args.batch or self.args.cache:
            raise ValueError("batch须为有效批次nbs的正整数因子，且cache=False")
        if self.args.epochs - self.args.close_mosaic < 5 or self.args.close_mosaic < MIN_POLISH_EPOCHS:
            raise ValueError("训练须留至少5轮主训练及20轮收尾")
        if self.args.cls_pw != 0 or self.args.optimizer != "AdamW" or self.args.nms is not True:
            raise ValueError("融合配方要求AdamW、cls_pw=0与nms=True")
        if self.data["names"] != dict(enumerate(CLASS_NAMES)) or self.data["channels"] != 5:
            raise ValueError("融合训练必须使用比赛原顺序的12类、五通道数据配置")
        self.audit = self._verify_audit()
        self.best_map50 = -math.inf
        self.last_aux_grad: torch.Tensor | None = None

    def _verify_audit(self) -> dict[str, Any]:
        """开训时只核验清单和标签指纹，拒绝未审计的删改，不重做数据清洗。"""
        path = Path(self.recipe.data_audit).resolve()
        audit = json.loads(path.read_text(encoding="utf-8"))
        root = Path(self.args.data).resolve().parent
        if audit.get("status") != "applied" or len(audit.get("samples", [])) != 2000:
            raise ValueError("清洗审计未落位或不是官方2000组")
        for split in ("train", "val"):
            rows = [row for row in audit["samples"] if row["after_split"] == split]
            names = {row["image"] for row in rows}
            for modality in ("images", "infrared", "depth"):
                present = {p.name for p in (root / split / modality).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
                if present != names:
                    raise ValueError(f"{split}/{modality}与清洗清单不一致")
            if {p.stem for p in (root / split / "labels").glob("*.txt")} != {Path(n).stem for n in names}:
                raise ValueError(f"{split}标签文件清单与清洗审计不一致")
            for row in rows:
                if file_hash(root / split / "labels" / (Path(row["image"]).stem + ".txt")) != row["after_sha256"]:
                    raise ValueError(f"标签发生未记录变更：{row['image']}")
        return {"manifest_sha256": file_hash(path), "yaml_sha256": file_hash(Path(self.args.data)),
                "counts": audit["counts"], "source": self.recipe.data_audit}

    def check_resume(self, overrides: dict[str, Any]) -> None:
        super().check_resume(overrides)
        if self.resume:
            # 恢复写新运行目录，避免覆盖历史日志和权重。
            self.args.project = overrides["project"]
            self.args.name = overrides["name"] + "_resume"
            self.args.exist_ok = False
            self.args.save_dir = None

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> FusionDetectionModel:
        """按语义迁移官方RGB权重，恢复仅接受同配方的融合检查点。"""
        if weights is None:
            raise ValueError("请从本地官方YOLO26l预训练权重创建融合模型")
        source = (weights.get("ema") or weights["model"]) if isinstance(weights, dict) else weights
        model = self.set_model_names_for_load(FusionDetectionModel(cfg or source.yaml, self.data["nc"], verbose))
        if getattr(source, "fusion_version", None):
            if not self.resume or source.fusion_version != FUSION_VERSION:
                raise ValueError("新训练须用官方RGB基底；融合检查点只用于同配方断点恢复")
            self.resume_metadata = deepcopy(getattr(source, "fusion_training", None))
            if not self.resume_metadata or self.resume_metadata.get("completed"):
                raise ValueError("该融合检查点已完成或缺少恢复元数据")
        else:
            if self.resume or source.model[0].conv.in_channels != 3:
                raise ValueError("首训只接受官方RGB预训练，不能恢复旧五通道或D-FINE模型")
        load_source = copy(source)
        load_source.names = {key: "ball" if str(name).lower() == "sports ball" else name for key, name in source.names.items()}
        model.load(load_source, verbose=verbose)
        model.content_hw = (self.recipe.image_height, self.recipe.image_width)
        if self.resume_metadata:
            model.fusion_training = deepcopy(self.resume_metadata)
        else:
            ball_id = next((key for key, value in source.names.items() if str(value).lower() == "sports ball"), None)
            if ball_id is None or model.names.get(7) != "ball":
                raise ValueError("缺少sports ball→ball=7的预训练语义映射")
            for branch in ("cv3", "one2one_cv3"):
                for target, original in zip(getattr(model.model[-1], branch), getattr(source.model[-1], branch), strict=True):
                    for field in ("weight", "bias"):
                        if not torch.equal(getattr(target[-1], field)[7].detach().cpu(),
                                           getattr(original[-1], field)[ball_id].detach().float().cpu()):
                            raise ValueError(f"ball迁移失败：{branch}/{field}")
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> RectangularDataset:
        return RectangularDataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
                                  augment=mode == "train", hyp=copy(self.args), rect=False, cache=False,
                                  single_cls=False, stride=32, pad=0.0, prefix=f"{mode}: ", task="detect",
                                  classes=None, data=self.data, fraction=1.0,
                                  content_hw=(self.recipe.image_height, self.recipe.image_width),
                                  audit_digest=self.audit["manifest_sha256"])

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = -1, mode: str = "train") -> Any:
        return super().get_dataloader(dataset_path, self.recipe.val_batch if mode == "val" else batch_size, rank, mode)

    def build_optimizer(self, model: torch.nn.Module, name: str = "AdamW", lr: float = 0.001,
                        momentum: float = 0.9, decay: float = 1e-5, iterations: float = 1e5) -> torch.optim.Optimizer:
        """将参数按主干/检测器/新增分支及衰减规则分组，预热保持学习率比例。"""
        native = unwrap_model(model)
        backbone_ids = {id(p) for module in native.model[:11] for p in module.parameters()}
        auxiliary_ids = {id(p) for module in (native.ir_encoder, native.depth_encoder, native.ir_fusion, native.depth_fusion)
                         for p in module.parameters()}
        groups: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
        for parameter in native.parameters():
            role = "backbone" if id(parameter) in backbone_ids else "auxiliary" if id(parameter) in auxiliary_ids else "head"
            groups.setdefault((role, parameter.ndim > 1), []).append(parameter)
        rates = {"backbone": self.recipe.backbone_lr, "auxiliary": self.recipe.auxiliary_lr, "head": lr}
        return torch.optim.AdamW([{"params": values, "lr": rates[role], "initial_lr": rates[role],
                                  "weight_decay": decay if apply_decay else 0.0, "role": role,
                                  "param_group": "weight" if apply_decay else "bias"}
                                 for (role, apply_decay), values in groups.items()], betas=(momentum, 0.999))

    def get_validator(self) -> RectangularValidator:
        return RectangularValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def _setup_train(self) -> None:
        super()._setup_train()
        self.stopper = PolishEarlyStopping(self.epochs - self.args.close_mosaic + 1, self.args.patience, self.recipe.min_delta)
        files = ("main.py", "三模态融合.py", "融合训练.py", "三模态训练.py", "准备三模态数据集.py")
        # main.py允许只改RESUME_PATH和资源参数，配方本身另行比较；组件源码不可偷偷变化。
        sources = {name: file_hash(Path(__file__).parent / name) for name in files if name != "main.py"}
        package = Path(ultralytics.__file__).parent
        framework = {"ultralytics": ultralytics.__version__, "torch": str(torch.__version__),
                     "files": {name: file_hash(package / name) for name in (
                         "nn/tasks.py", "nn/modules/head.py", "utils/loss.py", "utils/nms.py",
                         "engine/trainer.py", "engine/validator.py", "models/yolo/detect/val.py")}}
        selected = ("epochs", "nbs", "lr0", "lrf", "warmup_epochs", "weight_decay", "mosaic", "scale", "translate",
                    "fliplr", "hsv_h", "hsv_s", "hsv_v", "box", "cls", "close_mosaic", "patience", "seed")
        settings = {key: getattr(self.args, key) for key in selected}
        self.training_signature = {"version": FUSION_VERSION, "recipe": asdict(self.recipe), "hyp": settings,
                                   "audit": self.audit, "sources": sources, "framework": framework}
        if self.resume_metadata:
            if self.resume_metadata["signature"] != self.training_signature:
                raise ValueError("断点的配方、数据审计或源码已改变，请新训而非混合恢复")
            if any(key in self.requested and self.requested[key] != value for key, value in settings.items()):
                raise ValueError("恢复时请求参数与检查点配方不同")
            self.stopper.best = self.resume_metadata["stop_best"]
            self.stopper.wait = self.resume_metadata["stop_wait"]
            self.best_map50 = self.resume_metadata["best_map50"]
        for model in (unwrap_model(self.model), self.ema.ema):
            model.criterion = model.init_criterion()
        (self.save_dir / "optimization_recipe.json").write_text(json.dumps(
            {**self.training_signature, "content_hw": [self.recipe.image_height, self.recipe.image_width],
             "tensor_hw": canvas_shape((self.recipe.image_height, self.recipe.image_width)),
             "head_loss_weights": [0.8, 0.2], "validation_precision": "FP32",
             "optimizer_groups": [{"role": group["role"], "kind": group["param_group"],
                                   "initial_lr": group["initial_lr"], "weight_decay": group["weight_decay"]}
                                  for group in self.optimizer.param_groups]}, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(self.recipe.data_audit, self.save_dir / "dataset_audit.json")
        code = self.save_dir / "code"
        code.mkdir(exist_ok=True)
        for name in files:
            shutil.copy2(Path(__file__).parent / name, code / name)
        LOGGER.info(f"融合输入：内容1920×1080，张量1920×1088；物理批次{self.batch_size}，有效批次{self.args.nbs}")

    def optimizer_step(self) -> None:
        native = unwrap_model(self.model)
        gradients = [p.grad.detach().float().norm() for module in (native.ir_encoder, native.depth_encoder)
                     for p in module.parameters() if p.grad is not None]
        if gradients:
            self.last_aux_grad = torch.stack(gradients).norm()
        super().optimizer_step()

    def validate(self) -> tuple[dict[str, float], float]:
        metrics, _ = super().validate()
        fitness = float(metrics["metrics/mAP50-95(B)"])
        # 原生fitness当前也是AP95，显式锁定以免框架版本改变选择语义。
        self.best_fitness = max(self.best_fitness or 0.0, fitness)
        box = self.validator.metrics.box
        positions = {int(category): i for i, category in enumerate(box.ap_class_index)}
        rows = []
        for category, name in enumerate(CLASS_NAMES):
            i = positions.get(category)
            rows.append({"epoch": self.epoch + 1, "class_id": category, "name": name,
                         "precision": float(box.p[i]) if i is not None else 0.0,
                         "recall": float(box.r[i]) if i is not None else 0.0,
                         "AP50": float(box.ap50[i]) if i is not None else None,
                         "AP50_95": float(box.ap[i]) if i is not None else None})
        append_csv(self.save_dir / "per_class_metrics.csv", rows)
        native = unwrap_model(self.model)
        loss = native.criterion
        diagnostics = {"epoch": self.epoch + 1, "aux_grad_norm": float(self.last_aux_grad) if self.last_aux_grad is not None else None}
        for branch in ("ir", "depth"):
            for level, block in zip((3, 4, 5), getattr(native, f"{branch}_fusion"), strict=True):
                diagnostics[f"{branch}_p{level}_gate"] = float(block.last_gate_mean) if block.last_gate_mean is not None else None
        for key in sorted(loss.running):
            diagnostics[key] = float(loss.running[key]) / max(loss.batches, 1)
        append_csv(self.save_dir / "fusion_diagnostics.csv", [diagnostics])
        loss.running.clear()
        loss.batches = 0
        return metrics, fitness

    def save_model(self) -> bool:
        """保留原生最佳/断点，附加配方与早停状态，并独立留AP50最佳。"""
        current50 = float(self.metrics["metrics/mAP50(B)"])
        better50 = current50 > self.best_map50
        self.best_map50 = max(self.best_map50, current50)
        state = {"signature": self.training_signature, "best_map50": self.best_map50,
                 "stop_best": self.stopper.best, "stop_wait": self.stopper.wait, "completed": bool(self.stop)}
        for model in (unwrap_model(self.model), self.ema.ema):
            model.fusion_training = deepcopy(state)
        result = super().save_model()
        if result and better50:
            shutil.copy2(self.last, self.wdir / "best_map50.pt")
        return result

    def final_eval(self) -> None:
        """不重复加载最佳权重复评；每轮已完成正式验证，保留完整检查点。"""
        LOGGER.info(f"训练结束；最佳权重 {self.best}，AP50候选 {self.wdir / 'best_map50.pt'}。未追加独立复评。")

    def train(self) -> None:
        handler = logging.FileHandler(self.save_dir / "train.log", encoding="utf-8")
        LOGGER.addHandler(handler)
        try:
            with (self.save_dir / "train.log").open("a", encoding="utf-8") as log:
                with redirect_stdout(LogStream(sys.stdout, log)), redirect_stderr(LogStream(sys.stderr, log)):
                    super().train()
        finally:
            LOGGER.removeHandler(handler)
            handler.close()
