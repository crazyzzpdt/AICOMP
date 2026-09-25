"""以原生YOLO训练循环执行三模态融合，支持深度质量输入及历史结构。

train1.py 是唯一YOLO训练入口；此模块不独立运行，也不启动测试或显存探测。
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
from collections import Counter
from collections.abc import Iterator
from contextlib import redirect_stdout, redirect_stderr
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, TextIO

# 三方库
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics.data.augment import Compose, LetterBox, Mosaic, RandomPerspective
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import get_hash
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, plot_mc_curve, plot_pr_curve
from ultralytics.utils.torch_utils import unwrap_model

# 自己的模块
from .model import (EARLY_FUSION_VERSION, EVALUATION_PROTOCOL, FUSION_VERSION, SPLIT_STEM_VERSION, QUALITY_FUSION_VERSION,
                    RGB_DIAGNOSTIC_VERSION, RGBDiagnosticModel,
                    EarlyFusionDetectionModel, FusionDetectionModel, SplitModalStem, QualityFusionDetectionModel, QualityLetterBox,
                    canvas_shape, clip_canvas_boxes, letterbox_fused, resize_channels, resize_native_fused,
                    resize_native_quality, single_label_nms)
from .data import (CLASS_NAMES, IMAGE_SUFFIXES, QUALITY_PREPROCESS_VERSION, DEPTH_MIN_MM, DEPTH_MAX_MM,
                   file_hash, fuse_modalities, fuse_quality_modalities, resize_quality_image,
                   transform_quality_image, verify_source_images)
from src.modalities import (FLOAT_PREPROCESS_VERSION, SensorAugment, augment_sensors, configure_fp32,
                            read_float_modalities, resize_float_image, transform_float_image, float_canvas)


# 历史主训练默认至少二十轮；显式短收尾可单独设置，最佳权重始终按真实AP95保存。
MIN_POLISH_EPOCHS: int = 20


# 每进程有限缓存；学习率周期保持整理前的计算方式。
IMAGE_BUFFER_LIMIT: int = 8
LR_DECAY_EPOCHS: int = 200
# 内部包迁移后仍以仓库根目录查找入口与审计。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


def resolve_modality_directory(data: dict[str, object], modality: str, img_path: str) -> Path:
    """解析各拆分的本地模态目录，同时兼容历史共享目录配置。"""
    root = Path(str(data.get("path", Path.cwd()))).resolve()
    configured = data[modality]
    if isinstance(configured, dict):
        split = Path(img_path).resolve().parent.name
        if split not in configured:
            raise ValueError(f"模态 {modality} 没有拆分 {split} 的路径")
        configured = configured[split]
    return (root / str(configured)).resolve()


class MultimodalYOLODataset(YOLODataset):
    """读取同名 RGB、红外、深度图并交给 Ultralytics 检测增强流程。"""

    input_channels: int = 5
    retain_buffer_images: bool = True

    def __init__(self, *args: object, data: dict[str, object], polish_scale: float | None = None,
                 polish_translate: float | None = None, **kwargs: object) -> None:
        """初始化模态根目录后构建 YOLO 标签数据集。

        Args:
            args: 传递给 Ultralytics YOLODataset 的位置参数。
            data: 含 ``infrared``、``depth`` 与 ``channels: 5`` 的数据集配置。
            kwargs: 传递给 Ultralytics YOLODataset 的关键字参数。

        Raises:
            KeyError: 数据集配置缺少三模态目录。
            ValueError: 数据集通道数不是五。
        """
        self.polish_scale, self.polish_translate = polish_scale, polish_translate
        if data.get("channels") != 5:
            raise ValueError("三模态早期融合训练的数据集 channels 必须为 5")
        img_path = str(kwargs.get("img_path", args[0] if args else ""))
        self.infrared_dir = resolve_modality_directory(data, "infrared", img_path)
        self.depth_dir = resolve_modality_directory(data, "depth", img_path)
        if not self.infrared_dir.is_dir() or not self.depth_dir.is_dir():
            raise FileNotFoundError("红外或深度图目录不存在，请检查 datasets/data.yaml")
        super().__init__(*args, data=data, **kwargs)
        self.max_buffer_length = min(self.max_buffer_length, IMAGE_BUFFER_LIMIT)

    def get_label_files(self) -> list[str]:
        """建立可见光图、红外图、深度图与新版标签的同名映射。"""
        label_files = super().get_label_files()
        self.infrared_files = [self.infrared_dir / Path(image_path).name for image_path in self.im_files]
        self.depth_files = [self.depth_dir / Path(image_path).name for image_path in self.im_files]
        for modality_path, description in (
            *[(path, "红外图") for path in self.infrared_files],
            *[(path, "深度图") for path in self.depth_files],
        ):
            if not modality_path.is_file():
                raise FileNotFoundError(f"{description}缺失，无法构成三模态样本：{modality_path}")
        return label_files

    def get_cache_hash(self) -> str:
        """标签内容参与缓存键，防止同字节数改类后仍读到旧标签。"""
        files = self.label_files + self.im_files + [str(path) for path in self.infrared_files + self.depth_files]
        files.append(f"fusion-v2-classes-{','.join(map(str, self.data['names'].values()))}")
        labels = "|".join(file_hash(Path(path)) if Path(path).is_file() else "missing" for path in self.label_files)
        return hashlib.sha256(f"{get_hash(files)}|{labels}".encode("utf-8")).hexdigest()

    def load_fused_image(self, index: int) -> np.ndarray:
        """读取索引对应的原始尺寸五通道图像。"""
        visible_path = Path(self.im_files[index])
        return fuse_modalities(
            visible_path,
            self.infrared_dir / visible_path.name,
            self.depth_dir / visible_path.name,
        )

    def resize_image(self, image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
        """历史数据集保持原插值；v19覆盖此接口处理深度支持。"""
        return resize_channels(image, size_wh)

    def resize_native_image(self, image: np.ndarray) -> np.ndarray:
        """历史权重的原生长边缩放保持不变。"""
        return resize_native_fused(image, self.imgsz)

    def cache_images_to_disk(self, index: int) -> None:
        """将五通道融合图写入 NPY 缓存，避免后续反复解码三个源文件。"""
        cache_path = self.npy_files[index]
        if cache_path.exists() and not self.cache_is_current(index):
            cache_path.unlink()
        if not cache_path.exists():
            try:
                np.save(cache_path.as_posix(), self.load_fused_image(index), allow_pickle=False)
            except Exception as error:
                cache_path.unlink(missing_ok=True)
                LOGGER.warning(f"{self.prefix}无法缓存三模态图像 {cache_path}：{error}")

    def cache_is_current(self, index: int) -> bool:
        """源图或融合实现更新后重新生成缓存，避免复用旧 BGR 输入。"""
        visible_path = Path(self.im_files[index])
        sources = (visible_path, self.infrared_dir / visible_path.name, self.depth_dir / visible_path.name,
                   Path(__file__), Path(fuse_modalities.__code__.co_filename))
        cache_path = self.npy_files[index]
        return cache_path.is_file() and cache_path.stat().st_mtime_ns >= max(path.stat().st_mtime_ns for path in sources)

    def load_image(
        self, index: int, rect_mode: bool = True, resize_short: bool = False
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """加载五通道图像，并保持 Ultralytics 的原始缩放和缓存语义。"""
        image, cache_path = self.ims[index], self.npy_files[index]
        if image is None:
            if self.cache == "disk" and self.cache_is_current(index):
                try:
                    image = np.load(cache_path, allow_pickle=False)
                    if image.ndim != 3 or image.shape[2] != self.input_channels:
                        raise ValueError(f"缓存通道数不符合当前输入契约：期望 {self.input_channels}")
                except Exception as error:
                    LOGGER.warning(f"{self.prefix}移除失效三模态缓存 {cache_path}：{error}")
                    cache_path.unlink(missing_ok=True)
                    image = self.load_fused_image(index)
            else:
                image = self.load_fused_image(index)

            height_original, width_original = image.shape[:2]
            if rect_mode:
                if resize_short:
                    ratio = self.imgsz / min(height_original, width_original)
                    if ratio != 1:
                        width, height = (
                            (math.ceil(width_original * ratio), self.imgsz)
                            if height_original < width_original
                            else (self.imgsz, math.ceil(height_original * ratio))
                        )
                        image = self.resize_image(image, (width, height))
                else:
                    image = self.resize_native_image(image)
            elif not (height_original == width_original == self.imgsz):
                image = self.resize_image(image, (self.imgsz, self.imgsz))

            if self.augment and self.cache != "ram":
                if self.retain_buffer_images:
                    self.ims[index] = image
                    self.im_hw0[index] = (height_original, width_original)
                    self.im_hw[index] = image.shape[:2]
                # 浮点模式只保留同样的Mosaic候选索引；命中不刷新FIFO，保持原采样规则。
                if index not in self.buffer:
                    self.buffer.append(index)
                    if 1 < len(self.buffer) >= self.max_buffer_length:
                        old_index = self.buffer.pop(0)
                        self.ims[old_index], self.im_hw0[old_index], self.im_hw[old_index] = None, None, None

            return image, (height_original, width_original), image.shape[:2]
        return image, self.im_hw0[index], self.im_hw[index]

    def close_mosaic(self, hyp: Any) -> None:
        """关闭原生拼图，只有显式配置时才降低尺度和位移扰动。"""
        if self.polish_scale is not None:
            hyp.scale = self.polish_scale
        if self.polish_translate is not None:
            hyp.translate = self.polish_translate
        super().close_mosaic(hyp)


class RGBDiagnosticDataset(MultimodalYOLODataset):
    """复用v14完整增强后只输出RGB，避免三通道Format翻转颜色和额外增强。"""

    def __getitem__(self, index: int) -> dict[str, Any]:
        # 磁盘和增强仍是原五通道；网络收到真正三通道，不用辅助输入置零伪装融合。
        sample = super().__getitem__(index)
        sample["img"] = sample["img"][:3].contiguous()
        return sample


class FloatLetterBox(LetterBox):
    """保留原生标签更新，只替换连续浮点图像的缩放及填充。"""

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = resize_float_image(labels["img"], tuple(params["new_unpad"]))
        height, width = image.shape[:2]
        top, left = params["top"], params["left"]
        canvas = float_canvas(height + top + params["bottom"], width + left + params["right"])
        canvas[top:top + height, left:left + width] = image
        labels["img"], labels["resized_shape"] = canvas, params["new_shape"]
        return labels


class FloatMosaic(Mosaic):
    """四图布局和标签计算不变，避免原生uint8画布量化深度。"""

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or self.n != 4:
            raise ValueError("浮点协议只支持四图Mosaic")
        canvas = float_canvas(self.imgsz * 2, self.imgsz * 2)
        for item in params["layout"]:
            image = item["labels_patch"]["img"]
            canvas[item["y1a"]:item["y2a"], item["x1a"]:item["x2a"]] = image[item["y1b"]:item["y2b"], item["x1b"]:item["x2b"]]
        labels["img"] = canvas
        return labels


class FloatPerspective(RandomPerspective):
    """沿用原生矩阵和框过滤，图像全程连续浮点。"""

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None:
            raise ValueError("缺少几何矩阵")
        matrix, size = params["M"], params["size"]
        operation = (lambda plane, border, mode: cv2.warpPerspective(plane, matrix, size, flags=mode, borderValue=border)) if self.perspective else (
            lambda plane, border, mode: cv2.warpAffine(plane, matrix[:2], size, flags=mode, borderValue=border))
        labels["img"] = transform_float_image(labels["img"], operation)
        labels["resized_shape"] = labels["img"].shape[:2]
        return labels


class FloatYOLODataset(MultimodalYOLODataset):
    """新训练的连续浮点协议，不读取旧融合缓存。"""

    retain_buffer_images: bool = False

    def __init__(self, *args: Any, sensors: SensorAugment, **kwargs: Any) -> None:
        self.sensors = sensors
        self.metric_depth: dict[int, bool] = {}
        super().__init__(*args, **kwargs)

    def load_fused_image(self, index: int) -> np.ndarray:
        path = Path(self.im_files[index])
        image, metric = read_float_modalities(path, self.infrared_dir / path.name, self.depth_dir / path.name)
        self.metric_depth[index] = metric
        return image

    def resize_image(self, image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
        return resize_float_image(image, size_wh)

    def resize_native_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ratio = self.imgsz / max(h, w)
        return resize_float_image(image, (min(math.ceil(w * ratio), self.imgsz), min(math.ceil(h * ratio), self.imgsz)))

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        sample = super().get_image_and_label(index)
        if self.augment:
            # 缓存保留干净浮点图；每次取样重新增强，不原地累积噪声。
            sample["img"] = augment_sensors(sample["img"], self.metric_depth[index], self.sensors)
        return sample

    def build_transforms(self, hyp: Any = None) -> Compose:
        transforms = super().build_transforms(hyp)

        def replace_geometry(composition: Compose) -> None:
            for index, transform in enumerate(composition.transforms):
                if isinstance(transform, Compose):
                    replace_geometry(transform)
                elif isinstance(transform, Mosaic):
                    composition.transforms[index] = FloatMosaic(self, imgsz=self.imgsz, p=hyp.mosaic, n=4)
                elif isinstance(transform, RandomPerspective):
                    composition.transforms[index] = FloatPerspective(
                        degrees=hyp.degrees, translate=hyp.translate, scale=hyp.scale,
                        shear=hyp.shear, perspective=hyp.perspective, size=(self.imgsz, self.imgsz))
                elif isinstance(transform, LetterBox):
                    composition.transforms[index] = FloatLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)
                elif transform.__class__.__name__ == "Albumentations":
                    # 通用库的uint8/三通道增强不允许碰深度；传感器增强在上述取样阶段完成。
                    composition.transforms[index] = Compose([])

        replace_geometry(transforms)
        return transforms


class QualityMosaic(Mosaic):
    """保留原生四图布局与标签变换，空白区只有RGB填114。"""

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or self.n != 4:
            raise ValueError("v19只支持带布局参数的四图Mosaic")
        canvas = np.zeros((self.imgsz * 2, self.imgsz * 2, 6), dtype=np.uint8)
        canvas[:, :, :3] = 114
        for item in params["layout"]:
            image = item["labels_patch"]["img"]
            canvas[item["y1a"]:item["y2a"], item["x1a"]:item["x2a"]] = image[item["y1b"]:item["y2b"], item["x1b"]:item["x2b"]]
        labels["img"] = canvas
        return labels


class QualityRandomPerspective(RandomPerspective):
    """复用原生随机矩阵和目标筛选，只替换图像插值与补边。"""

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None:
            raise ValueError("缺少同步几何变换参数")
        matrix, size = params["M"], params["size"]
        operation = (lambda plane, border: cv2.warpPerspective(plane, matrix, dsize=size, borderValue=border)) if self.perspective else (
            lambda plane, border: cv2.warpAffine(plane, matrix[:2], dsize=size, borderValue=border))
        labels["img"] = transform_quality_image(labels["img"], operation)
        labels["resized_shape"] = labels["img"].shape[:2]
        return labels


class QualityYOLODataset(MultimodalYOLODataset):
    """只为v19派生支持通道，磁盘data.yaml继续表达五个传感器通道。"""

    input_channels: int = 6

    def load_fused_image(self, index: int) -> np.ndarray:
        path = Path(self.im_files[index])
        return fuse_quality_modalities(path, self.infrared_dir / path.name, self.depth_dir / path.name)

    def resize_image(self, image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
        return resize_quality_image(image, size_wh)

    def resize_native_image(self, image: np.ndarray) -> np.ndarray:
        return resize_native_quality(image, self.imgsz)

    def build_transforms(self, hyp: Any = None) -> Compose:
        """沿用本机原生增强图，仅替换三个负责插值或填充的节点。"""
        transforms = super().build_transforms(hyp)

        def replace_geometry(composition: Compose) -> None:
            for index, transform in enumerate(composition.transforms):
                if isinstance(transform, Compose):
                    replace_geometry(transform)
                elif isinstance(transform, Mosaic):
                    composition.transforms[index] = QualityMosaic(self, imgsz=self.imgsz, p=hyp.mosaic, n=4)
                elif isinstance(transform, RandomPerspective):
                    composition.transforms[index] = QualityRandomPerspective(
                        degrees=hyp.degrees, translate=hyp.translate, scale=hyp.scale,
                        shear=hyp.shear, perspective=hyp.perspective, size=(self.imgsz, self.imgsz))
                elif isinstance(transform, LetterBox):
                    composition.transforms[index] = QualityLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)

        replace_geometry(transforms)
        return transforms


def learning_rate_factor(epoch: int, decay_epochs: int, final_ratio: float, cosine: bool) -> float:
    """按独立收敛周期衰减学习率，到达下限后保持不反弹。"""
    progress: float = min(max(epoch / max(decay_epochs, 1), 0.0), 1.0)
    remaining: float = (1.0 + math.cos(math.pi * progress)) / 2.0 if cosine else 1.0 - progress
    return final_ratio + (1.0 - final_ratio) * remaining


@dataclass(frozen=True)
class FusionRecipe:
    """保存融合路径和审计；早期融合分层学习率需显式启用，不改变历史默认。"""

    image_height: int = 1080
    image_width: int = 1920
    backbone_lr: float = 0.00002
    auxiliary_lr: float = 0.0002
    val_batch: int = 1
    min_delta: float = 0.0002
    data_audit: str = "./runs/dataset_cleaning/official_refresh_20260918_214843/manifest.json"
    architecture: str = "gated_v9"
    polish_scale: float | None = None
    polish_translate: float | None = None
    screening_thresholds: tuple[tuple[int, float], ...] = ()
    geometry: str = "fixed_rect"
    early_backbone_lr: float | None = None
    training_stage: str = "main"
    min_stop_epochs: int = 20
    initial_weights_sha256: str | None = None
    repeat_threshold: float = 0.0
    repeat_max_factor: float = 2.0
    repeat_extra_fraction: float = 0.10
    repeat_group_limit: int = 2
    repeat_ball_cap: float = 1.25
    split_stem: bool = False
    # 与epochs学习率日程解耦；None保持历史行为，预算结束仍保存当轮完整产物。
    budget_epochs: int | None = None
    continuous_depth: bool = False
    sensors: SensorAugment = SensorAugment()


def sampling_group(path: str) -> str:
    """以文件名序列及已审阅训练场景限额，不当作全量视觉去重。

    Args:
        path: 训练图片路径，分组不会读取图像或改动标签。
    """
    stem = Path(path).stem
    reviewed = (
        ("000021_015_00000001", "000021_024_00000085", "001142", "001144", "001145", "001148"),
        ("000021_004_00000129", "000666"),
        ("hehe_41_00000096", "shuming_435_00000005"),
    )
    for index, members in enumerate(reviewed):
        prefixes = {member.rsplit("_", 1)[0] for member in members if "_" in member}
        if stem in members or ("_" in stem and stem.rsplit("_", 1)[0] in prefixes):
            return f"reviewed:{index}"
    return f"sequence:{stem.rsplit('_', 1)[0]}" if "_" in stem else f"single:{stem}"


class BoundedRepeatSampler(torch.utils.data.Sampler[int]):
    """保留全量基本索引，以固定额外数量执行带场景上限的长尾采样。

    Note:
        倍率只用于选择额外主样本，Mosaic附加图不受每图两次的主索引上限约束。
        固定轮长避免原生训练器缓存的nb与预热、梯度累积步号不一致。
    """

    def __init__(self, dataset: MultimodalYOLODataset, recipe: FusionRecipe, seed: int) -> None:
        self.names = [Path(path).name for path in dataset.im_files]
        self.lookup = {name: index for index, name in enumerate(self.names)}
        if len(self.lookup) != len(self.names):
            raise ValueError("受限采样要求训练图片文件名唯一")
        self.classes = [tuple(sorted({int(c) for c in row["cls"].reshape(-1)})) for row in dataset.labels]
        self.counts = Counter(c for classes in self.classes for c in classes)
        size = len(self.names)
        self.class_factors = [min(recipe.repeat_max_factor, max(1.0, math.sqrt(
            recipe.repeat_threshold * size / self.counts[c]))) if self.counts[c] else 1.0
                              for c in range(len(CLASS_NAMES))]
        self.class_factors[7] = min(self.class_factors[7], recipe.repeat_ball_cap)
        self.factors = [min(max((self.class_factors[c] for c in classes), default=1.0),
                            recipe.repeat_ball_cap if 7 in classes else recipe.repeat_max_factor)
                        for classes in self.classes]
        self.groups = [sampling_group(path) for path in dataset.im_files]
        self.weights = [factor - 1.0 for factor in self.factors]
        self.group_limit, self.seed = recipe.repeat_group_limit, seed
        demand: dict[str, float] = {}
        for group, weight in zip(self.groups, self.weights, strict=True):
            demand[group] = demand.get(group, 0.0) + weight
        self.extra_count = min(math.floor(size * recipe.repeat_extra_fraction),
                               math.floor(sum(min(value, self.group_limit) for value in demand.values())))
        self.epoch: int = -1
        self.indices: list[int] = []
        self.extras: list[int] = []
        self.seen: Counter[int] = Counter()
        self.target_boxes: list[int] = [0] * len(CLASS_NAMES)

    def __len__(self) -> int:
        """返回固定的每轮主样本数，预取不会改变轮长。"""
        return len(self.names) + self.extra_count

    def set_epoch(self, epoch: int) -> None:
        """按真实轮次重新选择额外图片，断点恢复不依赖预取次数。"""
        self.epoch = epoch
        rng = random.Random(self.seed + epoch)
        priorities = sorted((-math.log(max(rng.random(), 1e-12)) / weight, i)
                            for i, weight in enumerate(self.weights) if weight > 0)
        group_counts: Counter[str] = Counter()
        self.extras = []
        for _, index in priorities:
            if len(self.extras) == self.extra_count:
                break
            group = self.groups[index]
            if group_counts[group] < self.group_limit:
                self.extras.append(index)
                group_counts[group] += 1
        if len(self.extras) != self.extra_count:
            raise RuntimeError("采样候选容量不足，拒绝通过额外重复绕过场景上限")
        self.indices = list(range(len(self.names))) + self.extras
        rng.shuffle(self.indices)
        self.seen.clear()
        self.target_boxes = [0] * len(CLASS_NAMES)

    def __iter__(self) -> Iterator[int]:
        """仅产生当前轮，不提前准备下一轮计划。"""
        if self.epoch < 0:
            raise RuntimeError("开始读取训练批次前必须设置采样轮次")
        return iter(self.indices)

    def record_batch(self, batch: dict[str, Any]) -> None:
        """在CPU批次转设备前记录实际主图与增强后框数，不增加模型前向。"""
        self.seen.update(self.lookup[Path(path).name] for path in batch["im_file"])
        counts = torch.bincount(batch["cls"].reshape(-1).to(dtype=torch.int64), minlength=len(CLASS_NAMES)).tolist()
        self.target_boxes = [a + b for a, b in zip(self.target_boxes, counts, strict=True)]

    def write_plan(self, output: Path) -> None:
        """将频率、限额与文件分组写入运行目录，不修改数据集。"""
        payload = {"policy": "bounded_repeat_v17", "base_images": len(self.names), "extra_images": self.extra_count,
                   "epoch_images": len(self), "group_limit": self.group_limit,
                   "count_scope": "primary_indices_only_not_mosaic_source_appearances",
                   "classes": [{"id": c, "name": name, "images": self.counts[c], "factor": self.class_factors[c]}
                               for c, name in enumerate(CLASS_NAMES)],
                   "images": [{"name": name, "classes": self.classes[i], "group": self.groups[i], "factor": self.factors[i]}
                              for i, name in enumerate(self.names)]}
        (output / "sampling_plan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_epoch(self, output: Path, attempt: int) -> None:
        """记录本轮尝试的额外文件与完整索引摘要，失败重试不冒充已完成。"""
        digest = hashlib.sha256(",".join(map(str, self.indices)).encode("utf-8")).hexdigest()
        row = {"epoch": self.epoch + 1, "attempt": attempt, "planned_images": len(self),
               "extra_images": [self.names[i] for i in self.extras], "indices_sha256": digest}
        with (output / "sampling_epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def finish_epoch(self, output: Path, attempt: int) -> None:
        """核对实际读入索引，逐类写出完整轮次的曝光记录。"""
        if self.seen != Counter(self.indices):
            raise RuntimeError("本轮实际训练主样本与采样计划不一致，拒绝写入完成记录")
        actual = Counter(c for i, times in self.seen.items() for _ in range(times) for c in self.classes[i])
        append_csv(output / "sampling_history.csv", [
            {"epoch": self.epoch + 1, "attempt": attempt, "class_id": c, "name": name,
             "base_images": self.counts[c], "proposal_factor": self.class_factors[c],
             "planned_anchor_images": sum(c in self.classes[i] for i in self.indices),
             "consumed_anchor_images": actual[c], "augmented_target_boxes": self.target_boxes[c],
             "epoch_anchor_images": len(self), "extra_anchor_images": self.extra_count}
            for c, name in enumerate(CLASS_NAMES)])


class EpochDataLoader(torch.utils.data.DataLoader):
    """有限轮次加载器，保留持久worker但不跨轮预取旧采样计划。"""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """延迟创建迭代器，避免原生进度条两次enumerate触发多余预取。"""
        yield from super().__iter__()

    def close(self) -> None:
        """释放持久worker，关拼图后新进程会读取更新后的增强配置。"""
        iterator = getattr(self, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            self._iterator = None

    def reset(self) -> None:
        """兼容原生训练器的关Mosaic生命周期，不立即启动预取。"""
        self.close()


def copy_checkpoint_to_cpu(value: Any) -> Any:
    """递归复制优化器状态至CPU，不在GPU上复制Adam动量。"""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: copy_checkpoint_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_checkpoint_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_checkpoint_to_cpu(item) for item in value)
    return deepcopy(value)


def copy_ema_to_cpu(model: torch.nn.Module) -> torch.nn.Module:
    """利用deepcopy备忘录直接构建CPU模型，保持训练中的EMA设备与参数不变。"""
    memo: dict[int, Any] = {}
    criterion = getattr(model, "criterion", None)
    if criterion is not None:
        memo[id(criterion)] = None
    # 包括检测头动态生成但未注册的anchors/strides，避免其在GPU被深拷贝。
    for module in model.modules():
        tensors = list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))
        tensors.extend(item for item in vars(module).values() if isinstance(item, torch.Tensor))
        for tensor in tensors:
            if id(tensor) not in memo:
                copied = tensor.detach().to(device="cpu", copy=True)
                memo[id(tensor)] = (torch.nn.Parameter(copied, requires_grad=tensor.requires_grad)
                                    if isinstance(tensor, torch.nn.Parameter) else copied)
    result = deepcopy(model, memo).float().to(memory_format=torch.contiguous_format)
    result.criterion = None
    return result


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


def save_validation_artifacts(metrics: DetMetrics, confusion: ConfusionMatrix, output: Path, epoch: int) -> None:
    """复用已完成验证的指标保存同轮曲线及混淆矩阵，不再次运行网络。

    Args:
        metrics: 本轮已计算的检测指标与曲线数组。
        confusion: 本轮累计的混淆矩阵。
        output: 最佳权重或显式本地验证的专用目录。
        epoch: 对应权重的训练轮次。
    """
    output.mkdir(parents=True, exist_ok=True)
    report = {"epoch": epoch, "evaluation_protocol": EVALUATION_PROTOCOL,
              "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
              "per_class": [{key: value.item() if isinstance(value, np.generic) else value for key, value in row.items()}
                            for row in metrics.summary()]}
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    names = {i: metrics.names[int(category)] for i, category in enumerate(metrics.ap_class_index)}
    if len(names):
        box = metrics.box
        plot_pr_curve(box.px, box.prec_values, box.all_ap, output / "BoxPR_curve.png", names)
        for label, curve in (("F1", box.f1_curve), ("P", box.p_curve), ("R", box.r_curve)):
            plot_mc_curve(box.px, curve, output / f"Box{label}_curve.png", names, ylabel=label)
    for normalize in (False, True):
        confusion.plot(normalize=normalize, save_dir=output)


class RectangularDataset(MultimodalYOLODataset):
    """原尺寸读取三模态，所有批次输出相同的步长补齐矩形。"""

    def __init__(self, *args: Any, content_hw: tuple[int, int], audit_digest: str,
                 polish_scale: float = 0.0, polish_translate: float = 0.0, **kwargs: Any) -> None:
        self.content_hw = content_hw
        self.audit_digest = audit_digest
        self.polish = False
        self.hyp = kwargs["hyp"]
        super().__init__(*args, polish_scale=polish_scale, polish_translate=polish_translate, **kwargs)

    def get_cache_hash(self) -> str:
        """将审计指纹加入标签缓存键，防止复用旧配方的标签缓存。"""
        value = f"{super().get_cache_hash()}|{FUSION_VERSION}|{self.audit_digest}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def build_transforms(self, hyp: Any = None) -> Compose:
        """增强与格式化由本类统一完成，避免原生流程重新变成正方形。"""
        return Compose([])

    def close_mosaic(self, hyp: Any) -> None:
        """按配方关闭拼图；v10保留轻度几何扰动，历史v9默认归零。"""
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
            scale_range = self.polish_scale if self.polish else self.hyp.scale
            translate_range = self.polish_translate if self.polish else self.hyp.translate
            if scale_range or translate_range:
                scale = random.uniform(1 - scale_range, 1 + scale_range)
                dx = w * ((1 - scale) / 2 + random.uniform(-translate_range, translate_range))
                dy = h * ((1 - scale) / 2 + random.uniform(-translate_range, translate_range))
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
                "ori_shape": original_hw, "resized_shape": (h, w), "ratio_pad": ((sy, sx), (left, top))}


class RectangularValidator(DetectionValidator):
    """使用FP32单标签NMS，兼容原生矩形与历史固定矩形画布。"""

    clip_content: bool = True

    def __call__(self, trainer: Any = None, model: Any = None, **kwargs: Any) -> dict[str, float]:
        if trainer is None:
            if self.dataloader is None or not isinstance(self.dataloader.dataset, MultimodalYOLODataset):
                raise ValueError("独立本地验证须显式提供三模态验证加载器，不自动读取官方测试集")
            self.epoch = 0
            return super().__call__(trainer=None, model=model, **kwargs)
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
        # 与提交相同，先裁真实内容区；框架ratio_pad按高、宽顺序记录缩放。
        items = zip(preds, batch["ori_shape"], batch["ratio_pad"], strict=True) if self.clip_content else ()
        for prediction, original_hw, ratio_pad in items:
            (sy, sx), (left, top) = ratio_pad
            boxes, keep = clip_canvas_boxes(prediction["bboxes"], (sx, sy, left, top), original_hw)
            for key in prediction:
                prediction[key] = prediction[key][keep]
            prediction["bboxes"] = boxes
        super().update_metrics(preds, batch)
        # 普通轮次也累计混淆矩阵，最佳轮可直接存图而无需额外模型前向。
        if not self.args.plots:
            for index, prediction in enumerate(preds):
                self.confusion_matrix.process_batch(prediction, self._prepare_batch(index, batch),
                                                    conf=self.confusion_matrix_conf)
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
    """按明确的最低轮数控制耐心早停，不影响真实最佳权重保存。"""

    def __init__(self, polish_start: int, patience: int, min_delta: float,
                 min_epochs: int = MIN_POLISH_EPOCHS) -> None:
        if type(min_epochs) is not int or min_epochs < 1:
            raise ValueError("早停最低轮数必须是正整数")
        self.polish_start, self.patience, self.min_delta = polish_start, patience, min_delta
        self.min_epochs = min_epochs
        self.best = -math.inf
        self.wait = 0
        self.possible_stop = False

    def __call__(self, epoch: int, fitness: float) -> bool:
        if self.patience == 0:
            self.possible_stop = False
            return False
        if epoch < self.polish_start:
            return False
        if fitness > self.best + self.min_delta:
            self.best, self.wait = fitness, 0
        else:
            self.wait += 1
        allowed = epoch >= self.polish_start + self.min_epochs - 1
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


class FusionDetectionTrainer(DetectionTrainer):
    """沿用原生YOLO训练循环，显式配置融合结构、矩形数据与指标记录。"""

    def __init__(self, *args: Any, recipe: FusionRecipe, **kwargs: Any) -> None:
        self.recipe = recipe
        if recipe.architecture not in {"gated_v9", "early_v10", "quality_v19", "rgb_v20"}:
            raise ValueError("未知三模态融合配方")
        if recipe.geometry not in {"fixed_rect", "native_square"}:
            raise ValueError("未知输入几何配方")
        self.rgb_diagnostic = recipe.architecture == "rgb_v20"
        self.early_fusion = recipe.architecture in {"early_v10", "rgb_v20"}
        self.quality_fusion = recipe.architecture == "quality_v19"
        self.native_square = recipe.geometry == "native_square"
        self.continuous_depth = recipe.continuous_depth
        self.recipe_version = EARLY_FUSION_VERSION if self.early_fusion else FUSION_VERSION
        if recipe.split_stem:
            self.recipe_version = SPLIT_STEM_VERSION
        if self.quality_fusion:
            self.recipe_version = QUALITY_FUSION_VERSION
        if self.rgb_diagnostic:
            self.recipe_version = RGB_DIAGNOSTIC_VERSION
        self.requested = dict(kwargs.get("overrides") or {})
        self.resume_metadata: dict[str, Any] | None = None
        self.parent_initialization: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)
        if self.continuous_depth:
            if (recipe.architecture != "early_v10" or not self.native_square or recipe.split_stem or
                    self.args.resume or recipe.training_stage != "main" or self.args.amp or self.args.cache or
                    self.args.augmentations is not None or any(getattr(self.args, key) != 0 for key in
                    ("mixup", "cutmix", "copy_paste", "hsv_h", "hsv_s", "hsv_v", "bgr"))):
                raise ValueError("新浮点训练要求早期融合/原生方形/FP32/官方基底，不恢复历史训练或叠加非传感器颜色增强")
            configure_fp32()
        if self.args.imgsz != recipe.image_width or self.args.rect or self.args.multi_scale:
            raise ValueError("imgsz须等于配方宽度；训练rect和multi_scale须关闭")
        expected_hw = (self.args.imgsz, self.args.imgsz) if self.native_square else (1080, 1920)
        if (recipe.image_height, recipe.image_width) != expected_hw:
            raise ValueError(f"输入几何要求内容高宽为{expected_hw}")
        if self.args.batch < 1 or self.args.nbs % self.args.batch or self.args.cache:
            raise ValueError("batch须为有效批次nbs的正整数因子，且cache=False")
        self._validate_training_stage()
        if recipe.budget_epochs is not None and (
                type(recipe.budget_epochs) is not int or not 1 <= recipe.budget_epochs <= self.args.epochs):
            raise ValueError("训练预算轮次须为[1,epochs]中的整数；不会改变学习率日程")
        if self.rgb_diagnostic and (not self.native_square or recipe.split_stem or
                recipe.training_stage != "main" or recipe.repeat_threshold != 0 or
                recipe.budget_epochs is None or recipe.screening_thresholds or
                recipe.early_backbone_lr is None or not self.args.val or not self.args.save or
                self.args.cls_pw != 0 or self.args.augmentations is not None or
                any(getattr(self.args, key) != 0 for key in ("hsv_h", "hsv_s", "hsv_v", "bgr"))):
            raise ValueError("RGB诊断要求原生main、显式预算/骨干lr、无采样/类别加权/颜色增强/AP门槛且保存验证结果")
        if (not self.early_fusion and self.args.cls_pw != 0) or self.args.optimizer != "AdamW" or self.args.nms is not True:
            raise ValueError("融合配方要求AdamW与nms=True；v9另外要求cls_pw=0")
        for value, maximum in ((recipe.polish_scale, self.args.scale), (recipe.polish_translate, self.args.translate)):
            if value is not None and not 0 <= value <= maximum:
                raise ValueError("收尾尺度/位移须非负，且不能强于主训练阶段；None表示保留原值")
        if any(epoch < 2 or epoch > self.args.epochs or not 0 < threshold < 1 for epoch, threshold in recipe.screening_thresholds):
            raise ValueError("阶段筛选轮次须位于[2,epochs]，AP95门槛须位于(0,1)")
        screening_epochs = [epoch for epoch, _ in recipe.screening_thresholds]
        if screening_epochs != sorted(set(screening_epochs)):
            raise ValueError("阶段筛选轮次须严格递增且不能重复")
        if recipe.early_backbone_lr is not None and (not (self.early_fusion or self.quality_fusion) or
                not 0 < recipe.early_backbone_lr <= self.args.lr0):
            raise ValueError("early_backbone_lr仅用于原生早期/质量融合，且须为不超过lr0的正数")
        if self.quality_fusion and (not self.native_square or recipe.split_stem or recipe.training_stage != "main" or
                recipe.repeat_threshold != 0 or recipe.early_backbone_lr is None or self.args.compile or
                not math.isfinite(recipe.auxiliary_lr) or recipe.auxiliary_lr <= 0 or
                self.args.augmentations is not None or any(getattr(self.args, key) != 0 for key in (
                    "mixup", "cutmix", "copy_paste", "hsv_h", "hsv_s", "hsv_v", "bgr"))):
            raise ValueError("v19要求原生main、显式骨干/辅助学习率、无拆分首层/采样/颜色/混合增强/编译")
        if recipe.split_stem and (not self.early_fusion or not self.native_square or
                recipe.training_stage != "main" or recipe.repeat_threshold != 0 or self.args.cls_pw != 0 or
                self.args.compile or recipe.early_backbone_lr is None or
                not math.isfinite(recipe.auxiliary_lr) or recipe.auxiliary_lr <= 0):
            raise ValueError("v18拆分首层要求原生早期融合main、无采样/类别加权、显式骨干及正辅助学习率")
        if self.data["names"] != dict(enumerate(CLASS_NAMES)) or self.data["channels"] != 5:
            raise ValueError("融合训练必须使用比赛原顺序的12类、五通道数据配置")
        if (not 0 <= recipe.repeat_threshold <= 1 or not 1 <= recipe.repeat_max_factor <= 2 or
                not 0 <= recipe.repeat_extra_fraction <= 0.10 or
                type(recipe.repeat_group_limit) is not int or recipe.repeat_group_limit < 1 or
                not 1 <= recipe.repeat_ball_cap <= recipe.repeat_max_factor):
            raise ValueError("采样频率须在[0,1]，倍率[1,2]，额外比例[0,0.1]，组限额为正整数，球类上限不超总倍率")
        if recipe.repeat_threshold and (not self.native_square or not self.early_fusion or
                recipe.training_stage != "main" or self.args.cls_pw != 0 or self.args.compile or
                any(getattr(self.args, key) != 0 for key in ("mixup", "cutmix", "copy_paste"))):
            raise ValueError("受限采样仅支持原生五通道main训练、cls_pw=0、compile=False且不叠加混合/粘贴")
        self.audit = self._verify_audit()
        self.best_map50 = -math.inf
        self.last_aux_grad: torch.Tensor | None = None
        self.sampling_attempts: Counter[int] = Counter()
        self.stem_update_sums: dict[str, torch.Tensor] = {}
        self.stem_update_steps: int = 0
        if recipe.split_stem:
            self.add_callback("on_train_epoch_start", self._start_stem_diagnostics)
        if recipe.repeat_threshold:
            self.add_callback("on_train_epoch_start", self._start_sampling_epoch)

    def _start_stem_diagnostics(self, trainer: Any) -> None:
        """只标记正式首批前向，不为诊断创建额外批次。"""
        stem = unwrap_model(self.model).model[0]
        if not isinstance(stem, SplitModalStem):
            raise ValueError("v18配方与实际首层结构不一致")
        stem.response_rms.clear()
        stem.capture_response = True
        self.stem_update_sums.clear()
        self.stem_update_steps = 0

    def _start_sampling_epoch(self, trainer: Any) -> None:
        """正式轮次开始时设置计划，有限加载器尚未预取当前轮。"""
        sampler = self.train_loader.sampler
        if not isinstance(sampler, BoundedRepeatSampler):
            raise RuntimeError("已启用受限采样但训练加载器未使用对应采样器")
        sampler.set_epoch(self.epoch)
        self.sampling_attempts[self.epoch] += 1
        sampler.write_epoch(self.save_dir, self.sampling_attempts[self.epoch])
        LOGGER.info(f"第{self.epoch + 1}轮受限采样：{len(sampler.names)}张基本主样本 + "
                    f"{sampler.extra_count}张额外主样本；每图最多2次，每组额外≤{sampler.group_limit}")

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """复用训练批次记录采样，不改变图像归一化或标签。"""
        if self.rgb_diagnostic and (batch["img"].ndim != 4 or batch["img"].shape[1] != 3):
            raise ValueError("RGB诊断网络只接收NCHW三通道，不能误传旧五通道或v19支持张量")
        if self.recipe.repeat_threshold:
            self.train_loader.sampler.record_batch(batch)
        return super().preprocess_batch(batch)

    def _validate_training_stage(self) -> None:
        """只对显式短收尾开放全程无拼图，保留主训练的历史日程保护。"""
        recipe, args = self.recipe, self.args
        if recipe.training_stage not in {"main", "polish"}:
            raise ValueError("训练阶段只允许main或polish")
        if type(recipe.min_stop_epochs) is not int or not 1 <= recipe.min_stop_epochs <= args.epochs:
            raise ValueError("早停最低轮数须位于[1,epochs]")
        if recipe.training_stage == "main":
            if args.epochs - args.close_mosaic < 5 or args.close_mosaic < MIN_POLISH_EPOCHS:
                raise ValueError("主训练须留至少5轮主训练及20轮收尾")
            if recipe.initial_weights_sha256 is not None:
                raise ValueError("父权重摘要只用于polish阶段")
            return
        if recipe.architecture != "early_v10" or recipe.geometry != "native_square":
            raise ValueError("polish阶段仅支持原生方形五通道早期融合")
        if any(getattr(args, key) != 0 for key in ("mosaic", "close_mosaic", "mixup", "cutmix", "copy_paste")):
            raise ValueError("polish阶段须从第一轮关闭全部拼图与混合增强")
        if args.cls_remap or args.cls_pw != 0:
            raise ValueError("polish继承同12类权重，不重新映射或添加类别频率权重")
        digest = recipe.initial_weights_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("polish阶段必须显式锁定父权重SHA256")

    def _verify_audit(self) -> dict[str, Any]:
        """开训核验清单、标签及三模态内容指纹，不重做数据清洗。"""
        path = Path(self.recipe.data_audit).resolve()
        audit = json.loads(path.read_text(encoding="utf-8"))
        root = Path(self.args.data).resolve().parent
        if audit.get("status") != "applied" or len(audit.get("samples", [])) != 2000:
            raise ValueError("清洗审计未落位或不是官方2000组")
        if len({row["image"] for row in audit["samples"]}) != 2000 or any(
                row["after_split"] not in {"train", "val"} for row in audit["samples"]):
            raise ValueError("清洗审计存在重复图像或非法划分")
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
        LOGGER.info("开训来源审计：核对2000组、6000张三模态图像内容，不修改数据")
        verify_source_images(root, audit["samples"])
        return {"manifest_sha256": file_hash(path), "yaml_sha256": file_hash(Path(self.args.data)),
                "source_images_verified": True,
                "counts": audit["counts"], "source": self.recipe.data_audit}

    def check_resume(self, overrides: dict[str, Any]) -> None:
        super().check_resume(overrides)
        if self.resume:
            # 恢复写新运行目录，避免覆盖历史日志和权重。
            self.args.project = overrides["project"]
            self.args.name = overrides["name"] + "_resume"
            self.args.exist_ok = False
            self.args.save_dir = None

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> FusionDetectionModel | EarlyFusionDetectionModel:
        """分离官方迁移、最佳权重微调和同配方中断恢复，不混用优化状态。"""
        if weights is None:
            raise ValueError("请从本地官方YOLO26l预训练权重创建融合模型")
        source = (weights.get("ema") or weights["model"]) if isinstance(weights, dict) else weights
        model_type = (RGBDiagnosticModel if self.rgb_diagnostic else QualityFusionDetectionModel if self.quality_fusion
                      else EarlyFusionDetectionModel if self.early_fusion else FusionDetectionModel)
        model = self.set_model_names_for_load(model_type(cfg or source.yaml, self.data["nc"], verbose))
        if self.recipe.training_stage == "polish":
            # cls_remap=False只禁止重排分类权重，不能把真实类名留为构建器的数字占位名。
            model.names = deepcopy(self.data["names"])
        source_version = getattr(source, "early_fusion_version", None) or getattr(source, "fusion_version", None)
        if source_version:
            if source_version != self.recipe_version:
                raise ValueError("融合检查点结构版本与当前配方不一致")
            if self.quality_fusion and getattr(source, "preprocess_version", None) != QUALITY_PREPROCESS_VERSION:
                raise ValueError("v19断点缺少一致的深度支持预处理版本")
            if not self.resume and self.recipe.training_stage == "polish":
                self.parent_initialization = self._load_polish_weights(model, source)
                model.content_hw = (self.recipe.image_height, self.recipe.image_width)
                return model
            if not self.resume:
                raise ValueError("融合检查点仅支持显式polish新阶段或同配方断点恢复")
            self.resume_metadata = deepcopy(getattr(source, "fusion_training", None))
            if not self.resume_metadata or self.resume_metadata.get("completed"):
                raise ValueError("该融合检查点已完成或缺少恢复元数据")
            self.parent_initialization = deepcopy(self.resume_metadata["signature"].get("parent_initialization"))
            if self.recipe.training_stage == "polish" and (
                    not self.parent_initialization or
                    self.parent_initialization.get("weights_sha256") != self.recipe.initial_weights_sha256):
                raise ValueError("polish断点缺少一致的原始父权重记录")
        else:
            if self.resume or self.recipe.training_stage == "polish" or source.model[0].conv.in_channels != 3:
                raise ValueError("首训只接受官方RGB预训练，不能恢复旧五通道或D-FINE模型")
        load_source = copy(source)
        load_source.names = {key: "ball" if str(name).lower() == "sports ball" else name for key, name in source.names.items()}
        if self.quality_fusion and self.resume:
            model.load_state_dict(load_source.float().state_dict(), strict=True)
            model.pt_path = getattr(load_source, "pt_path", None)
        elif self.recipe.split_stem and self.resume:
            # 新参数路径必须在加载前建立，不用宽松交集悄悄丢弃首层。
            model.split_modal_stem()
            model.load_state_dict(load_source.float().state_dict(), strict=True)
            model.pt_path = getattr(load_source, "pt_path", None)
        else:
            model.load(load_source, verbose=verbose)
            if self.recipe.split_stem:
                model.split_modal_stem()
            if self.quality_fusion:
                model.initialize_auxiliary_from_rgb()
                LOGGER.info("v19迁移：IR第0–4层继承RGB（首层通道求和）；Depth首层32个值滤波器继承RGB，支持权重零初始化可训练")
        model.content_hw = (self.recipe.image_height, self.recipe.image_width)
        if self.continuous_depth:
            model.preprocess_version = FLOAT_PREPROCESS_VERSION
        if self.rgb_diagnostic:
            LOGGER.info("v20实际网络首层输入3通道；构建期间显示的五通道摘要仅为保留v14初始化顺序，非最终结构")
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

    def _load_polish_weights(self, model: EarlyFusionDetectionModel, source: torch.nn.Module) -> dict[str, Any]:
        """严格继承同划分五通道模型，返回父来源记录；不加载优化器状态。

        Args:
            model: 新建且尚未开始优化的目标检测器。
            source: 从指定本地检查点加载的父模型。

        Returns:
            可写入配方与断点的父权重摘要、轮次和原始训练签名。

        Raises:
            ValueError: 类型、输入几何、类别、审计或指定文件摘要不匹配。
        """
        if not isinstance(source, EarlyFusionDetectionModel) or not isinstance(model, EarlyFusionDetectionModel):
            raise ValueError("polish父权重必须是五通道早期融合模型")
        signature = getattr(source, "fusion_training", {}).get("signature", {})
        expected_hw = (self.recipe.image_height, self.recipe.image_width)
        if (getattr(source, "early_fusion_version", None) != self.recipe_version or
                signature.get("evaluation_protocol") != EVALUATION_PROTOCOL or
                signature.get("recipe", {}).get("geometry") != "native_square" or
                tuple(source.content_hw) != expected_hw or source.model[0].conv.in_channels != 5):
            raise ValueError("polish父权重结构、几何或评估协议不一致")
        if source.names != dict(enumerate(CLASS_NAMES)) or model.names != source.names:
            raise ValueError("polish父权重类别及顺序不一致")
        if any(signature.get("audit", {}).get(key) != self.audit[key] for key in ("manifest_sha256", "yaml_sha256")):
            raise ValueError("polish父权重使用的数据审计不一致，拒绝跨划分初始化")
        path = Path(self.args.model).resolve(strict=True)
        if Path(getattr(source, "pt_path", "")).resolve() != path:
            raise ValueError("已加载模型与指定父权重文件不一致")
        digest = file_hash(path)
        if digest != self.recipe.initial_weights_sha256:
            raise ValueError("polish父权重SHA256不一致")
        # 文件摘要已锁定，只读取可信本地检查点的轮次；不向新优化器恢复其状态。
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        epoch = int(checkpoint["epoch"]) + 1
        del checkpoint
        if epoch < 1:
            raise ValueError("polish父权重缺少可审计的训练轮次")
        model.load_state_dict(source.float().state_dict(), strict=True)
        LOGGER.info(f"完整继承五通道父权重：{path.name}，第{epoch}轮；新建优化器与训练日程")
        return {"kind": "finetune_existing_five_channel", "weights": str(path.relative_to(PROJECT_ROOT))
                if path.is_relative_to(PROJECT_ROOT) else path.name, "weights_sha256": digest,
                "epoch": epoch, "source_signature": deepcopy(signature)}

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> MultimodalYOLODataset:
        if self.native_square:
            dataset_type = (RGBDiagnosticDataset if self.rgb_diagnostic else
                            FloatYOLODataset if self.continuous_depth else
                            QualityYOLODataset if self.quality_fusion else MultimodalYOLODataset)
            return dataset_type(
                img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
                augment=mode == "train", hyp=copy(self.args), rect=False, cache=False,
                single_cls=False, stride=32, pad=0.0, prefix=f"{mode}: ", task="detect",
                classes=None, data=self.data, fraction=1.0,
                polish_scale=self.recipe.polish_scale, polish_translate=self.recipe.polish_translate,
                **({"sensors": self.recipe.sensors} if self.continuous_depth else {}),
            )
        return RectangularDataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
                                  augment=mode == "train", hyp=copy(self.args), rect=False, cache=False,
                                  single_cls=False, stride=32, pad=0.0, prefix=f"{mode}: ", task="detect",
                                  classes=None, data=self.data, fraction=1.0,
                                  content_hw=(self.recipe.image_height, self.recipe.image_width),
                                  audit_digest=self.audit["manifest_sha256"],
                                  polish_scale=self.recipe.polish_scale or 0.0, polish_translate=self.recipe.polish_translate or 0.0)

    # 框架生命周期方法名必须与父类一致。
    def _setup_scheduler(self) -> None:
        """保留训练轮数上限，将实际学习率衰减限制在前 200 轮。"""
        self.lf = partial(
            learning_rate_factor,
            decay_epochs=min(self.epochs, LR_DECAY_EPOCHS),
            final_ratio=self.args.lrf,
            cosine=self.args.cos_lr,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def get_dataloader(
        self, dataset_path: str, batch_size: int = 16, rank: int = -1, mode: str = "train"
    ) -> InfiniteDataLoader | EpochDataLoader:
        """为本机单卡构建低预取加载器，连续浮点模式验证不另建进程池。

        Note:
            五通道增强曾出现 CPU 内存分配失败，因此每个进程仅预取一批，关闭锁页。
            当前实现限定单卡训练；Windows 多进程入口由 train1.py 保护。
        """
        if rank != -1:
            raise ValueError("当前三模态加载器面向本机单卡，请使用 device=0")
        if mode not in {"train", "val"}:
            raise ValueError(f"不支持的数据加载模式：{mode}")
        if mode == "train" and isinstance(getattr(self, "train_loader", None), EpochDataLoader):
            self.train_loader.close()
        batch_size = self.recipe.val_batch if mode == "val" else batch_size
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        batch_size = min(batch_size, len(dataset))
        shuffle: bool = mode == "train" and not dataset.rect
        workers: int = min(self.args.workers, math.ceil(len(dataset) / batch_size))
        if self.continuous_depth and mode == "val":
            # FP32验证逐张在主进程读图，避免Windows再常驻一整组PyTorch子进程。
            workers = 0
        generator = torch.Generator().manual_seed(self.args.seed)
        LOGGER.info(f"{mode}: 加载进程 {workers}，每进程预取 1 批，锁页内存关闭")
        if mode == "train" and self.recipe.repeat_threshold:
            sampler = BoundedRepeatSampler(dataset, self.recipe, self.args.seed)
            sampler.write_plan(self.save_dir)
            return EpochDataLoader(
                dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                prefetch_factor=1 if workers else None, persistent_workers=workers > 0,
                pin_memory=False, collate_fn=dataset.collate_fn, worker_init_fn=seed_worker,
                generator=generator, drop_last=False,
            )
        loader_type = EpochDataLoader if self.continuous_depth else InfiniteDataLoader
        return loader_type(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            prefetch_factor=1 if workers else None,
            pin_memory=False,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=bool(self.args.compile and mode == "train"),
            **({"persistent_workers": workers > 0} if self.continuous_depth else {}),
        )

    def build_optimizer(self, model: torch.nn.Module, name: str = "AdamW", lr: float = 0.001,
                        momentum: float = 0.9, decay: float = 1e-5, iterations: float = 1e5) -> torch.optim.Optimizer:
        """将参数按主干/检测器/新增分支及衰减规则分组，预热保持学习率比例。"""
        if self.early_fusion or self.quality_fusion:
            optimizer = super().build_optimizer(model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations)
            if self.recipe.early_backbone_lr is None:
                return optimizer
            # 降低第1–10层更新速度；v18另外分出新增IR/Depth首层参数。
            native = unwrap_model(model)
            backbone_ids = {id(p) for module in native.model[1:11] for p in module.parameters()}
            auxiliary_ids = ({id(p) for module in (native.model[0].infrared, native.model[0].depth)
                              for p in module.parameters()} if self.recipe.split_stem else set())
            if self.quality_fusion:
                backbone_ids.update(id(p) for p in native.ir_encoder.parameters())
                auxiliary_ids = {id(p) for module in (native.depth_encoder, native.ir_fusion, native.depth_fusion)
                                 for p in module.parameters()}
            auxiliary_role = "quality_auxiliary" if self.quality_fusion else "auxiliary_stem"
            rates = {"backbone": self.recipe.early_backbone_lr, "stem_neck_head": lr}
            if self.recipe.split_stem or self.quality_fusion:
                rates[auxiliary_role] = self.recipe.auxiliary_lr
            parameter_groups: list[dict[str, Any]] = []
            for group in optimizer.param_groups:
                for role, rate in rates.items():
                    members = [p for p in group["params"] if
                               ("backbone" if id(p) in backbone_ids else
                                auxiliary_role if id(p) in auxiliary_ids else "stem_neck_head") == role]
                    if members:
                        parameter_groups.append({**{key: value for key, value in group.items() if key != "params"},
                                                 "params": members, "lr": rate, "initial_lr": rate, "role": role})
            grouped_ids = [id(p) for group in parameter_groups for p in group["params"]]
            if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {id(p) for p in native.parameters()}:
                raise ValueError("分层优化组存在遗漏或重复参数，拒绝开始训练")
            LOGGER.info(f"{'质量融合（含IR预训练分支）' if self.quality_fusion else '早期融合'}分层AdamW：骨干lr={self.recipe.early_backbone_lr:g}，"
                        f"RGB首层/颈部/双头lr={lr:g}，"
                        f"新增辅助参数lr={rates.get(auxiliary_role, lr):g}，沿用原生偏置与归一化不衰减规则")
            return torch.optim.AdamW(parameter_groups, lr=lr, betas=(momentum, 0.999))
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
        validator = RectangularValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
        validator.clip_content = True
        return validator

    def _setup_train(self) -> None:
        super()._setup_train()
        if self.continuous_depth:
            configure_fp32()
        start = 1 if self.early_fusion or self.quality_fusion else self.epochs - self.args.close_mosaic + 1
        self.stopper = PolishEarlyStopping(start, self.args.patience, self.recipe.min_delta, self.recipe.min_stop_epochs)
        files = ("train1.py", "src/modalities.py", "src/yolo/aic/__init__.py", "src/yolo/aic/model.py", "src/yolo/aic/training.py", "src/yolo/aic/data.py")
        # main.py允许只改RESUME_PATH和资源参数，配方本身另行比较；组件源码不可偷偷变化。
        sources = {name: file_hash(PROJECT_ROOT / name) for name in files if name != "train1.py"}
        package = Path(ultralytics.__file__).parent
        framework = {"ultralytics": ultralytics.__version__, "torch": str(torch.__version__),
                     "files": {name: file_hash(package / name) for name in (
                         "nn/tasks.py", "nn/modules/head.py", "utils/loss.py", "utils/nms.py", "data/augment.py", "data/base.py",
                         "engine/trainer.py", "engine/validator.py", "models/yolo/detect/val.py")}}
        selected = ("epochs", "imgsz", "nbs", "lr0", "lrf", "warmup_epochs", "weight_decay", "mosaic", "scale", "translate",
                    "fliplr", "hsv_h", "hsv_s", "hsv_v", "box", "cls", "cls_pw", "dfl", "close_mosaic", "patience", "seed",
                    "amp", "nms", "conf", "iou", "max_det", "deterministic", "optimizer", "cos_lr", "momentum",
                    "warmup_bias_lr", "warmup_momentum", "freeze", "cls_remap", "rect", "multi_scale", "fraction",
                    "degrees", "shear", "perspective", "flipud", "bgr", "mixup", "cutmix", "copy_paste",
                    "copy_paste_mode", "single_cls", "classes", "agnostic_nms")
        settings = {key: getattr(self.args, key) for key in selected}
        self.training_signature = {"version": self.recipe_version, "evaluation_protocol": EVALUATION_PROTOCOL,
                                   "recipe": asdict(self.recipe), "hyp": settings,
                                   "audit": self.audit, "sources": sources, "framework": framework,
                                   "parent_initialization": self.parent_initialization}
        if self.continuous_depth:
            self.training_signature["float_preprocessing"] = {
                "version": FLOAT_PREPROCESS_VERSION, "host_dtype": "float32", "host_range": [0, 255],
                "network_range": [0, 1], "depth_mm_range": [0, 20000],
                "invalid_depth": "zero_or_above_20000", "padding": [114, 114, 114, 0, 0],
                "precision": "FP32_no_autocast_no_tf32", "checkpoint_dtype": "float32",
            }
        if self.rgb_diagnostic:
            self.training_signature["diagnostic"] = {
                "submission_candidate": False, "network_channels": ["R", "G", "B"],
                "data_pipeline": "v14_five_channel_augmentation_then_select_rgb",
                "schedule_epochs": self.epochs, "budget_epochs": self.recipe.budget_epochs,
                "comparison": "v14_same_split_first_20_epochs_not_final_ceiling",
                "train_batch": self.batch_size, "val_batch": self.recipe.val_batch,
            }
        if self.quality_fusion:
            self.training_signature["quality_preprocessing"] = {
                "version": QUALITY_PREPROCESS_VERSION, "input_channels": 6,
                "channel_order": ["R", "G", "B", "IR", "Depth", "Depth_support"],
                "png_valid_mm": [DEPTH_MIN_MM, DEPTH_MAX_MM], "jpg_support": "gray_gt_zero_proxy",
                "support_interpolation": "bilinear_fraction", "depth_interpolation": "support_weighted_bilinear",
                "fusion_level": "P3_neck_input_only", "ir_window": 3,
                "initialization": "official_rgb_ir_layers_0_to_4_depth_first_32_filters",
            }
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
        validation_geometry = "native_square" if self.native_square else "fixed_rect"
        (self.save_dir / "optimization_recipe.json").write_text(json.dumps(
            {**self.training_signature, "content_hw": [self.recipe.image_height, self.recipe.image_width],
             "tensor_hw": canvas_shape((self.recipe.image_height, self.recipe.image_width)),
             "validation_geometry": validation_geometry,
             "validation_padding": [114, 114, 114, 0, 0] if self.continuous_depth else [114] * 3 if self.rgb_diagnostic else [114, 114, 114, 0, 0, 0] if self.quality_fusion else [114] * 5 if self.native_square else [114, 114, 114, 0, 0],
             "polish_start_epoch": 1 if self.recipe.training_stage == "polish" else self.epochs - self.args.close_mosaic + 1,
             "head_loss_weights": [0.8, 0.2], "validation_precision": "FP32",
             "initialization": self.parent_initialization["kind"] if self.parent_initialization else
                               "official_rgb_only_diagnostic" if self.rgb_diagnostic else
                               "official_rgb_pretrained_ir_supported_depth" if self.quality_fusion else
                               "official_rgb_plus_trainable_zero_ir_depth" if self.early_fusion else "rgb_plus_auxiliary_encoders",
             "checkpoint_selection": "mAP50-95",
             "class_aliases": {} if self.parent_initialization else {"sports ball": "ball"},
             "optimizer_groups": [{"role": group.get("role", "all"), "kind": group.get("param_group", "native"),
                                   "initial_lr": group["initial_lr"], "weight_decay": group["weight_decay"],
                                   "parameters": sum(p.numel() for p in group["params"])}
                                  for group in self.optimizer.param_groups]}, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(self.recipe.data_audit, self.save_dir / "dataset_audit.json")
        code = self.save_dir / "code"
        code.mkdir(exist_ok=True)
        for name in files:
            destination = code / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT_ROOT / name, destination)
        structure = "RGB三通道诊断（非赛事提交）" if self.rgb_diagnostic else "质量感知P3融合v19" if self.quality_fusion else "分模态首层v18" if self.recipe.split_stem else "五通道首层" if self.early_fusion else "三分支门控"
        geometry = (f"原生方形{self.args.imgsz}×{self.args.imgsz}，{structure}" if self.native_square
                    else f"内容1920×1080，张量1920×1088，{structure}")
        LOGGER.info(f"融合输入：{geometry}；物理批次{self.batch_size}，有效批次{self.args.nbs}")
        if self.recipe.budget_epochs is not None:
            LOGGER.info(f"训练预算：第{self.recipe.budget_epochs}轮验证和保存后停止；学习率仍按{self.epochs}轮日程")

    def optimizer_step(self) -> None:
        if self.early_fusion:
            before = ({name: getattr(unwrap_model(self.model).model[0], name).weight.detach().clone()
                       for name in ("rgb", "infrared", "depth")} if self.recipe.split_stem else {})
            super().optimizer_step()
            for name, previous in before.items():
                current = getattr(unwrap_model(self.model).model[0], name).weight.detach()
                change = (current.float() - previous.float()).norm()
                self.stem_update_sums[name] = self.stem_update_sums.get(name, torch.zeros_like(change)) + change
            if before:
                self.stem_update_steps += 1
            return
        native = unwrap_model(self.model)
        gradients = [p.grad.detach().float().norm() for module in (native.ir_encoder, native.depth_encoder)
                     for p in module.parameters() if p.grad is not None]
        if gradients:
            self.last_aux_grad = torch.stack(gradients).norm()
        super().optimizer_step()

    def validate(self) -> tuple[dict[str, float], float]:
        if self.recipe.repeat_threshold:
            self.train_loader.sampler.finish_epoch(self.save_dir, self.sampling_attempts[self.epoch])
        metrics, _ = super().validate()
        fitness = float(metrics["metrics/mAP50-95(B)"])
        # 原生fitness当前也是AP95，显式锁定以免框架版本改变选择语义。
        self.best_fitness = max(self.best_fitness or 0.0, fitness)
        epoch = self.epoch + 1
        if self.recipe.budget_epochs is not None:
            exhausted = epoch >= self.recipe.budget_epochs
            append_csv(self.save_dir / "budget_history.csv", [{
                "epoch": epoch, "schedule_epochs": self.epochs, "budget_epochs": self.recipe.budget_epochs,
                "current_AP95": fitness, "best_AP95": self.best_fitness,
                "decision": "stop" if exhausted else "continue",
                "reason": "budget_exhausted" if exhausted else "within_budget",
            }])
            if exhausted:
                self.stop = True
                LOGGER.info(f"第{epoch}轮诊断预算结束：最佳AP95={self.best_fitness:.5f}；保存后停止，不判定模型最终上限")
        screening = dict(self.recipe.screening_thresholds)
        if (epoch := self.epoch + 1) in screening:
            threshold = screening[epoch]
            passed = math.isfinite(fitness) and self.best_fitness >= threshold
            append_csv(self.save_dir / "screening_history.csv", [{
                "epoch": epoch, "metric": "metrics/mAP50-95(B)", "current": fitness,
                "best_so_far": self.best_fitness, "threshold": threshold,
                "decision": "continue" if passed else "stop",
                "reason": "threshold_met" if passed else "non_finite_metric" if not math.isfinite(fitness) else "below_threshold",
            }])
            LOGGER.info(f"阶段筛选：截至第{epoch}轮最佳AP95={self.best_fitness:.5f}，"
                        f"门槛={threshold:.5f}，结果={'继续' if passed else '停止'}")
            if not passed:
                self.stop = True
        box = self.validator.metrics.box
        positions = {int(category): i for i, category in enumerate(box.ap_class_index)}
        rows = []
        for category, name in enumerate(CLASS_NAMES):
            i = positions.get(category)
            rows.append({"epoch": epoch, "class_id": category, "name": name,
                         "precision": float(box.p[i]) if i is not None else 0.0,
                         "recall": float(box.r[i]) if i is not None else 0.0,
                         "AP50": float(box.ap50[i]) if i is not None else None,
                         "AP50_95": float(box.ap[i]) if i is not None else None})
        append_csv(self.save_dir / "per_class_metrics.csv", rows)
        if fitness >= self.best_fitness and self.args.plots:
            save_validation_artifacts(self.validator.metrics, self.validator.confusion_matrix,
                                      self.save_dir / "best_validation", epoch)
        (self.save_dir / "validation_status.json").write_text(json.dumps(
            {"latest_epoch": epoch, "evaluation_protocol": EVALUATION_PROTOCOL,
             "best_artifacts": "best_validation", "root_plots": "latest_plotted_validation_not_necessarily_best"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        native = unwrap_model(self.model)
        loss = native.criterion
        diagnostics = {"epoch": self.epoch + 1, "aux_grad_norm": float(self.last_aux_grad) if self.last_aux_grad is not None else None}
        for branch in (() if self.early_fusion or self.quality_fusion else ("ir", "depth")):
            for level, block in zip((3, 4, 5), getattr(native, f"{branch}_fusion"), strict=True):
                diagnostics[f"{branch}_p{level}_gate"] = float(block.last_gate_mean) if block.last_gate_mean is not None else None
        if self.quality_fusion:
            for key, value in (("ir_p3_gate", native.ir_fusion.last_gate_mean),
                               ("ir_p3_center_weight", native.ir_fusion.last_center_weight),
                               ("depth_p3_gate", native.depth_fusion.last_gate_mean),
                               ("depth_support_mean", native.last_support_mean)):
                diagnostics[key] = float(value) if value is not None else None
        for key in sorted(loss.running):
            diagnostics[key] = float(loss.running[key]) / max(loss.batches, 1)
        if self.early_fusion:
            stem = native.model[0]
            weights = (torch.cat([getattr(stem, name).weight.detach().float()
                                  for name in ("rgb", "infrared", "depth")], dim=1)
                       if self.recipe.split_stem else stem.conv.weight.detach().float())
            channel_names = ("red", "green", "blue") if self.rgb_diagnostic else ("red", "green", "blue", "infrared", "depth")
            for channel, name in enumerate(channel_names):
                diagnostics[f"stem_{name}_weight_norm"] = float(weights[:, channel].norm())
            if self.recipe.split_stem:
                diagnostics["stem_optimizer_steps"] = self.stem_update_steps
                for name in ("rgb", "infrared", "depth", "sum"):
                    value = stem.response_rms.get(name)
                    diagnostics[f"stem_{name}_response_rms"] = float(value) if value is not None else None
                for name in ("rgb", "infrared", "depth"):
                    value = self.stem_update_sums.get(name)
                    diagnostics[f"stem_{name}_mean_update_norm"] = (
                        float(value) / self.stem_update_steps if value is not None and self.stem_update_steps else None)
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
        if self.continuous_depth:
            ema = copy_ema_to_cpu(self.ema.ema)
            if not all(torch.isfinite(value).all() for value in ema.state_dict().values()):
                raise FloatingPointError("EMA含非有限参数，拒绝伪造有效检查点")
            checkpoint = {"epoch": self.epoch, "best_fitness": self.best_fitness, "model": None,
                          "ema": ema, "updates": self.ema.updates,
                          "optimizer": copy_checkpoint_to_cpu(self.optimizer.state_dict()), "scaler": self.scaler.state_dict(),
                          "train_args": vars(self.args), "train_metrics": {**self.metrics, "fitness": self.fitness},
                          "train_results": self.read_results_csv(), "date": datetime.now().astimezone().isoformat(),
                          "version": ultralytics.__version__, "preprocess": FLOAT_PREPROCESS_VERSION,
                          "precision": "FP32", "license": "AGPL-3.0"}
            self.wdir.mkdir(parents=True, exist_ok=True)
            # 流式写临时文件再替换，省去整个BytesIO副本；失败时保留上一轮last。
            temporary = self.last.with_suffix(".pt.tmp")
            try:
                torch.save(checkpoint, temporary)
                temporary.replace(self.last)
            finally:
                temporary.unlink(missing_ok=True)
            if self.best_fitness == self.fitness:
                shutil.copy2(self.last, self.best)
            if self.save_period > 0 and (self.epoch + 1) % self.save_period == 0:
                shutil.copy2(self.last, self.wdir / f"epoch{self.epoch + 1}.pt")
            result = True
        else:
            result = super().save_model()
        if result and better50:
            shutil.copy2(self.last, self.wdir / "best_map50.pt")
        return result

    def final_eval(self) -> None:
        """不重复加载最佳权重复评；每轮已完成正式验证，保留完整检查点。"""
        LOGGER.info(f"训练结束；最佳权重 {self.best}，对应曲线与指标 {self.save_dir / 'best_validation'}；"
                    f"AP50候选 {self.wdir / 'best_map50.pt'}。未追加独立复评。")

    def train(self) -> None:
        handler = logging.FileHandler(self.save_dir / "train.log", encoding="utf-8")
        LOGGER.addHandler(handler)
        try:
            with (self.save_dir / "train.log").open("a", encoding="utf-8") as log:
                with redirect_stdout(LogStream(sys.stdout, log)), redirect_stderr(LogStream(sys.stderr, log)):
                    super().train()
        finally:
            for name in ("train_loader", "test_loader"):
                loader = getattr(self, name, None)
                if isinstance(loader, (EpochDataLoader, InfiniteDataLoader)):
                    try:
                        loader.close()
                    except Exception as error:
                        LOGGER.warning(f"关闭{name}时发生异常：{error}")
            LOGGER.removeHandler(handler)
            handler.close()
