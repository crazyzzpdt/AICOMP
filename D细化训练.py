"""将官方 D-FINE-L 接入本地 RGB、红外、深度五通道训练与预测。

由 main.py 启动训练，predict.py 复用模型加载、输入变换与坐标恢复。
不联网下载、不改写数据集，不运行独立评估；验证仅发生在用户启动的训练轮末。
"""

# 内置库
from collections import Counter
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from typing import TextIO

# 必须在第三方导入前设置，禁止训练和预测时自动联网或上传遥测。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

# 三方库
import cv2
from faster_coco_eval import COCO, COCOeval_faster
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.ops import box_convert
from tqdm import tqdm
import yaml

# 自己的模块
from 三模态训练 import adapt_rgb_stem_weights, fuse_modalities
from 准备三模态数据集 import CLASS_NAMES, IMAGE_SUFFIXES


# 固定官方源码版本，避免本机更新第三方仓库后静默改变训练行为。
DFINE_COMMIT: str = "956d1709314c2c6a4df6f34de232054578a7449f"
# 路径相对本项目解析，不依赖 IDE 当前目录。
PROJECT_ROOT: Path = Path(__file__).resolve().parent
# 366 输出槽包含 0 占位；这里直接写官方 1 起始 category_id，不再额外加一。
# 宽泛类别使用代表类别作初始化，不宣称定义完全相同；animal/uav 无直接对应时重新初始化。
OBJECTS365_ROWS: dict[str, tuple[int, str] | None] = {
    "person": (1, "Person"), "boat": (22, "Boat"), "animal": None,
    "seat": (3, "Chair"), "sign": (90, "Traffic Sign"), "bicycle": (47, "Bicycle"),
    "car": (6, "Car"), "ball": (157, "Other Balls"), "light": (7, "Lamp"),
    "garbage can": (45, "Trash bin Can"), "uav": None, "tricycle": (184, "Tricycle"),
}
# 与官方预训练一致采用 0–1 输入，不额外套用 ImageNet 均值和标准差。
PREPROCESS_VERSION: str = "rgbirdepth_uint8_letterbox_div255_v1"
# 区分本项目检查点与 Ultralytics 序列化模型，拒绝误加载 RGB 官方原权重。
CHECKPOINT_FORMAT: str = "aic_dfine_l_5ch_v1"


@dataclass(frozen=True)
class TrainingConfig:
    """保存入口显式给定的训练配方，用于复现、推理和同配方断点恢复。"""

    model: str
    data: str
    project: str
    name: str
    epochs: int
    imgsz: int
    batch: int
    effective_batch: int
    val_batch: int
    workers: int
    device: int
    lr0: float
    backbone_lr: float
    lrf: float
    warmup_epochs: int
    weight_decay: float
    clip_grad: float
    polish_epoch: int
    scale_min: float
    fliplr: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    conf: float
    max_det: int
    ema_decay: float
    ema_warmup: int
    seed: int
    save_period: int
    resume: str | None
    # 缺省关闭新增增强，兼容历史 v6 配方；新入口显式设置。
    crop_prob: float = 0.0
    crop_min: float = 0.75
    lowres_prob: float = 0.0
    lowres_min: float = 0.6
    # 旧配方不启用重复采样或优化步日程，避免历史断点静默改变行为。
    boat_repeat: float = 1.0
    garbage_repeat: float = 1.0
    repeat_extra_fraction: float = 0.10
    repeat_group_limit: int = 4
    lr_schedule: str = "legacy_batch"
    # 对应官方 D-FINECriterion，而非 YOLO 的 cls/box/dfl 超参数。
    loss_vfl: float = 1.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    loss_fgl: float = 0.15
    loss_ddf: float = 1.5
    # 历史配方默认关闭；v8 显式启用正式训练内早停及分域指标。
    patience: int = 0
    min_delta: float = 0.0005
    domain_metrics: bool = False
    data_audit: str | None = None


def check_source() -> Path:
    """检查已下载的固定版本官方源码；缺失时报告准备命令而不联网。"""
    source = PROJECT_ROOT / "vendor" / "D-FINE"
    if not (source / "src/core/yaml_config.py").is_file():
        raise FileNotFoundError("缺少 vendor/D-FINE，请先按 docs/D-FINE三模态实施方案.md 准备官方源码")
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
                           check=True, capture_output=True, text=True).stdout.strip()
    if revision != DFINE_COMMIT or dirty:
        raise ValueError(f"官方源码版本不符或被修改：{revision}；请使用文档指定的未修改版本")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def build_model(imgsz: int, training: bool = False) -> tuple[nn.Module, nn.Module | None]:
    """构建不联网的 12 类五通道 D-FINE-L，输入端不冻结。

    Args:
        imgsz: 正方形填充尺寸，必须为 32 的倍数。
        training: 是否同时创建官方匹配器与损失函数。

    Returns:
        五通道模型，以及训练时使用的损失函数。
    """
    source = check_source()
    from src.core import YAMLConfig

    config = YAMLConfig(
        str(source / "configs/dfine/include/dfine_hgnetv2.yml"),
        num_classes=len(CLASS_NAMES), remap_mscoco_category=False,
        eval_spatial_size=[imgsz, imgsz], num_top_queries=100,
        HGNetv2={"name": "B4", "pretrained": False, "freeze_at": -1, "freeze_norm": True},
    )
    model = config.model
    original = model.backbone.stem.stem1.conv
    expanded = nn.Conv2d(5, original.out_channels, original.kernel_size, original.stride,
                         original.padding, bias=original.bias is not None)
    model.backbone.stem.stem1.conv = expanded
    return model, config.criterion if training else None


def transfer_pretrained(model: nn.Module, weights: Path) -> dict[str, object]:
    """迁移官方 Objects365 E25 参数，明确处理输入层、类别头和分辨率缓存。

    Note:
        只加载用户本地可信权重。animal/uav 保留新类别初始化；其余代表类别的
        分类头和去噪嵌入同时迁移，所有类别仍使用比赛标签继续学习。
    """
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    source = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    if source["decoder.enc_score_head.weight"].shape[0] != 366:
        raise ValueError("预训练权重不是 366 输出槽的官方 Objects365 模型，不能按当前类别表迁移")
    target = model.state_dict()
    state: dict[str, torch.Tensor] = {}
    regenerated: list[str] = []
    remapped: list[str] = []
    for name, value in target.items():
        if name in {"decoder.anchors", "decoder.valid_mask"}:
            regenerated.append(name)
            continue
        if name not in source:
            # FrozenBN 不含 batch 计数；其余未匹配参数必须报告，防止静默使用随机骨干。
            raise ValueError(f"官方权重缺少模型参数：{name}")
        incoming = source[name]
        if name == "backbone.stem.stem1.conv.weight":
            state[name] = adapt_rgb_stem_weights(incoming, 5)
        elif name.startswith("decoder.dec_score_head.") or name.startswith("decoder.enc_score_head.") or name == "decoder.denoising_class_embed.weight":
            adapted = value.clone()
            for class_id, class_name in enumerate(CLASS_NAMES):
                mapping = OBJECTS365_ROWS[class_name]
                if mapping is not None:
                    adapted[class_id] = incoming[mapping[0]]
            if name == "decoder.denoising_class_embed.weight":
                adapted[-1] = incoming[-1]
            state[name] = adapted
            remapped.append(name)
        elif value.shape == incoming.shape:
            state[name] = incoming
        else:
            raise ValueError(f"官方权重结构不匹配：{name}，{tuple(incoming.shape)} → {tuple(value.shape)}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != set(regenerated) or unexpected:
        raise ValueError(f"权重迁移存在未处理参数：{missing}, {unexpected}")
    stem = model.backbone.stem.stem1.conv
    if not stem.weight.requires_grad:
        raise ValueError("五通道输入层被意外冻结，红外和深度将无法学习")
    return {"loaded_tensors": len(state), "remapped_heads": remapped,
            "regenerated_buffers": regenerated, "class_initialization": OBJECTS365_ROWS}


def resize_multichannel(image: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    """缩放任意通道数的图像，绕过 OpenCV 对五通道输入的限制。

    Args:
        image: 最后一个维度为通道的图像数组。
        size: 目标尺寸，按 ``(宽, 高)`` 表示。
        interpolation: OpenCV 插值方式。

    Returns:
        与输入通道数一致、内存连续的缩放图像。

    Raises:
        ValueError: 输入不是二维或三维图像。
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"图像维度必须为二维或三维，实际为 {image.shape}")
    if image.ndim == 2 or image.shape[2] <= 4:
        return np.ascontiguousarray(cv2.resize(image, size, interpolation=interpolation))
    # OpenCV 的 resize 在当前版本拒绝五通道数组；逐通道处理仍保持三模态像素严格对齐。
    planes = [cv2.resize(image[:, :, channel], size, interpolation=interpolation)
              for channel in range(image.shape[2])]
    return np.ascontiguousarray(np.stack(planes, axis=2))


def resize_fused(image: np.ndarray, imgsz: int, scale: float = 1.0) -> tuple[np.ndarray, tuple[float, float, int, int]]:
    """对五通道同时等比缩放并居中填充，返回可精确反算的几何信息。"""
    height, width = image.shape[:2]
    ratio = imgsz / max(height, width) * scale
    resized_w, resized_h = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (imgsz - resized_w) // 2, (imgsz - resized_h) // 2
    resized = resize_multichannel(image, (resized_w, resized_h), cv2.INTER_LINEAR)
    canvas = np.empty((imgsz, imgsz, 5), dtype=np.uint8)
    # RGB 使用常规灰色填充，辅助模态以 0 表示无测量区域。
    canvas[:, :, :3] = 114
    canvas[:, :, 3:] = 0
    canvas[top:top + resized_h, left:left + resized_w] = resized
    return canvas, (resized_w / width, resized_h / height, left, top)


def image_tensor(image: np.ndarray) -> torch.Tensor:
    """转为 CHW 浮点输入，不翻转已经是 RGB 的前三通道。"""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)


def crop_complete_objects(image: np.ndarray, labels: np.ndarray, minimum: float) -> tuple[np.ndarray, np.ndarray]:
    """同步裁剪五通道，保留窗口内全部目标；切到任一目标时放弃该候选。

    Args:
        image: 原尺寸 RGBIRDepth 图像，不原地修改源数组或磁盘缓存。
        labels: class、cx、cy、w、h 五列归一化标签。
        minimum: 裁剪窗口边长相对原图的最小比例。

    Returns:
        裁剪图像及重算标签；没有合适窗口时原样返回。

    Note:
        不按模型预测删标签，不将部分可见目标静默丢弃。保留所有窗口内类别，
        不只保留锚定目标；验证集不调用本函数。
    """
    if not len(labels):
        return image, labels
    height, width = image.shape[:2]
    boxes = labels[:, 1:].copy()
    left = (boxes[:, 0] - boxes[:, 2] / 2).clip(0, 1) * width
    right = (boxes[:, 0] + boxes[:, 2] / 2).clip(0, 1) * width
    top = (boxes[:, 1] - boxes[:, 3] / 2).clip(0, 1) * height
    bottom = (boxes[:, 1] + boxes[:, 3] / 2).clip(0, 1) * height
    for _ in range(8):
        ratio = random.uniform(minimum, 1.0)
        crop_w, crop_h = max(1, round(width * ratio)), max(1, round(height * ratio))
        x, y = random.randint(0, width - crop_w), random.randint(0, height - crop_h)
        intersects = (right > x) & (left < x + crop_w) & (bottom > y) & (top < y + crop_h)
        inside = (left >= x) & (right <= x + crop_w) & (top >= y) & (bottom <= y + crop_h)
        if not inside.any() or np.any(intersects & ~inside):
            continue
        cropped = labels[inside].copy()
        cropped[:, 1] = ((left[inside] + right[inside]) / 2 - x) / crop_w
        cropped[:, 2] = ((top[inside] + bottom[inside]) / 2 - y) / crop_h
        cropped[:, 3] = (right[inside] - left[inside]) / crop_w
        cropped[:, 4] = (bottom[inside] - top[inside]) / crop_h
        return image[y:y + crop_h, x:x + crop_w].copy(), cropped
    return image, labels


class MultimodalDFineDataset(Dataset):
    """直接读取 datasets 的现有划分，标签只读，不复制或重建训练数据。"""

    def __init__(self, data_path: Path, split: str, config: TrainingConfig) -> None:
        self.config = config
        self.split = split
        self.epoch: int = 0
        data = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
        if data.get("channels") != 5 or list(data["names"].values()) != list(CLASS_NAMES):
            raise ValueError("datasets/data.yaml 的五通道或 12 类顺序不符")
        root = data_path.parent
        self.images = sorted(p for p in (root / data[split]).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self.infrared = root / data["infrared"][split]
        self.depth = root / data["depth"][split]
        label_dir = (root / data[split]).parent / "labels"
        if not self.images or len({p.stem for p in self.images}) != len(self.images) or {p.stem for p in self.images} != {p.stem for p in label_dir.glob("*.txt")}:
            raise ValueError(f"{split} 图像为空或与标签未一一对应")
        self.labels: list[np.ndarray] = []
        self.sizes: list[tuple[int, int]] = []
        annotations: list[dict[str, object]] = []
        image_records: list[dict[str, object]] = []
        signature = hashlib.sha256()
        for index, path in enumerate(self.images):
            for modality in (path, self.infrared / path.name, self.depth / path.name):
                if not modality.is_file():
                    raise FileNotFoundError(f"{split} 缺少配对图像：{modality}")
                stat = modality.stat()
                signature.update(f"{modality.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            text = (label_dir / f"{path.stem}.txt").read_text(encoding="utf-8-sig")
            signature.update(text.encode())
            rows = [line.split() for line in text.splitlines() if line.strip()]
            if any(len(row) != 5 for row in rows):
                raise ValueError(f"标签不是五列：{path.stem}")
            labels = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            if not np.isfinite(labels).all() or np.any(labels[:, 0] != labels[:, 0].astype(int)) or np.any((labels[:, 0] < 0) | (labels[:, 0] >= len(CLASS_NAMES))) or np.any(labels[:, 3:] <= 0):
                raise ValueError(f"标签类别、尺寸或数值非法：{path.stem}")
            with Image.open(path) as opened:
                width, height = opened.size
            self.sizes.append((width, height))
            self.labels.append(labels)
            image_records.append({"id": index, "file_name": path.name, "width": width, "height": height})
            for category, cx, cy, bw, bh in labels:
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                if min(x1, y1) < -1e-4 or max(x2, y2) > 1.0001:
                    raise ValueError(f"标签越界，请人工复核而非训练时静默清洗：{path.stem}")
                left, top = max(0.0, float(x1)) * width, max(0.0, float(y1)) * height
                box_width = min(1.0, float(x2)) * width - left
                box_height = min(1.0, float(y2)) * height - top
                annotations.append({"id": len(annotations) + 1, "image_id": index, "category_id": int(category),
                                    "bbox": [left, top, box_width, box_height], "area": box_width * box_height, "iscrowd": 0})
        self.signature: str = signature.hexdigest()
        self.coco_data: dict[str, object] = {"info": {}, "images": image_records, "annotations": annotations,
                                           "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        path = self.images[index]
        cache = path.with_suffix(".npy")
        sources = (path, self.infrared / path.name, self.depth / path.name, PROJECT_ROOT / "三模态训练.py")
        if cache.is_file() and cache.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in sources):
            image = np.load(cache, mmap_mode="r", allow_pickle=False)
            if image.dtype != np.uint8 or image.shape != (self.sizes[index][1], self.sizes[index][0], 5):
                raise ValueError(f"现有五通道缓存格式不符：{cache}；请人工处理，不自动删除")
        else:
            image = fuse_modalities(*sources[:3])
        training = self.split == "train"
        augmenting = training and self.epoch < self.config.polish_epoch
        labels = self.labels[index].copy()
        if augmenting and self.config.crop_prob > 0 and random.random() < self.config.crop_prob:
            image, labels = crop_complete_objects(image, labels, self.config.crop_min)
        height, width = image.shape[:2]
        if augmenting and self.config.lowres_prob > 0 and random.random() < self.config.lowres_prob:
            ratio = random.uniform(self.config.lowres_min, 0.9)
            reduced = resize_multichannel(
                image,
                (max(1, round(width * ratio)), max(1, round(height * ratio))),
                cv2.INTER_AREA,
            )
            # 同步模拟低清输入，画面坐标不变；不是伪造红外或深度，也不替代 JPG 域验证。
            image = resize_multichannel(reduced, (width, height), cv2.INTER_LINEAR)
        scale = random.uniform(self.config.scale_min, 1.0) if augmenting else 1.0
        canvas, geometry = resize_fused(image, self.config.imgsz, scale)
        sx, sy, left, top = geometry
        boxes = labels[:, 1:].copy()
        boxes[:, [0, 2]] *= width * sx
        boxes[:, [1, 3]] *= height * sy
        boxes[:, 0] += left
        boxes[:, 1] += top
        boxes /= self.config.imgsz
        if training and random.random() < self.config.fliplr:
            canvas = canvas[:, ::-1].copy()
            boxes[:, 0] = 1 - boxes[:, 0]
        if training:
            rgb = np.ascontiguousarray(canvas[:, :, :3])
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-1, 1) * self.config.hsv_h * 180) % 180
            hsv[:, :, 1] *= random.uniform(1 - self.config.hsv_s, 1 + self.config.hsv_s)
            hsv[:, :, 2] *= random.uniform(1 - self.config.hsv_v, 1 + self.config.hsv_v)
            canvas[:, :, :3] = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        target = {"labels": torch.tensor(labels[:, 0], dtype=torch.int64),
                  "boxes": torch.from_numpy(boxes), "image_id": torch.tensor(index),
                  "orig_size": torch.tensor([width, height]), "geometry": torch.tensor(geometry)}
        return image_tensor(canvas), target


class TargetedRepeatSampler(Sampler[int]):
    """覆盖全部训练图一次，仅给船和垃圾桶补充受限重复；验证集不参与。

    Note:
        倍率是上限约束前的期望值；每图最多增加一次，同图含两类时取较大倍率。
        文件名前缀与数字连续段仅作为重复限制分组，不冒充真实场景标注。
    """

    def __init__(self, dataset: MultimodalDFineDataset, config: TrainingConfig) -> None:
        self.config: TrainingConfig = config
        self.size: int = len(dataset)
        self.categories: list[set[int]] = [set(rows[:, 0].astype(int).tolist()) for rows in dataset.labels]
        numeric = sorted({int(path.stem) for path in dataset.images if path.stem.isdecimal()})
        numeric_groups: dict[int, str] = {}
        previous, start = -1000, 0
        for number in numeric:
            if number - previous > 50:
                start = number
            numeric_groups[number] = f"numeric_{start}"
            previous = number
        self.groups: list[str] = [numeric_groups[int(path.stem)] if path.stem.isdecimal()
                                  else path.stem.replace("_suppl_", "_").rsplit("_", 1)[0]
                                  for path in dataset.images]
        self.extra_probabilities: list[float] = [max(config.boat_repeat if 1 in classes else 1.0,
                                                   config.garbage_repeat if 9 in classes else 1.0) - 1.0
                                                for classes in self.categories]
        self.indices: list[int] = self.indices_for_epoch(0)

    def indices_for_epoch(self, epoch: int) -> list[int]:
        """只用局部随机状态生成该轮清单，断点恢复不依赖前几轮迭代次数。"""
        rng = random.Random(self.config.seed + epoch)
        indices = list(range(self.size))
        if epoch < self.config.polish_epoch:
            candidates = [i for i, probability in enumerate(self.extra_probabilities)
                          if probability > 0 and rng.random() < probability]
            rng.shuffle(candidates)
            group_counts: Counter[str] = Counter()
            budget = math.floor(self.size * self.config.repeat_extra_fraction)
            for index in candidates:
                if len(indices) - self.size >= budget:
                    break
                group = self.groups[index]
                if group_counts[group] < self.config.repeat_group_limit:
                    indices.append(index)
                    group_counts[group] += 1
        rng.shuffle(indices)
        return indices

    def set_epoch(self, epoch: int) -> None:
        """在创建当轮 DataLoader 迭代器前固定清单与长度。"""
        self.indices = self.indices_for_epoch(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)


def collate_samples(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """堆叠固定尺寸图像，保留每张图不同数量的目标。"""
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def seed_worker(worker_id: int) -> None:
    """Windows 子进程分别设置随机种子并限制 OpenCV 线程。"""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def float_outputs(value: object) -> object:
    """让匹配与几何损失用 FP32 计算，保留反向梯度和整数元数据。"""
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: float_outputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [float_outputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(float_outputs(item) for item in value)
    return value


def decode_predictions(outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]], imgsz: int,
                       conf: float, max_det: int) -> list[dict[str, torch.Tensor]]:
    """保留 D-FINE 原生查询类别排序，反算填充和缩放；不使用 NMS 或集成。"""
    probabilities = outputs["pred_logits"].float().sigmoid()
    # 与官方后处理一致，从查询×类别联合分数选择候选；不是 YOLO 的多标签 NMS。
    scores, positions = probabilities.flatten(1).topk(min(300, probabilities.shape[1] * len(CLASS_NAMES)), dim=1)
    categories = positions % len(CLASS_NAMES)
    queries = positions // len(CLASS_NAMES)
    all_boxes = box_convert(outputs["pred_boxes"].float(), "cxcywh", "xyxy") * imgsz
    all_boxes = all_boxes.gather(1, queries.unsqueeze(-1).expand(-1, -1, 4))
    results: list[dict[str, torch.Tensor]] = []
    for boxes, confidence, labels, target in zip(all_boxes, scores, categories, targets):
        sx, sy, left, top = [float(v) for v in target["geometry"]]
        width, height = [int(v) for v in target["orig_size"]]
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / sx).clamp(0, width)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / sy).clamp(0, height)
        valid = (confidence >= conf) & torch.isfinite(boxes).all(1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        results.append({"boxes": boxes[valid][:max_det], "scores": confidence[valid][:max_det], "labels": labels[valid][:max_det]})
    return results


@torch.inference_mode()
def validate_epoch(model: nn.Module, loader: DataLoader, device: torch.device, config: TrainingConfig) -> tuple[float, float, list[dict[str, object]], list[dict[str, object]]]:
    """只在正式训练轮末计算官方 COCO 口径 AP，供同轮最佳权重选择。"""
    model.eval()
    detections: list[dict[str, object]] = []
    for images, targets in tqdm(loader, desc="验证", file=sys.stdout, mininterval=5):
        outputs = model(images.to(device))
        predictions = decode_predictions(outputs, targets, config.imgsz, config.conf, config.max_det)
        for target, result in zip(targets, predictions):
            boxes = box_convert(result["boxes"], "xyxy", "xywh").cpu().tolist()
            for box, score, label in zip(boxes, result["scores"].cpu().tolist(), result["labels"].cpu().tolist()):
                detections.append({"image_id": int(target["image_id"]), "category_id": label, "bbox": box, "score": score})
    gt = COCO()
    gt.dataset = loader.dataset.coco_data
    gt.createIndex()
    if detections:
        predicted = gt.loadRes(detections)
    else:
        predicted = COCO()
        predicted.dataset = {**gt.dataset, "annotations": []}
        predicted.createIndex()
    evaluator = COCOeval_faster(gt, predicted, "bbox")
    evaluator.params.maxDets = [1, 10, config.max_det]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"]
    per_class: list[dict[str, object]] = []
    for index, name in enumerate(CLASS_NAMES):
        ap = precision[:, :, index, 0, -1]
        ap50 = precision[0, :, index, 0, -1]
        per_class.append({"class_id": index, "name": name,
                          "mAP50": float(ap50[ap50 >= 0].mean()) if np.any(ap50 >= 0) else None,
                          "mAP50-95": float(ap[ap >= 0].mean()) if np.any(ap >= 0) else None})
    domains: list[dict[str, object]] = []
    if config.domain_metrics:
        # 复用同一次验证的预测结果；按图像域筛选 GT，不额外执行模型前向。
        for domain, suffixes in (("png", {".png"}), ("jpg", {".jpg", ".jpeg"})):
            image_ids = [index for index, path in enumerate(loader.dataset.images) if path.suffix.lower() in suffixes]
            if not image_ids:
                continue
            subset = COCOeval_faster(gt, predicted, "bbox")
            subset.params.imgIds = image_ids
            subset.params.maxDets = [1, 10, config.max_det]
            subset.evaluate()
            subset.accumulate()
            subset.summarize()
            domains.append({"domain": domain, "name": "all", "images": len(image_ids),
                            "mAP50": float(subset.stats[1]), "mAP50-95": float(subset.stats[0])})
            for index, name in enumerate(CLASS_NAMES):
                ap = subset.eval["precision"][:, :, index, 0, -1]
                ap50 = subset.eval["precision"][0, :, index, 0, -1]
                domains.append({"domain": domain, "name": name, "images": len(image_ids),
                                "mAP50": float(ap50[ap50 >= 0].mean()) if np.any(ap50 >= 0) else None,
                                "mAP50-95": float(ap[ap >= 0].mean()) if np.any(ap >= 0) else None})
    return float(evaluator.stats[1]), float(evaluator.stats[0]), per_class, domains


class ConsoleLog:
    """将正式训练输出同时写入终端与运行日志。"""

    def __init__(self, console: TextIO, log: TextIO) -> None:
        self.console = console
        self.log = log

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return False


def append_csv(path: Path, row: dict[str, object]) -> None:
    """追加单轮记录，第一次写入时创建表头。"""
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    """先写临时权重，再替换本次运行的检查点，减少中断损坏风险。"""
    temporary = path.with_suffix(".pth.partial")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def plot_results(path: Path) -> None:
    """从已产生的 CSV 绘制趋势，不加载模型或额外评估。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (path / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, keys in zip(axes, (("train/loss",), ("metrics/mAP50(B)", "metrics/mAP50-95(B)"), ("lr/head", "lr/backbone"))):
        for key in keys:
            axis.plot(epochs, [float(row[key]) for row in rows], label=key)
        axis.legend()
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path / "results.png", dpi=150)
    plt.close(figure)


def train(config: TrainingConfig) -> None:
    """创建独立运行目录并执行固定预算微调；只能由 main.py 显式调用。"""
    if config.imgsz < 32 or config.imgsz % 32 or min(config.batch, config.val_batch, config.effective_batch) < 1 or config.effective_batch % config.batch:
        raise ValueError("imgsz 必须为 32 的倍数；effective_batch 必须为 batch 的正整数倍")
    if config.epochs <= config.warmup_epochs or config.warmup_epochs < 0 or not 0 < config.scale_min <= 1 or not 0 <= config.polish_epoch < config.epochs:
        raise ValueError("训练轮数、预热、收尾或尺度设置非法")
    if min(config.lr0, config.backbone_lr, config.clip_grad) <= 0 or not 0 < config.lrf <= 1 or config.weight_decay < 0:
        raise ValueError("学习率、衰减比例、梯度裁剪或权重衰减设置非法")
    if config.workers < 0 or config.max_det != 100 or not 0 <= config.conf <= 1 or not 0 <= config.fliplr <= 1:
        raise ValueError("加载进程、阈值或翻转概率非法；本项目 max_det 固定为 100")
    if not 0 <= config.crop_prob <= 1 or not 0 < config.crop_min <= 1 or not 0 <= config.lowres_prob <= 1 or not 0 < config.lowres_min <= 0.9:
        raise ValueError("裁剪或低清增强概率、尺度设置非法")
    if not all(1 <= value <= 2 for value in (config.boat_repeat, config.garbage_repeat)) or not 0 <= config.repeat_extra_fraction <= 0.10 or config.repeat_group_limit < 1:
        raise ValueError("重复倍率必须在1–2，每轮额外比例不能超过10%，每组上限必须为正整数")
    if config.lr_schedule not in {"legacy_batch", "optimizer_step"}:
        raise ValueError("学习率日程必须为 legacy_batch 或 optimizer_step")
    if max(config.boat_repeat, config.garbage_repeat) > 1 and config.lr_schedule != "optimizer_step":
        raise ValueError("受限重采样必须使用 optimizer_step，避免变化的轮长度造成学习率跳变")
    if not all(math.isfinite(value) and value > 0 for value in (config.loss_vfl, config.loss_bbox, config.loss_giou, config.loss_fgl, config.loss_ddf)):
        raise ValueError("D-FINE 损失权重必须为有限正数")
    if config.patience < 0 or not math.isfinite(config.min_delta) or config.min_delta < 0:
        raise ValueError("早停等待轮数和最小提升必须为非负数")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前配方需要支持 BF16 的 CUDA 显卡")
    for required in (config.data, config.resume or config.model):
        if not (PROJECT_ROOT / required).is_file():
            raise FileNotFoundError(f"缺少本地文件：{required}；训练不会联网下载")
    check_source()
    root = PROJECT_ROOT / config.project
    output = root / config.name
    if output.exists():
        output = root / f"{config.name}_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "weights").mkdir()
    with (output / "train.log").open("x", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(ConsoleLog(sys.stdout, log)), redirect_stderr(ConsoleLog(sys.stderr, log)):
            try:
                run_training(config, output)
            except BaseException:
                traceback.print_exc()
                raise


def run_training(config: TrainingConfig, output: Path) -> None:
    """执行单卡训练，按真实优化步更新学习率、EMA 与梯度累积。"""
    from src.optim import ModelEMA

    device = torch.device(f"cuda:{config.device}")
    torch.cuda.set_device(device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(0)
    training = MultimodalDFineDataset(PROJECT_ROOT / config.data, "train", config)
    validation = MultimodalDFineDataset(PROJECT_ROOT / config.data, "val", config)
    adapter_hashes = {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                      for name in ("D细化训练.py", "三模态训练.py", "准备三模态数据集.py")}
    if {p.stem for p in training.images} & {p.stem for p in validation.images}:
        raise ValueError("训练和验证存在同名样本，停止训练以避免泄漏")
    audit_hash: str | None = None
    if config.data_audit:
        audit_path = PROJECT_ROOT / config.data_audit
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)
        if audit.get("status") != "applied":
            raise ValueError("数据审计尚未落位，不能使用仅拟定的清洗计划训练")
        for split, dataset in (("train", training), ("val", validation)):
            expected = {row["image"]: row["after_sha256"] for row in audit["samples"] if row["after_split"] == split}
            if set(expected) != {path.name for path in dataset.images}:
                raise ValueError(f"{split} 图像清单不匹配 v8 数据审计")
            for path in dataset.images:
                label = path.parent.parent / "labels" / f"{path.stem}.txt"
                if hashlib.sha256(label.read_bytes()).hexdigest() != expected[path.name]:
                    raise ValueError(f"审计后标签发生变化，请先更新清洗记录：{label}")
        audit_hash = hashlib.sha256(audit_bytes).hexdigest()
    model, criterion = build_model(config.imgsz, training=True)
    if criterion is None:
        raise RuntimeError("未创建官方 D-FINE 损失函数")
    criterion.weight_dict.update({"loss_vfl": config.loss_vfl, "loss_bbox": config.loss_bbox,
                                  "loss_giou": config.loss_giou, "loss_fgl": config.loss_fgl,
                                  "loss_ddf": config.loss_ddf})
    resumed = torch.load(PROJECT_ROOT / config.resume, map_location="cpu", weights_only=False) if config.resume else None
    if resumed is not None:
        if resumed.get("format") != CHECKPOINT_FORMAT or resumed.get("dfine_commit") != DFINE_COMMIT or resumed.get("preprocess") != PREPROCESS_VERSION:
            raise ValueError("恢复权重的架构、源码或预处理版本不符")
        old_config = asdict(TrainingConfig(**resumed["config"]))
        for key, value in asdict(config).items():
            if key not in {"resume", "project", "name", "model"} and old_config[key] != value:
                raise ValueError(f"恢复时不能修改配方 {key}；新配方应重新初始化")
        if resumed["dataset_signatures"] != [training.signature, validation.signature]:
            raise ValueError("断点保存后数据发生变化，不能声称同状态续训")
        if resumed["epoch"] + 1 >= config.epochs or resumed.get("early_stopping", {}).get("stopped", False):
            raise ValueError("该训练已经完成，不能当作中断任务续训")
        if resumed.get("data_audit_sha256") != audit_hash:
            raise ValueError("审计记录发生变化，不能恢复旧配方断点")
        if config.data_audit and resumed.get("adapter_sha256") != adapter_hashes:
            raise ValueError("训练适配代码发生变化，不能恢复旧 v8 断点")
        model.load_state_dict(resumed["model"], strict=True)
        transfer = resumed["transfer"]
    else:
        transfer = transfer_pretrained(model, PROJECT_ROOT / config.model)
    model.to(device)
    criterion.to(device)
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            lr = config.backbone_lr if name.startswith("backbone.") and not name.startswith("backbone.stem.stem1.conv.") else config.lr0
            decay = config.weight_decay if parameter.ndim > 1 and not name.endswith("bias") else 0.0
            groups.setdefault((lr, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([{"params": values, "lr": lr, "initial_lr": lr, "weight_decay": decay}
                                  for (lr, decay), values in groups.items()], betas=(0.9, 0.999))
    ema = ModelEMA(model, decay=config.ema_decay, warmups=config.ema_warmup)
    start_epoch, best_ap, best_ap50 = 0, -1.0, -1.0
    early_stopping: dict[str, object] = {"best": -1.0, "bad_epochs": 0, "best_epoch": 0, "stopped": False}
    if resumed is not None:
        optimizer.load_state_dict(resumed["optimizer"])
        ema.load_state_dict(resumed["ema"])
        start_epoch = resumed["epoch"] + 1
        best_ap, best_ap50 = resumed["best_ap"], resumed["best_ap50"]
        early_stopping.update(resumed.get("early_stopping", {}))
        torch.set_rng_state(resumed["rng_torch"])
        torch.cuda.set_rng_state(resumed["rng_cuda"], device)
        random.setstate(resumed["rng_python"])
        np.random.set_state(resumed["rng_numpy"])
        # 新恢复目录仍保留原最佳文件，不把较差的后续轮次冒充历史最佳。
        previous = (PROJECT_ROOT / config.resume).parent
        for filename in ("best.pth", "best_map50.pth"):
            if not (previous / filename).is_file():
                raise FileNotFoundError(f"恢复所需的历史最佳权重缺失：{previous / filename}")
            shutil.copy2(previous / filename, output / "weights" / filename)
    generator = torch.Generator()
    loader_args = {"num_workers": config.workers, "pin_memory": False, "persistent_workers": False,
                   "prefetch_factor": 1 if config.workers else None, "worker_init_fn": seed_worker,
                   "collate_fn": collate_samples}
    sampler = TargetedRepeatSampler(training, config) if max(config.boat_repeat, config.garbage_repeat) > 1 else None
    train_loader = DataLoader(training, batch_size=config.batch, shuffle=sampler is None, sampler=sampler,
                              generator=generator, **loader_args)
    val_loader = DataLoader(validation, batch_size=config.val_batch, shuffle=False, **loader_args)
    # 清单只由训练标签、配方和 seed 决定；预先计数不读取图像、不执行模型。
    epoch_samples = [len(sampler.indices_for_epoch(epoch)) if sampler is not None else len(training)
                     for epoch in range(config.epochs)]
    epoch_updates = [math.ceil(count / config.effective_batch) for count in epoch_samples]
    total_updates = sum(epoch_updates)
    warmup_updates = sum(count * min(1.0, max(0.0, config.warmup_epochs - epoch))
                         for epoch, count in enumerate(epoch_updates))
    recipe = {"format": CHECKPOINT_FORMAT, "dfine_commit": DFINE_COMMIT, "preprocess": PREPROCESS_VERSION,
              "config": asdict(config), "classes": CLASS_NAMES, "transfer": transfer,
              "train_images": len(training), "val_images": len(validation),
              "dataset_signatures": [training.signature, validation.signature]}
    recipe["data_audit_sha256"] = audit_hash
    recipe["adapter_sha256"] = adapter_hashes
    recipe["loss_weights"] = dict(criterion.weight_dict)
    recipe["sampling"] = {"enabled": sampler is not None, "target_classes": {1: "boat", 9: "garbage can"},
                          "max_image_occurrences": 2 if sampler is not None else 1,
                          "grouping": "filename_prefix_or_numeric_gap50_not_verified_scene",
                          "stop_epoch": config.polish_epoch + 1, "samples_per_epoch": epoch_samples,
                          "optimizer_updates_per_epoch": epoch_updates, "total_optimizer_updates": total_updates}
    # 随正式训练留存类别和输入域组成，提醒本地验证的覆盖范围，不额外执行模型评估。
    recipe["dataset_composition"] = {
        split: {"images": len(dataset),
                "extensions": {suffix: sum(p.suffix.lower() == suffix for p in dataset.images) for suffix in sorted(IMAGE_SUFFIXES)},
                "class_boxes": {name: sum(int((rows[:, 0] == index).sum()) for rows in dataset.labels)
                                for index, name in enumerate(CLASS_NAMES)}}
        for split, dataset in (("train", training), ("val", validation))
    }
    with (PROJECT_ROOT / (config.resume or config.model)).open("rb") as handle:
        recipe["initial_weights_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    (output / "optimization_recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "args.yaml").write_text(yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=False), encoding="utf-8")
    snapshot = output / "code"
    snapshot.mkdir()
    if config.data_audit:
        shutil.copy2(PROJECT_ROOT / config.data_audit, output / "dataset_audit.json")
    for filename in ("main.py", "D细化训练.py", "三模态训练.py", "准备三模态数据集.py", "三模态预测.py", "predict.py", "pyproject.toml", "uv.lock"):
        shutil.copy2(PROJECT_ROOT / filename, snapshot / filename)
    print(f"D-FINE-L 五通道训练：{len(training)} train / {len(validation)} val，{config.imgsz}px，{start_epoch + 1}–{config.epochs} 轮")
    print(f"迁移记录：{transfer}\n结果目录：{output}\nBF16 前向、FP32 损失；物理批次 {config.batch}，有效批次 {config.effective_batch}")
    accumulate = config.effective_batch // config.batch
    # 释放 CPU 反序列化副本，避免整轮训练保留模型、优化器和 EMA 的重复存储。
    del resumed
    started = time.monotonic()
    for epoch in range(start_epoch, config.epochs):
        training.epoch = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        sample_count = epoch_samples[epoch]
        updates_before_epoch = sum(epoch_updates[:epoch])
        if sampler is not None:
            sampled_categories = [sampler.categories[index] for index in sampler.indices]
            sampling_record = {"epoch": epoch + 1, "original_images": len(training), "samples": sample_count,
                               "extras": sample_count - len(training),
                               "boat_image_draws": sum(1 in categories for categories in sampled_categories),
                               "garbage_image_draws": sum(9 in categories for categories in sampled_categories),
                               "optimizer_updates": epoch_updates[epoch]}
            append_csv(output / "sampling_history.csv", sampling_record)
            print(f"本轮采样：{sample_count} 张（额外 {sampling_record['extras']}）；含船 {sampling_record['boat_image_draws']} 次，含垃圾桶 {sampling_record['garbage_image_draws']} 次")
        generator.manual_seed(config.seed + epoch)
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_images = 0
        for step, (images, targets) in enumerate(tqdm(train_loader, desc=f"训练 {epoch + 1}/{config.epochs}", file=sys.stdout, mininterval=5)):
            if config.lr_schedule == "optimizer_step":
                update_index = updates_before_epoch + step // accumulate
                progress = update_index / max(total_updates - 1, 1)
                warmup = min(1.0, (update_index + 1) / max(warmup_updates, 1))
            else:
                # 保留旧版本日程，仅用于不含重采样的历史配方恢复。
                progress = (epoch * len(train_loader) + step) / max(config.epochs * len(train_loader) - 1, 1)
                warmup = min(1.0, (epoch * len(train_loader) + step + 1) / max(config.warmup_epochs * len(train_loader), 1))
            factor = (config.lrf + (1 - config.lrf) * (1 + math.cos(math.pi * progress)) / 2) * warmup
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            batch_count = images.shape[0]
            images = images.to(device)
            device_targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = model(images, targets=device_targets)
            losses = criterion(float_outputs(predictions), device_targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"第 {epoch + 1} 轮第 {step + 1} 批损失非有限值，已停止；上一轮 last.pth 保留")
            group_start = (step // accumulate) * accumulate * config.batch
            group_size = min(config.effective_batch, sample_count - group_start)
            (loss * batch_count / group_size).backward()
            if (step + 1) % accumulate == 0 or step + 1 == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            total_loss += float(loss.detach()) * batch_count
            total_images += batch_count
            del predictions, losses, loss
        ap50, ap, per_class, domains = validate_epoch(ema.module, val_loader, device, config)
        if not math.isfinite(ap) or not math.isfinite(ap50):
            raise FloatingPointError("验证指标非有限值，拒绝保存为正常检查点")
        improved, improved50 = ap > best_ap, ap50 > best_ap50
        best_ap, best_ap50 = max(best_ap, ap), max(best_ap50, ap50)
        if ap > float(early_stopping["best"]) + config.min_delta:
            early_stopping.update(best=ap, bad_epochs=0, best_epoch=epoch + 1)
        else:
            early_stopping["bad_epochs"] = int(early_stopping["bad_epochs"]) + 1
        early_stopping["stopped"] = bool(config.patience and int(early_stopping["bad_epochs"]) >= config.patience)
        checkpoint = {**recipe, "epoch": epoch, "model": model.state_dict(), "ema": ema.state_dict(),
                      "optimizer": optimizer.state_dict(), "best_ap": best_ap, "best_ap50": best_ap50,
                      "early_stopping": dict(early_stopping),
                      "metrics": {"mAP50": ap50, "mAP50-95": ap},
                      "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(device),
                      "rng_python": random.getstate(), "rng_numpy": np.random.get_state()}
        save_checkpoint(output / "weights/last.pth", checkpoint)
        if improved:
            save_checkpoint(output / "weights/best.pth", checkpoint)
        if improved50:
            save_checkpoint(output / "weights/best_map50.pth", checkpoint)
        if config.save_period > 0 and (epoch + 1) % config.save_period == 0:
            save_checkpoint(output / f"weights/epoch{epoch + 1}.pth", checkpoint)
        append_csv(output / "results.csv", {"epoch": epoch + 1, "time": time.monotonic() - started,
                   "train/loss": total_loss / total_images, "metrics/mAP50(B)": ap50, "metrics/mAP50-95(B)": ap,
                   "lr/head": config.lr0 * factor, "lr/backbone": config.backbone_lr * factor,
                   "train/samples": total_images, "train/optimizer_updates": epoch_updates[epoch]})
        for row in per_class:
            append_csv(output / "per_class_metrics.csv", {"epoch": epoch + 1, **row})
        for row in domains:
            append_csv(output / "domain_metrics.csv", {"epoch": epoch + 1, **row})
        print(f"第 {epoch + 1} 轮：AP50={ap50:.5f}，AP50-95={ap:.5f}；历史最佳 {best_ap50:.5f}/{best_ap:.5f}")
        if improved or (epoch + 1) % 5 == 0 or epoch + 1 == config.epochs or early_stopping["stopped"]:
            plot_results(output)
        if early_stopping["stopped"]:
            print(f"早停：mAP50-95 连续 {config.patience} 轮未超过累计有效提升阈值 {config.min_delta}；两种最佳权重均已保留")
            break
    print(f"训练完成：{output / 'weights/best.pth'}；推理使用同一模型的 EMA，不做多模型集成")


class DFinePredictor:
    """加载本项目 D-FINE 检查点，以训练同口径处理单张或批量五通道图像。"""

    def __init__(self, weights: Path, device: str, imgsz: int) -> None:
        self.device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        self.imgsz = imgsz
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("dfine_commit") != DFINE_COMMIT:
            raise ValueError("不是本项目固定版本的 D-FINE 五通道权重")
        if checkpoint.get("preprocess") != PREPROCESS_VERSION or list(checkpoint["classes"]) != list(CLASS_NAMES):
            raise ValueError("权重的预处理或类别顺序与当前预测代码不符")
        if checkpoint["config"]["imgsz"] != imgsz:
            raise ValueError("首版 D-FINE 预测必须与训练 imgsz 一致，避免隐式更改位置编码与锚点")
        self.model, _ = build_model(imgsz)
        self.model.load_state_dict(checkpoint["ema"]["module"], strict=True)
        self.model.to(self.device).eval()
        self.names: dict[int, str] = dict(enumerate(CLASS_NAMES))
        self.epoch: int = checkpoint["epoch"] + 1

    @torch.inference_mode()
    def predict(self, fused: np.ndarray, conf: float, max_det: int = 100) -> dict[str, torch.Tensor]:
        """FP32 单模型预测并还原为原图像素坐标。"""
        canvas, geometry = resize_fused(fused, self.imgsz)
        height, width = fused.shape[:2]
        targets = [{"orig_size": torch.tensor([width, height]), "geometry": torch.tensor(geometry)}]
        images = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).unsqueeze(0)
        return self.predict_batch(images, targets, conf, max_det)[0]

    @torch.inference_mode()
    def predict_batch(self, images: torch.Tensor, targets: list[dict[str, torch.Tensor]],
                      conf: float, max_det: int = 100) -> list[dict[str, torch.Tensor]]:
        """整批传入 GPU，以 FP32 前向并分别还原原图坐标。

        Args:
            images: 已等比填充的 CPU uint8 NCHW 五通道批次，可使用锁页内存。
            targets: 每张原图的尺寸与缩放、填充信息，顺序必须与批次一致。
            conf: 提交候选的最低置信度。
            max_det: 每图保留的候选框上限。

        Returns:
            与输入顺序一致的原图像素坐标、置信度和类别。
        """
        if images.dtype != torch.uint8 or images.ndim != 4 or tuple(images.shape[1:]) != (5, self.imgsz, self.imgsz):
            raise ValueError("D-FINE 批次必须是 uint8 的 [N, 5, imgsz, imgsz]")
        if not len(images) or len(images) != len(targets):
            raise ValueError("D-FINE 批次不能为空，且图像与坐标信息数量必须一致")
        # 先传 uint8 再转 FP32，传输字节数为 CPU 浮点输入的四分之一。
        inputs = images.to(self.device, non_blocking=images.is_pinned()).float().div_(255)
        outputs = self.model(inputs)
        return decode_predictions(outputs, targets, self.imgsz, conf, max_det)
