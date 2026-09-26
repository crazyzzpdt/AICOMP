"""将官方 D-FINE-L 接入本地 RGB、红外、深度五通道训练与预测。

由 train2.py 启动；仅支持新浮点协议训练，旧权重推理由predict2.py保留。
不联网下载、不改写数据集，不运行独立评估；验证仅发生在用户启动的训练轮末。
"""

# 内置库
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
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert
from tqdm import tqdm
import yaml

# 自己的模块
from src.modalities import (CLASS_NAMES, IMAGE_SUFFIXES, FLOAT_PREPROCESS_VERSION, SensorAugment,
                            read_float_modalities, letterbox_float, augment_sensors, configure_fp32)
from src.dfine.runtime import check_source, build_model, decode_predictions


# 固定官方源码版本，避免本机更新第三方仓库后静默改变训练行为。
DFINE_COMMIT: str = "956d1709314c2c6a4df6f34de232054578a7449f"
# 路径相对本项目解析，不依赖 IDE 当前目录。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
# 366 输出槽包含 0 占位；这里直接写官方 1 起始 category_id，不再额外加一。
# 宽泛类别使用代表类别作初始化，不宣称定义完全相同；animal/uav 无直接对应时重新初始化。
OBJECTS365_ROWS: dict[str, tuple[int, str] | None] = {
    "person": (1, "Person"), "boat": (22, "Boat"), "animal": None,
    "seat": (3, "Chair"), "sign": (90, "Traffic Sign"), "bicycle": (47, "Bicycle"),
    "car": (6, "Car"), "ball": (157, "Other Balls"), "light": (7, "Lamp"),
    "garbage can": (45, "Trash bin Can"), "uav": None, "tricycle": (184, "Tricycle"),
}
# 与官方预训练一致采用 0–1 输入，不额外套用 ImageNet 均值和标准差。
PREPROCESS_VERSION: str = FLOAT_PREPROCESS_VERSION
# 区分本项目检查点与 Ultralytics 序列化模型，拒绝误加载 RGB 官方原权重。
CHECKPOINT_FORMAT: str = "aic_dfine_l_5ch_v1"


@dataclass(frozen=True)
class TrainingConfig:
    """保存新训练配方；历史权重仅保留推理，不恢复旧优化状态。"""

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
    # 新训练不恢复旧优化状态，也不叠加历史定向采样或裁剪。
    sensors: SensorAugment = SensorAugment()
    loss_vfl: float = 1.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    loss_fgl: float = 0.15
    loss_ddf: float = 1.5
    patience: int = 15
    min_delta: float = 0.0005
    domain_metrics: bool = True
    data_audit: str | None = None


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
            adapted = torch.zeros_like(value)
            adapted[:, :3] = incoming
            state[name] = adapted
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



def image_tensor(image: np.ndarray) -> torch.Tensor:
    """连续浮点CHW，网络只除255一次，不做取整。"""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)


class MultimodalDFineDataset(Dataset):
    """只读现有划分，内存中统一训练与验证的原图边界裁框，不写回标签。"""

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
        for folder in (self.infrared, self.depth):
            if not folder.is_dir() or {p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES} != {p.name for p in self.images}:
                raise ValueError(f"{split}模态目录与可见光清单不一致：{folder}")
        label_dir = (root / data[split]).parent / "labels"
        if not self.images or len({p.stem for p in self.images}) != len(self.images) or {p.stem for p in self.images} != {p.stem for p in label_dir.glob("*.txt")}:
            raise ValueError(f"{split} 图像为空或与标签未一一对应")
        self.labels: list[np.ndarray] = []
        self.sizes: list[tuple[int, int]] = []
        self.clipped_images: int = 0
        self.clipped_boxes: int = 0
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
            image_records.append({"id": index, "file_name": path.name, "width": width, "height": height})
            clipped_in_image: int = 0
            for box_index, row in enumerate(labels):
                category, cx, cy, bw, bh = (float(value) for value in row)
                original_box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
                x1, y1 = max(0.0, original_box[0]), max(0.0, original_box[1])
                x2, y2 = min(1.0, original_box[2]), min(1.0, original_box[3])
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"标签框与原图无有效交集：{path.stem}，第 {box_index + 1} 个框；请核对原始标注")
                # 官方标签中存在部分越界的贴边框；仅处理内存副本，原文及其审计摘要保持不变。
                # 训练目标与COCO标注共用此处的原图交集，避免一边用原框、一边用裁后框。
                if original_box != (x1, y1, x2, y2):
                    labels[box_index, 1:] = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                    clipped_in_image += 1
                left, top = x1 * width, y1 * height
                box_width = (x2 - x1) * width
                box_height = (y2 - y1) * height
                annotations.append({"id": len(annotations) + 1, "image_id": index, "category_id": int(category),
                                    "bbox": [left, top, box_width, box_height], "area": box_width * box_height, "iscrowd": 0})
            self.labels.append(labels)
            self.clipped_images += int(clipped_in_image > 0)
            self.clipped_boxes += clipped_in_image
        self.signature: str = signature.hexdigest()
        self.coco_data: dict[str, object] = {"info": {}, "images": image_records, "annotations": annotations,
                                           "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]}
        print(f"{split} 标签边界处理：{self.clipped_images} 张图、{self.clipped_boxes} 个框在内存中裁至原图边界；标签文件未修改，目标未删除")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        path = self.images[index]
        image, metric_depth = read_float_modalities(path, self.infrared / path.name, self.depth / path.name)
        training = self.split == "train"
        augmenting = training and self.epoch < self.config.polish_epoch
        labels = self.labels[index].copy()
        if training:
            image = augment_sensors(image, metric_depth, self.config.sensors)
        height, width = image.shape[:2]
        scale = random.uniform(self.config.scale_min, 1.0) if augmenting else 1.0
        canvas, geometry = letterbox_float(image, self.config.imgsz, scale=scale)
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
        if training and any((self.config.hsv_h, self.config.hsv_s, self.config.hsv_v)):
            # OpenCV浮点HSV：H为0–360，S/V为0–1，不转回uint8。
            rgb = np.ascontiguousarray(canvas[:, :, :3] / 255.0)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-1, 1) * self.config.hsv_h * 360) % 360
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(1 - self.config.hsv_s, 1 + self.config.hsv_s), 0, 1)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(1 - self.config.hsv_v, 1 + self.config.hsv_v), 0, 1)
            canvas[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255.0
        target = {"labels": torch.tensor(labels[:, 0], dtype=torch.int64),
                  "boxes": torch.from_numpy(boxes), "image_id": torch.tensor(index),
                  "orig_size": torch.tensor([width, height]), "geometry": torch.tensor(geometry)}
        return image_tensor(canvas), target


def collate_samples(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """堆叠固定尺寸图像，保留每张图不同数量的目标。"""
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def seed_worker(worker_id: int) -> None:
    """Windows 子进程分别设置随机种子并限制 OpenCV 线程。"""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


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
    """创建独立运行目录并执行固定预算微调；只能由 train2.py 显式调用。"""
    if config.imgsz < 32 or config.imgsz % 32 or min(config.batch, config.val_batch, config.effective_batch) < 1 or config.effective_batch % config.batch:
        raise ValueError("imgsz 必须为 32 的倍数；effective_batch 必须为 batch 的正整数倍")
    if config.epochs <= config.warmup_epochs or config.warmup_epochs < 0 or not 0 < config.scale_min <= 1 or not 0 <= config.polish_epoch < config.epochs:
        raise ValueError("训练轮数、预热、收尾或尺度设置非法")
    if min(config.lr0, config.backbone_lr, config.clip_grad) <= 0 or not 0 < config.lrf <= 1 or config.weight_decay < 0:
        raise ValueError("学习率、衰减比例、梯度裁剪或权重衰减设置非法")
    if config.workers < 0 or config.max_det != 100 or not 0 <= config.conf <= 1 or not 0 <= config.fliplr <= 1:
        raise ValueError("加载进程、阈值或翻转概率非法；本项目 max_det 固定为 100")
    if not all(math.isfinite(value) and value > 0 for value in (config.loss_vfl, config.loss_bbox, config.loss_giou, config.loss_fgl, config.loss_ddf)):
        raise ValueError("D-FINE 损失权重必须为有限正数")
    if config.patience < 0 or not math.isfinite(config.min_delta) or config.min_delta < 0:
        raise ValueError("早停等待轮数和最小提升必须为非负数")
    if config.resume is not None:
        raise ValueError("新浮点训练只从官方Objects365迁移，不恢复旧训练状态")
    if not torch.cuda.is_available():
        raise RuntimeError("当前训练配方需要CUDA显卡")
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
    check_source()
    from src.optim import ModelEMA

    configure_fp32()

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
                      for name in ("src/dfine/__init__.py", "src/dfine/training.py", "src/modalities.py", "src/dfine/runtime.py", "src/D-FINE/source_manifest.json")}
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
        for sample in audit["samples"]:
            for folder, modality in (("images", "visible"), ("infrared", "infrared"), ("depth", "depth")):
                path = (PROJECT_ROOT / config.data).parent / sample["after_split"] / folder / sample["image"]
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if digest != sample.get("source_hashes", {}).get(modality):
                    raise ValueError(f"三模态来源与清洗审计不符：{path.name}/{modality}")
    model, criterion = build_model(config.imgsz, training=True)
    if criterion is None:
        raise RuntimeError("未创建官方 D-FINE 损失函数")
    criterion.weight_dict.update({"loss_vfl": config.loss_vfl, "loss_bbox": config.loss_bbox,
                                  "loss_giou": config.loss_giou, "loss_fgl": config.loss_fgl,
                                  "loss_ddf": config.loss_ddf})
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
    generator = torch.Generator()
    loader_args = {"num_workers": config.workers, "pin_memory": False, "persistent_workers": False,
                   "prefetch_factor": 1 if config.workers else None, "worker_init_fn": seed_worker,
                   "collate_fn": collate_samples}
    train_loader = DataLoader(training, batch_size=config.batch, shuffle=True,
                              generator=generator, **loader_args)
    val_loader = DataLoader(validation, batch_size=config.val_batch, shuffle=False, **loader_args)
    # 清单只由训练标签、配方和 seed 决定；预先计数不读取图像、不执行模型。
    epoch_samples = [len(training)] * config.epochs
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
    recipe["label_geometry"] = {
        "policy": "clip_to_image_bounds_v1", "source_files_modified": False,
        "degenerate_boxes": "error",
        "clipped": {split: {"images": dataset.clipped_images, "boxes": dataset.clipped_boxes}
                    for split, dataset in (("train", training), ("val", validation))},
    }
    recipe["loss_weights"] = dict(criterion.weight_dict)
    recipe["precision"] = "FP32_no_autocast_no_tf32"
    recipe["sampling"] = {"enabled": False, "samples_per_epoch": epoch_samples}
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
    for filename in ("train2.py", "predict2.py", "predict1.py", "src/__init__.py", "src/dfine/__init__.py", "src/dfine/training.py",
                     "src/D-FINE/source_manifest.json", "src/modalities.py", "src/dfine/runtime.py", "pyproject.toml", "uv.lock"):
        target = snapshot / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / filename, target)
    print(f"D-FINE-L 五通道训练：{len(training)} train / {len(validation)} val，{config.imgsz}px，{start_epoch + 1}–{config.epochs} 轮")
    print(f"迁移记录：{transfer}\n结果目录：{output}\nFP32 前向、损失与EMA；物理批次 {config.batch}，有效批次 {config.effective_batch}")
    accumulate = config.effective_batch // config.batch
    started = time.monotonic()
    for epoch in range(start_epoch, config.epochs):
        training.epoch = epoch
        sample_count = epoch_samples[epoch]
        updates_before_epoch = sum(epoch_updates[:epoch])
        generator.manual_seed(config.seed + epoch)
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_images = 0
        for step, (images, targets) in enumerate(tqdm(train_loader, desc=f"训练 {epoch + 1}/{config.epochs}", file=sys.stdout, mininterval=5)):
            update_index = updates_before_epoch + step // accumulate
            progress = update_index / max(total_updates - 1, 1)
            warmup = min(1.0, (update_index + 1) / max(warmup_updates, 1))
            factor = (config.lrf + (1 - config.lrf) * (1 + math.cos(math.pi * progress)) / 2) * warmup
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            batch_count = images.shape[0]
            images = images.to(device)
            device_targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            predictions = model(images.float(), targets=device_targets)
            losses = criterion(predictions, device_targets)
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
