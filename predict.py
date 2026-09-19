"""批量预测 RGB、红外、深度五通道图像，并生成赛事提交 ZIP。

读取、推理与保存并行调度；检测保持 FP32、单模型、训练同口径预处理。
结果存入 images、labels、比赛提交内容三个子目录，已有输出目录不覆盖。

在项目根目录执行预测：
    uv run python predict.py
调整批次与输出位置：
    uv run python predict.py --batch 4 --output predict_v9_batch4
使用历史 YOLO 权重：
    uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_yolo26l_1280_v4_clean_lr1e4/weights/best.pt --imgsz 1280 --yolo-profile v4 --output predict_v4_restored
查看命令行覆盖参数：
    uv run python predict.py --help
"""

from __future__ import annotations

# 内置库
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# 必须在导入 Ultralytics 前设置，推理不联网检查或自动安装依赖。
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# 三方库
import cv2
import numpy as np
import torch
import ultralytics
from torch import nn
from torchvision.ops import box_convert
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops

# 自己的模块
from aic.data import CLASS_NAMES, IMAGE_SUFFIXES, fuse_modalities
from aic.model import (FUSION_VERSION, FusionDetectionModel, canvas_shape, clip_canvas_boxes,
                       letterbox_fused, restore_boxes, single_label_nms)


# 相对路径以入口文件所在目录为基准，兼容 IDE 从其他位置启动。
PROJECT_ROOT: Path = Path(__file__).resolve().parent
# 本轮训练完成后使用v9 AP95最佳权重；同名训练递增目录须填写实际路径。
MODEL_PATH: Path = PROJECT_ROOT / "runs/detect/AIC_RGBIRDepth_yolo26l_1920x1080_v9_fusion/weights/best.pt"
# 官方初赛的同名 visible、infrared、depth 三模态图像。
SOURCE_PATH: Path = PROJECT_ROOT / "数据集/测试集/AIC2026_PHASE_1_1000"
# 与训练产物隔离；再次预测时修改这里或传 --output，不清空已有结果。
OUTPUT_PATH: Path = PROJECT_ROOT / "predict_v9"


# 赛事规定单图最多100框，所有后端统一截断。
MAX_DETECTIONS: int = 100


# 历史D-FINE权重仍严格校验原源码、预处理及检查点格式。
DFINE_COMMIT: str = "956d1709314c2c6a4df6f34de232054578a7449f"
DFINE_PREPROCESS_VERSION: str = "rgbirdepth_uint8_letterbox_div255_v1"
DFINE_CHECKPOINT_FORMAT: str = "aic_dfine_l_5ch_v1"


@dataclass(frozen=True)
class PredictionConfig:
    """统一保存入口参数，命令行仅覆盖用户明确指定的字段。"""

    weights: Path
    source: Path
    output: Path
    backend: str
    expected_count: int
    imgsz: int
    batch: int
    workers: int
    save_workers: int
    prefetch_batches: int
    pin_memory: bool
    device: str
    conf: float
    iou: float
    max_det: int
    multi_label: bool
    visual_conf: float
    png_compression: int
    log_every: int
    # 恢复历史 YOLO 逐张原生后处理；D-FINE 后端不读取该设置。
    yolo_profile: str = "v4"
    # 新融合模型的内容高度；旧YOLO/D-FINE保持各自历史预处理。
    height: int = 1080


@dataclass
class PredictionSample:
    """保存单张 CPU 输入及坐标信息，不在线程之间传递 GPU 对象。"""

    path: Path
    image: np.ndarray
    visible: np.ndarray | None
    target: dict[str, torch.Tensor]


# 一、模型后端：当前融合模型及历史D-FINE兼容
class FusionPredictor:
    """单次加载融合模型，对已矩形化的五通道批次执行 FP32 推理。"""

    def __init__(self, model: FusionDetectionModel, device: str, content_hw: tuple[int, int]) -> None:
        if model.fusion_version != FUSION_VERSION or tuple(model.content_hw) != tuple(content_hw):
            raise ValueError(f"权重要求版本 {getattr(model, 'fusion_version', None)}、高宽 {model.content_hw}，请同步预测参数")
        self.device = torch.device("cpu" if str(device) == "cpu" else f"cuda:{int(device)}")
        self.model = model.to(self.device).float().eval()
        self.model.end2end = False
        self.names = model.names
        self.content_hw = content_hw

    @torch.inference_mode()
    def predict_batch(self, images: torch.Tensor, targets: list[dict[str, torch.Tensor]],
                      conf: float, max_det: int, iou: float) -> list[dict[str, torch.Tensor]]:
        """只做一次模型前向，用训练验证相同的NMS与原图坐标还原。"""
        images = images.to(self.device, non_blocking=True).float().div_(255)
        rows = single_label_nms(self.model(images), conf, iou, max_det)
        outputs: list[dict[str, torch.Tensor]] = []
        for row, target in zip(rows, targets, strict=True):
            width, height = target["orig_size"].tolist()
            geometry = tuple(target["geometry"].tolist())
            boxes, keep = clip_canvas_boxes(row[:, :4], geometry, (height, width))
            boxes = restore_boxes(boxes, geometry, (height, width))
            outputs.append({"boxes": boxes, "scores": row[keep, 4], "labels": row[keep, 5].long()})
        return outputs


def check_dfine_source() -> Path:
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


def build_dfine_model(imgsz: int) -> nn.Module:
    """构建不联网的 12 类五通道 D-FINE-L，输入端不冻结。

    Args:
        imgsz: 正方形填充尺寸，必须为 32 的倍数。

    Returns:
        与历史检查点结构一致的五通道模型。
    """
    source = check_dfine_source()
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
    return model


def resize_dfine_channels(image: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
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


def resize_dfine_fused(image: np.ndarray, imgsz: int, scale: float = 1.0) -> tuple[np.ndarray, tuple[float, float, int, int]]:
    """对五通道同时等比缩放并居中填充，返回可精确反算的几何信息。"""
    height, width = image.shape[:2]
    ratio = imgsz / max(height, width) * scale
    resized_w, resized_h = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (imgsz - resized_w) // 2, (imgsz - resized_h) // 2
    resized = resize_dfine_channels(image, (resized_w, resized_h), cv2.INTER_LINEAR)
    canvas = np.empty((imgsz, imgsz, 5), dtype=np.uint8)
    # RGB 使用常规灰色填充，辅助模态以 0 表示无测量区域。
    canvas[:, :, :3] = 114
    canvas[:, :, 3:] = 0
    canvas[top:top + resized_h, left:left + resized_w] = resized
    return canvas, (resized_w / width, resized_h / height, left, top)


def decode_dfine_predictions(outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]], imgsz: int,
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


class DFinePredictor:
    """加载本项目 D-FINE 检查点，以训练同口径处理单张或批量五通道图像。"""

    def __init__(self, weights: Path, device: str, imgsz: int) -> None:
        """核对历史检查点格式，并加载EMA模型，不恢复训练状态。"""
        self.device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        self.imgsz = imgsz
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != DFINE_CHECKPOINT_FORMAT or checkpoint.get("dfine_commit") != DFINE_COMMIT:
            raise ValueError("不是本项目固定版本的 D-FINE 五通道权重")
        if checkpoint.get("preprocess") != DFINE_PREPROCESS_VERSION or list(checkpoint["classes"]) != list(CLASS_NAMES):
            raise ValueError("权重的预处理或类别顺序与当前预测代码不符")
        if checkpoint["config"]["imgsz"] != imgsz:
            raise ValueError("首版 D-FINE 预测必须与训练 imgsz 一致，避免隐式更改位置编码与锚点")
        self.model = build_dfine_model(imgsz)
        self.model.load_state_dict(checkpoint["ema"]["module"], strict=True)
        self.model.to(self.device).eval()
        self.names: dict[int, str] = dict(enumerate(CLASS_NAMES))
        self.epoch: int = checkpoint["epoch"] + 1

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
        return decode_dfine_predictions(outputs, targets, self.imgsz, conf, max_det)


# 二、输入配对与模型检查
def collect_samples(source: Path) -> list[tuple[Path, Path, Path]]:
    """按完整文件名配对三模态，拒绝缺图与提交 TXT 同名冲突。

    Args:
        source: 包含 visible、infrared、depth 的官方测试集根目录。

    Returns:
        按文件名排序的可见光、红外、深度路径三元组。

    Raises:
        FileNotFoundError: 模态目录缺失。
        ValueError: 没有图像、模态配对不完整或不同图像共用同一词干。
    """
    folders: list[Path] = [source / modality for modality in ("visible", "infrared", "depth")]
    files: list[dict[str, Path]] = []
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到模态目录：{folder}")
        files.append({p.name: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES})
    names: set[str] = set(files[0])
    if not names:
        raise ValueError(f"可见光目录没有 PNG/JPG/JPEG 图像：{folders[0]}")
    for folder, modality_files in zip(folders[1:], files[1:]):
        if names != set(modality_files):
            missing = sorted(names - set(modality_files))[:5]
            extra = sorted(set(modality_files) - names)[:5]
            raise ValueError(f"三模态配对不完整：{folder.name}，缺失 {missing}，多余 {extra}")
    if len({Path(name).stem.casefold() for name in names}) != len(names):
        raise ValueError("不同图片会生成同名 TXT，请检查大小写或不同扩展名的词干冲突")
    return [(files[0][name], files[1][name], files[2][name]) for name in sorted(names)]


def validate_model(model: YOLO) -> None:
    """只接受本项目的五通道、12 类本地检测权重。

    Args:
        model: 已从本地检查点加载的 YOLO 模型。

    Raises:
        ValueError: 任务、首层通道数或类别编号不符。
    """
    first_conv = next((layer for layer in model.model.modules() if isinstance(layer, torch.nn.Conv2d)), None)
    if model.task != "detect" or first_conv is None or (first_conv.in_channels != 5 and not isinstance(model.model, FusionDetectionModel)):
        raise ValueError("必须使用训练后的五通道检测权重，不能使用 orgin_models 中的 RGB 预训练权重")
    if model.names != dict(enumerate(CLASS_NAMES)):
        raise ValueError(f"模型类别编号与比赛 12 类不一致：{model.names}")
    # 框架图片加载器从 YAML 读取通道数，缺省的 3 会把五通道数组裁掉后两通道。
    model.model.yaml["channels"] = 5


# 三、五通道预测与原图坐标恢复
class MultimodalDetectionPredictor(DetectionPredictor):
    """保留五通道输入，仅在构建结果图片时将前三通道恢复为 BGR。"""

    def __init__(self, *args: object, multi_label: bool = False, **kwargs: object) -> None:
        """将候选筛选开关保存在预测器中，不传入框架不支持的配置参数。"""
        super().__init__(*args, **kwargs)
        self.multi_label: bool = multi_label

    def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        """检查 RGB、红外、深度顺序的 uint8 输入，再统一缩放和归一化。"""
        for image in images:
            if image.ndim != 3 or image.shape[2] != 5 or image.dtype != np.uint8:
                raise ValueError("预测输入必须是 RGB、红外、深度顺序的 uint8 五通道图像")
        # 当前框架仅翻转三通道 BGR；五通道保持 fuse_modalities 的 RGBIRDepth 顺序。
        return super().preprocess(images)

    def postprocess(self, preds: torch.Tensor, img: torch.Tensor, orig_imgs: list[np.ndarray], **kwargs: object) -> list[Results]:
        """可选保留同框多类别候选，再按类别 NMS；仍仅使用一个模型。

        Note:
            默认沿用框架单标签筛选。多标签只保留模型已给出的分数，不人为抬高
            ball 置信度；低分候选可能争用每图 100 框的名额，效果需实际提交确认。
        """
        if not isinstance(orig_imgs, list) or getattr(self.model, "end2end", False):
            raise ValueError("批量候选要求五通道 NumPy 图像和 nms=True 的一对多检测头")
        rows = nms.non_max_suppression(
            preds, self.args.conf, self.args.iou, classes=self.args.classes,
            agnostic=self.args.agnostic_nms, multi_label=self.multi_label, max_det=self.args.max_det,
            nc=len(self.model.names), end2end=False, rotated=False,
            max_time_img=float("inf"),  # 离线提交必须处理完整批次，不因实时场景的超时预算跳过后续图片
        )
        return self.construct_results(rows, img, orig_imgs)

    def construct_result(self, pred: torch.Tensor, img: torch.Tensor, orig_img: np.ndarray, img_path: str) -> Results:
        """去掉 LetterBox 缩放和填充，返回原图坐标及可绘制的三通道图片。"""
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        visible: np.ndarray = np.ascontiguousarray(orig_img[:, :, :3][:, :, ::-1])
        return Results(visible, path=img_path, names=self.model.names, boxes=pred[:, :6])


class V4MultimodalDetectionPredictor(MultimodalDetectionPredictor):
    """恢复历史提交 86411da 的 YOLO 原生单标签后处理。

    Note:
        五通道检查与原图坐标恢复沿用相同实现；原生后处理由框架执行。
        该路径仅适用于已训练的 YOLO 五通道权重，不用于 D-FINE。
    """

    def postprocess(self, preds: torch.Tensor, img: torch.Tensor, orig_imgs: list[np.ndarray], **kwargs: object) -> list[Results]:
        """沿用 v4 时未重写的框架后处理，保留原生单标签 NMS。"""
        return DetectionPredictor.postprocess(self, preds, img, orig_imgs, **kwargs)


def predict_yolo_batch(model: YOLO, images: list[np.ndarray], config: PredictionConfig) -> list[Results]:
    """按选定配方预测同尺寸五通道图，保持最小矩形填充口径。

    Args:
        model: 已校验的五通道模型，重复使用同一个实例。
        images: 已同步融合、原尺寸相同的图像列表。
        config: 本次预测参数。

    Returns:
        与输入顺序一致、包含原图坐标与 BGR 图像的检测结果。
    """
    if not images or len({image.shape for image in images}) != 1:
        raise ValueError("YOLO 批量预测要求同批原图尺寸一致，以保持 rect=True 的填充方式")
    legacy = config.yolo_profile == "v4"
    if legacy and (len(images) != 1 or config.multi_label):
        raise ValueError("v4 推理配方要求逐张输入及单标签后处理")
    predictor_class = V4MultimodalDetectionPredictor if legacy else MultimodalDetectionPredictor
    if model.predictor is not None:
        if type(model.predictor) is not predictor_class:
            raise ValueError("当前模型已绑定其他预测器，请重新加载五通道权重")
        model.predictor.multi_label = config.multi_label
    results: list[Results] = model.predict(
        predictor=partial(predictor_class, multi_label=config.multi_label),  # v4 恢复原生后处理，current 保留可选自定义候选
        # 一、输入、检测与精度
        source=images[0] if legacy else images,  # v4 与历史代码一致传原尺寸单张数组
        imgsz=config.imgsz,  # 默认 1280，与训练验证分辨率一致
        rect=True,  # 按相同原尺寸分组，避免混合尺寸批次退化为正方形填充
        conf=config.conf,  # 默认 0.001，为 AP 评测保留低分候选
        iou=config.iou,  # 默认 0.7，与训练验证的 NMS 一致
        nms=True,  # 明确使用带 NMS 的一对多头，与 v5 训练验证一致
        max_det=config.max_det,  # 不超过官方每图 100 框上限
        agnostic_nms=False,  # 不跨类别互相抑制，保留人与车等重叠目标
        classes=None,  # 预测全部 12 类
        augment=False,  # 不引入未经验证的 TTA 或多模型集成
        # 二、设备与性能
        device=config.device,  # 默认本机 CUDA:0，也可显式指定 cpu
        quantize=32,  # FP32 推理基线，不使用 INT8 量化或未验证的精度压缩
        batch=len(images),  # v4 固定单张；current 使用当前数组列表长度
        stream=False,  # 只返回当前批次，不积累整个测试集的 GPU 结果
        compile=False,  # 避免 Windows 上首次编译开销和额外依赖
        channels_last=False,  # 沿用训练的 NCHW 布局
        verbose=False,  # 外层输出样本进度，不重复输出每张图的框架日志
        # 三、输出由本脚本统一管理
        project=str(config.output.parent),  # 不向训练 runs 目录写入预测产物
        name=config.output.name,  # 与用户指定的输出根目录一致
        save_dir=str(config.output),  # 固定位置；防覆盖由 prepare_output 独立保证
        exist_ok=True,  # 框架不另建 predict2；本脚本已经独占创建输出目录
        save=False,  # 手工保存前三通道的可见光带框图片
        save_txt=False,  # 手工排他写入，确保无目标也有空 TXT，避免框架追加写
        save_conf=True,  # 六列 TXT 必须含置信度，由 save_prediction 实际写出
        save_crop=False,  # 不创建额外 crops 文件夹
        show=False,  # 批量预测不弹出窗口
        show_labels=True,  # 图片中保留类别名
        show_conf=True,  # 图片中保留置信度
        show_boxes=True,  # 图片中绘制预测框
        line_width=None,  # 绘图随原图尺寸自动调整线宽
        # 四、其他输入和任务参数：当前静态图检测不使用
        data=None,  # 使用检查点名称，并在预测前核对完整类别顺序
        dnn=False,  # 本地 PyTorch 权重，不使用 ONNX OpenCV 后端
        vid_stride=1,  # 不使用视频抽帧
        stream_buffer=False,  # 不使用实时视频队列
        save_frames=False,  # 静态图无视频帧输出
        visualize=False,  # 不计算激活热图
        embed=None,  # 不输出特征向量
        retina_masks=False,  # 检测任务无分割掩码
    )
    return results


def load_prediction_sample(paths: tuple[Path, Path, Path], backend: str, imgsz: int, height: int = 1080) -> PredictionSample:
    """在读取线程中融合模态，并完成相应后端的同步缩放与填充。"""
    fused = fuse_modalities(*paths)
    if fused.dtype != np.uint8 or fused.ndim != 3 or fused.shape[2] != 5:
        raise ValueError(f"预测输入必须是 uint8 五通道图：{paths[0].name}")
    if backend == "yolo":
        return PredictionSample(paths[0], fused, None, {})
    if backend == "fusion":
        canvas, geometry = letterbox_fused(fused, (height, imgsz))
    else:
        canvas, geometry = resize_dfine_fused(fused, imgsz)
    original_height, width = fused.shape[:2]
    visible = np.ascontiguousarray(fused[:, :, :3][:, :, ::-1])
    target = {"orig_size": torch.tensor([width, original_height]),
              "geometry": torch.tensor(geometry, dtype=torch.float64 if backend == "fusion" else torch.float32)}
    # 预取阶段只保留 uint8；归一化移到 GPU，减少 CPU 拷贝与传输体积。
    return PredictionSample(paths[0], np.ascontiguousarray(canvas.transpose(2, 0, 1)), visible, target)


def prediction_batches(samples: list[tuple[Path, Path, Path]], config: PredictionConfig,
                       reader: ThreadPoolExecutor) -> Iterator[list[PredictionSample]]:
    """并行预读有限窗口并按清单顺序成批返回，读取异常直接交给主线程。"""
    paths = iter(samples)
    pending: deque[Future[PredictionSample]] = deque()
    capacity = config.batch * config.prefetch_batches
    for sample in paths:
        pending.append(reader.submit(load_prediction_sample, sample, config.backend, config.imgsz, config.height))
        if len(pending) == capacity:
            break
    batch: list[PredictionSample] = []
    while pending:
        batch.append(pending.popleft().result())
        next_paths = next(paths, None)
        if next_paths is not None:
            pending.append(reader.submit(load_prediction_sample, next_paths, config.backend, config.imgsz, config.height))
        if len(batch) == config.batch or not pending:
            yield batch
            batch = []


def predict_batch(model: YOLO | DFinePredictor | FusionPredictor, batch: list[PredictionSample],
                  config: PredictionConfig) -> list[Results]:
    """只在主线程调用 GPU，将结果转回 CPU 后交给后台绘图线程。"""
    if config.backend == "yolo":
        if config.yolo_profile == "v4":
            return [predict_yolo_batch(model, [sample.image], config)[0].cpu() for sample in batch]
        return [result.cpu() for result in predict_yolo_batch(model, [sample.image for sample in batch], config)]
    images = torch.from_numpy(np.stack([sample.image for sample in batch]))
    if config.pin_memory and model.device.type == "cuda":
        images = images.pin_memory()
    if config.backend == "fusion":
        predictions = model.predict_batch(images, [sample.target for sample in batch], config.conf, config.max_det, config.iou)
    else:
        predictions = model.predict_batch(images, [sample.target for sample in batch], config.conf, config.max_det)
    results: list[Results] = []
    for sample, prediction in zip(batch, predictions, strict=True):
        boxes = torch.cat((prediction["boxes"], prediction["scores"][:, None], prediction["labels"][:, None].float()), dim=1).cpu()
        results.append(Results(sample.visible, path=str(sample.path), names=model.names, boxes=boxes))
    return results


# 四、标签、图片与提交打包
def prepare_output(output: Path) -> None:
    """独占创建输出根目录；即使已有目录为空也不复用。

    Raises:
        FileExistsError: 输出位置已经存在，需通过 --output 指定新位置。
    """
    if output.exists():
        raise FileExistsError(f"输出目录已存在，不会覆盖：{output}；请用 --output 指定新目录")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("images", "labels", "比赛提交内容"):
        (output / name).mkdir()


def validate_rows(rows: np.ndarray) -> None:
    """检查六列提交格式、类别、置信度及框的归一化边界。

    Raises:
        ValueError: 存在非法预测行、超限或未按置信度降序排列。
    """
    if rows.ndim != 2 or rows.shape[1] != 6 or len(rows) > MAX_DETECTIONS or not np.isfinite(rows).all():
        raise ValueError("预测标签必须为不超过 100 行的有限数值六列表格")
    if not len(rows):
        return
    classes, coords, scores = rows[:, 0], rows[:, 1:5], rows[:, 5]
    if np.any(classes != np.floor(classes)) or np.any((classes < 0) | (classes >= len(CLASS_NAMES))):
        raise ValueError("预测类别必须是 0 至 11 的整数")
    if np.any((coords < 0) | (coords > 1)) or np.any(coords[:, 2:] <= 0):
        raise ValueError("预测坐标必须归一化到 [0,1]，框宽高必须大于 0")
    if np.any(coords[:, :2] - coords[:, 2:] / 2 < -1e-7) or np.any(coords[:, :2] + coords[:, 2:] / 2 > 1 + 1e-7):
        raise ValueError("预测框边界超出原图")
    if np.any((scores < 0) | (scores > 1)) or np.any(np.diff(scores) > 0):
        raise ValueError("置信度必须位于 [0,1] 并按降序排列")


def prediction_rows(result: Results) -> np.ndarray:
    """以原图宽高归一化、过滤退化框，并按置信度保留前 100 项。

    Args:
        result: 已恢复到原图坐标的检测结果。

    Returns:
        class_id、cx、cy、w、h、confidence 六列数组，空预测形状为 (0, 6)。
    """
    if result.boxes is None:
        raise ValueError("检测结果缺少 boxes，不能作为有效的空检测处理")
    boxes = result.boxes.data.detach().cpu().numpy().astype(np.float64, copy=True)
    if boxes.shape[1] != 6 or not np.isfinite(boxes).all():
        raise ValueError("检测输出包含非有限数值或非六列检测框")
    height, width = result.orig_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    boxes = boxes[np.argsort(-boxes[:, 4], kind="stable")[:MAX_DETECTIONS]]
    xywh: np.ndarray = np.column_stack((
        (boxes[:, 0] + boxes[:, 2]) / (2 * width), (boxes[:, 1] + boxes[:, 3]) / (2 * height),
        (boxes[:, 2] - boxes[:, 0]) / width, (boxes[:, 3] - boxes[:, 1]) / height,
    ))
    rows: np.ndarray = np.column_stack((boxes[:, 5], xywh, boxes[:, 4]))
    validate_rows(rows)
    return rows


def save_prediction(result: Results, image_name: str, output: Path, visual_conf: float, png_compression: int) -> None:
    """写入同名带框图片和六列标签；展示阈值不影响提交候选。

    Args:
        result: 已恢复原图坐标的检测结果。
        image_name: 配对清单中的原始图片文件名。
        output: 本次独占的输出根目录。
        visual_conf: 仅用于筛选展示框的置信度。
        png_compression: PNG 无损压缩级别，较低值减少编码耗时但增加文件体积。

    Raises:
        FileExistsError: 同名结果已经存在。
        OSError: 图片无法编码或文件无法写入。
    """
    image_path, label_path = output / "images" / image_name, output / "labels" / f"{Path(image_name).stem}.txt"
    if image_path.exists() or label_path.exists():
        raise FileExistsError(f"预测结果已存在，拒绝覆盖：{image_name}")
    rows = prediction_rows(result)
    content: str = "".join(f"{int(row[0])} " + " ".join(f"{value:.10f}" for value in row[1:]) + "\n" for row in rows)
    # 绘图也使用实际提交的框，只有显示阈值不同；原始 Results 不做阈值修改。
    visible_rows = rows[rows[:, 5] >= visual_conf]
    height, width = result.orig_shape
    xyxy: np.ndarray = np.column_stack((
        (visible_rows[:, 1] - visible_rows[:, 3] / 2) * width, (visible_rows[:, 2] - visible_rows[:, 4] / 2) * height,
        (visible_rows[:, 1] + visible_rows[:, 3] / 2) * width, (visible_rows[:, 2] + visible_rows[:, 4] / 2) * height,
    ))
    display = Results(result.orig_img, path=image_name, names=result.names, boxes=np.column_stack((xyxy, visible_rows[:, 5], visible_rows[:, 0])))
    plotted: np.ndarray = display.plot(conf=True, labels=True, boxes=True, line_width=None, pil=False)
    parameters = [cv2.IMWRITE_PNG_COMPRESSION, png_compression] if image_path.suffix.lower() == ".png" else []
    success, encoded = cv2.imencode(image_path.suffix, plotted, parameters)
    if not success:
        raise OSError(f"预测图片编码失败：{image_path}")
    # 排他创建避免追加写和静默覆盖；空内容也会创建空 TXT。
    with label_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    with image_path.open("xb") as handle:
        handle.write(encoded.tobytes())


def build_submission(output: Path, image_names: list[str]) -> Path:
    """重新核验全部输出后生成仅含根目录 TXT 的 ZIP。

    Args:
        output: 预测结果根目录。
        image_names: 本次完整输入清单中的原始图片文件名。

    Returns:
        完成完整性校验的正式 submission.zip 路径。

    Note:
        打包中断仅留下 submission.zip.partial，不会留下冒充完整提交的 ZIP。
        Windows rename 不覆盖已有文件，符合本项目的防覆盖要求。
    """
    archive: Path = output / "比赛提交内容" / "submission.zip"
    if archive.exists():
        raise FileExistsError(f"提交包已存在，拒绝覆盖：{archive}")
    expected: set[str] = {f"{Path(name).stem}.txt" for name in image_names}
    if not expected or len(expected) != len(image_names):
        raise ValueError("提交清单为空或存在同名标签")
    if {p.name for p in (output / "labels").iterdir()} != expected:
        raise ValueError("标签文件数量或名称与待预测图片不一致，不能生成提交包")
    if {p.name for p in (output / "images").iterdir()} != set(image_names):
        raise ValueError("结果图片数量或名称与待预测图片不一致，不能生成提交包")
    # 将核验过的文本直接打包，不在核验后重新读取可能被外部修改的文件。
    contents: dict[str, str] = {}
    for name in sorted(expected):
        content = (output / "labels" / name).read_text(encoding="utf-8")
        lines = [line.split() for line in content.splitlines()]
        if any(len(line) != 6 or not line[0].isdigit() for line in lines):
            raise ValueError(f"标签不是整数类别开头的六列格式：{name}")
        validate_rows(np.array(lines, dtype=np.float64).reshape(-1, 6))
        contents[name] = content
    temporary: Path = archive.with_suffix(".zip.partial")
    with ZipFile(temporary, "x", compression=ZIP_DEFLATED) as zipped:
        for name, content in contents.items():
            zipped.writestr(name, content.encode("utf-8"))
    with ZipFile(temporary) as zipped:
        if zipped.testzip() is not None or set(zipped.namelist()) != expected:
            raise ValueError("提交 ZIP 完整性校验失败，保留临时文件供检查")
    temporary.rename(archive)
    return archive


# 五、配置覆盖与预测调度
def parse_arguments(config: PredictionConfig, argv: list[str]) -> PredictionConfig:
    """保留既有命令行用法，未传参数时采用 predict.py 中的分节配置。"""
    parser = argparse.ArgumentParser(description="三模态批量预测与赛事提交，默认值见 predict.py", argument_default=argparse.SUPPRESS)
    for name in ("weights", "source", "output"):
        parser.add_argument(f"--{name}", type=Path)
    for name in ("imgsz", "height", "batch", "workers", "save-workers", "prefetch-batches", "expected-count", "max-det", "png-compression", "log-every"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("conf", "iou", "visual-conf"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--backend", choices=("auto", "yolo", "dfine", "fusion"))
    parser.add_argument("--yolo-profile", choices=("v4", "current"), help="仅 YOLO：v4 逐张原生后处理，current 批量自定义后处理")
    parser.add_argument("--device", help="单张 GPU 编号或 cpu")
    parser.add_argument("--multi-label", action=argparse.BooleanOptionalAction, help="仅适用于 YOLO 的多标签 NMS")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, help="D-FINE 使用锁页内存传输当前批次")
    return replace(config, **vars(parser.parse_args(argv)))


def predict(config: PredictionConfig, argv: list[str] | None = None) -> None:
    """并行读取、批量推理、后台保存，全部完成后生成提交 ZIP。

    Args:
        config: 入口文件中显式设置的预测参数。
        argv: 可选命令行覆盖；不传时完全使用 config。

    Note:
        GPU 只在主线程执行。后台保存仅持有 CPU 结果，异常会阻止生成正式 ZIP。
        不自动试跑探测显存，不重训，不修改数据或覆盖旧输出。
    """
    if argv is not None:
        config = parse_arguments(config, argv)
    if config.imgsz <= 0 or config.imgsz % 32:
        raise ValueError("imgsz 必须为 32 的正整数倍")
    if min(config.batch, config.workers, config.save_workers, config.prefetch_batches, config.expected_count, config.log_every) <= 0:
        raise ValueError("批次、线程数、预取批数、图像数量和日志间隔必须为正整数")
    if not 1 <= config.max_det <= MAX_DETECTIONS or not 0 <= config.png_compression <= 9:
        raise ValueError("max_det 必须在 1–100，png_compression 必须在 0–9")
    if not all(0 <= value <= 1 for value in (config.conf, config.iou, config.visual_conf)):
        raise ValueError("conf、iou、visual_conf 必须位于 [0,1]")
    if config.backend not in {"auto", "yolo", "dfine", "fusion"}:
        raise ValueError("backend 必须为 auto、yolo、dfine 或 fusion")
    if config.height <= 0:
        raise ValueError("height必须为正整数")
    if config.yolo_profile not in {"v4", "current"}:
        raise ValueError("yolo_profile 必须为 v4 或 current")
    weights, source, output = ((PROJECT_ROOT / path).resolve() for path in (config.weights, config.source, config.output))
    if output.exists():
        raise FileExistsError(f"输出目录已存在，不会覆盖：{output}；请用 --output 指定新目录")
    if not weights.is_file() or weights.suffix.lower() not in {".pt", ".pth"}:
        raise FileNotFoundError(f"请指定已有的本地 .pt/.pth 五通道检测权重：{weights}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("预测输出与官方源数据目录不能互相包含")
    samples = collect_samples(source)
    if len(samples) != config.expected_count:
        raise ValueError(f"测试集应有 {config.expected_count} 组，实际找到 {len(samples)} 组；请检查 --source")
    with weights.open("rb") as handle:
        weight_hash: str = hashlib.file_digest(handle, "sha256").hexdigest()
    backend = ("dfine" if weights.suffix.lower() == ".pth" else "yolo") if config.backend == "auto" else config.backend
    config = replace(config, weights=weights, source=source, output=output, backend=backend)
    if backend == "dfine":
        if config.multi_label:
            raise ValueError("--multi-label 是 YOLO NMS 开关；D-FINE 使用原生查询类别排序，请移除此开关")
        model = DFinePredictor(weights, config.device, config.imgsz)
        print("D-FINE 使用 FP32、正方形等比填充及原生查询排序；--iou 不参与后处理")
    else:
        if config.yolo_profile == "v4" and config.multi_label:
            raise ValueError("v4 配方使用单标签；多标签预测须显式选择 --yolo-profile current")
        model = YOLO(str(weights), task="detect")
        validate_model(model)
        if isinstance(model.model, FusionDetectionModel):
            if config.multi_label:
                raise ValueError("融合模型验证采用单标签NMS，不允许单独更改提交筛选方式")
            backend = "fusion"
            config = replace(config, backend=backend)
            fusion_epoch = int(model.ckpt["epoch"]) + 1
            model = FusionPredictor(model.model, config.device, (config.height, config.imgsz))
            print(f"融合模型FP32批量预测：内容{config.imgsz}×{config.height}；张量高宽{canvas_shape(model.content_hw)}")
        elif backend == "fusion":
            raise ValueError("--backend fusion要求v9融合模型，不能用于历史五通道权重")
        else:
            print(f"YOLO 推理配方：{config.yolo_profile}；v4 使用逐张矩形填充及框架原生单标签 NMS" if config.yolo_profile == "v4"
                  else "YOLO 推理配方：current；同尺寸批量填充及自定义候选筛选")
    prepare_output(output)
    print(f"预测权重：{weights}\n三模态配对完成，共 {len(samples)} 组；结果写入 {output}")
    model_batch = 1 if backend == "yolo" and config.yolo_profile == "v4" else config.batch
    print(f"FP32 预测：模型批次={model_batch}，调度批次={config.batch}，读取线程={config.workers}，保存线程={config.save_workers}，预取={config.prefetch_batches} 批")
    started: float = time.perf_counter()
    completed: int = 0
    reported: int = 0
    batch_sizes: Counter[int] = Counter()
    pending_saves: deque[Future[None]] = deque()
    previous_cv_threads: int = cv2.getNumThreads()
    # 外层已经并行读写，避免每个线程再启动一组 OpenCV 线程争抢 CPU。
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="模态读取") as reader, \
                ThreadPoolExecutor(max_workers=config.save_workers, thread_name_prefix="结果保存") as writer:
            for batch in prediction_batches(samples, config, reader):
                groups: dict[tuple[int, ...], list[PredictionSample]] = {}
                for sample in batch:
                    # 融合与D-FINE已统一各自画布；旧YOLO按原尺寸分组。
                    groups.setdefault(sample.image.shape, []).append(sample)
                for group in groups.values():
                    results = predict_batch(model, group, config)
                    if backend == "yolo" and config.yolo_profile == "v4":
                        batch_sizes[1] += len(group)
                    else:
                        batch_sizes[len(group)] += 1
                    for sample, result in zip(group, results, strict=True):
                        while pending_saves and (pending_saves[0].done() or len(pending_saves) >= config.batch * config.prefetch_batches):
                            pending_saves.popleft().result()
                        pending_saves.append(writer.submit(save_prediction, result, sample.path.name, output,
                                                           config.visual_conf, config.png_compression))
                    completed += len(group)
                if completed - reported >= config.log_every or completed == len(samples):
                    elapsed = time.perf_counter() - started
                    print(f"已推理 {completed}/{len(samples)}，流水线累计 {elapsed:.1f} 秒；图片与标签后台保存中")
                    reported = completed
            # 所有保存异常必须在打包前传播，不能产生缺图缺标签的正式提交包。
            while pending_saves:
                pending_saves.popleft().result()
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(f"当前 batch={config.batch} 超出可用显存；请减小 batch 并指定新的 --output，已有结果保留") from error
    finally:
        cv2.setNumThreads(previous_cv_threads)
    elapsed = time.perf_counter() - started
    metadata: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(), "weights": weights.name,
        "run": weights.parent.parent.name, "weights_sha256": weight_hash, "source": source.name,
        "images": len(samples), "classes": list(CLASS_NAMES), "ultralytics": ultralytics.__version__,
        "torch": torch.__version__, "imgsz": config.imgsz, "conf": config.conf, "iou": config.iou,
        "visual_conf": config.visual_conf, "nms": True, "multi_label": config.multi_label, "max_det": config.max_det,
        "quantize": 32, "rect": True, "device": config.device, "batch": config.batch, "augment": False,
        "backend": backend, "workers": config.workers, "save_workers": config.save_workers,
        "prefetch_batches": config.prefetch_batches, "pin_memory": config.pin_memory and backend in {"dfine", "fusion"} and config.device != "cpu",
        "png_compression": config.png_compression, "actual_batch_sizes": dict(sorted(batch_sizes.items())),
        "pipeline_seconds": elapsed, "pipeline_images_per_second": len(samples) / elapsed,
    }
    if backend == "dfine":
        metadata.update({"dfine_commit": DFINE_COMMIT, "epoch": model.epoch, "nms": False,
                         "iou": None, "multi_label": None, "rect": False,
                         "postprocess": "native_query_class_topk", "preprocess": "square_letterbox_rgbirdepth_div255",
                         "weights_kind": "ema"})
    elif backend == "fusion":
        metadata.update({"fusion_version": FUSION_VERSION, "content_hw": list(model.content_hw),
                         "epoch": fusion_epoch,
                         "tensor_hw": list(canvas_shape(model.content_hw)), "height": config.height,
                         "yolo_profile": None, "postprocess": "shared_single_label_nms",
                         "preprocess": "fixed_rectangle_rgbirdepth_div255", "effective_model_batch": config.batch,
                         "weights_kind": "ema"})
    else:
        metadata.update({"yolo_profile": config.yolo_profile,
                         "postprocess": "framework_native_single_label_nms" if config.yolo_profile == "v4" else "custom_nms",
                         "effective_model_batch": 1 if config.yolo_profile == "v4" else config.batch,
                         "legacy_source_commit": "86411da" if config.yolo_profile == "v4" else None})
    with (output / "prediction.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    archive = build_submission(output, [sample[0].name for sample in samples])
    print(f"读取、推理、绘图与保存共 {elapsed:.1f} 秒，平均 {len(samples) / elapsed:.2f} 组/秒（不含模型加载与 ZIP 打包）")
    print(f"提交包校验通过，共 {len(samples)} 个 TXT：{archive}\n初赛只提交此 ZIP；images、prediction.json 不放入提交包。")


# Windows 子进程或其他模块可能导入入口，正式预测必须放在入口保护内。
if __name__ == "__main__":
    predict(PredictionConfig(
        # 一、模型、数据与输出
        weights=MODEL_PATH,  # v9训练完成后使用同一训练轨迹的EMA最佳模型
        source=SOURCE_PATH,  # 只读官方三模态图像，不生成缓存或修改原始文件
        output=OUTPUT_PATH,  # 保存带框图片、六列标签和提交 ZIP，已有目录不覆盖
        backend="auto",  # 从检查点识别融合结构；仍兼容旧YOLO和D-FINE
        expected_count=1000,  # 已确认初赛为 1000 组，仅核对文件清单避免交错目录

        # 二、设备、加载与资源：由用户调整，不自动试跑探测显存
        imgsz=1920,  # 与v9训练一致的内容宽度，原1920宽图片不先缩到1280
        height=1080,  # 内容高度1080；网络画布仅补齐到1088，不拉伸
        batch=4,  # 全尺寸新结构FP32批量起点，由用户调整，不自动探测
        workers=8,  # 并行解码三模态并提前缩放，给 GPU 连续准备输入
        save_workers=8,  # 后台画框、编码和写文件，不再逐张阻塞下一次推理
        prefetch_batches=2,  # 全尺寸图片预取2批，控制五通道与待保存原图内存
        pin_memory=True,  # 当前批次uint8锁页传输，在GPU转FP32并归一化
        device="0",  # 本机 RTX 5080；保持 FP32，不默认使用半精度或 INT8

        # 三、检测候选与后处理：保持比赛口径，不用降低精度换速度
        conf=0.001,  # 保留低分候选计算 AP，与验证一致，不是图片展示阈值
        iou=0.7,  # v9与轮末验证共享单标签NMS；D-FINE仍忽略此参数
        max_det=100,  # 赛事每图最多 100 框，按置信度保留，不做多模型集成
        multi_label=False,  # YOLO 默认单标签；D-FINE 原生查询类别排序不受此开关控制
        yolo_profile="v4",  # 仅用于旧YOLO；v9自动采用训练共用的矩形批量后端

        # 四、可视化与进度：只影响输出耗时和图片，不改提交预测框
        visual_conf=0.25,  # 仅绘制较高置信度框，六列 TXT 仍保留 conf 以上候选
        png_compression=1,  # PNG 低级别无损压缩，以磁盘空间换编码速度，JPG 保持默认
        log_every=25,  # 按批次报告累计进度，结束后统计包含读图和保存的实际吞吐量
    ), argv=sys.argv[1:])
