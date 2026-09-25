"""D-FINE独立预测，原生查询排序，不导入YOLO项目模块。"""
from __future__ import annotations

from src.prediction_io import (PROJECT_ROOT, OutputConfig, PredictionSample, parse_arguments, run_prediction)
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from src.modalities import CLASS_NAMES, FLOAT_PREPROCESS_VERSION, configure_fp32, read_float_modalities, letterbox_float
from src.dfine.runtime import (build_model as build_dfine_runtime_model,
                               decode_predictions as decode_dfine_runtime_predictions)

DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
DFINE_PREPROCESS_VERSION = "rgbirdepth_uint8_letterbox_div255_v1"
DFINE_CHECKPOINT_FORMAT = "aic_dfine_l_5ch_v1"
SOURCE_PATHS = ("predict2.py", "src/__init__.py", "src/prediction_io.py", "src/modalities.py",
                "src/dfine", "src/D-FINE")


@dataclass(frozen=True)
class PredictionConfig(OutputConfig):
    """原生查询后处理不包含YOLO的NMS或多标签开关。"""

    imgsz: int = 0


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


class DFinePredictor:
    """加载本项目 D-FINE 检查点，以训练同口径处理单张或批量五通道图像。"""

    def __init__(self, weights: Path, device: str, imgsz: int) -> None:
        """核对历史检查点格式，并加载EMA模型，不恢复训练状态。"""
        self.device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        configure_fp32()
        self.imgsz = imgsz
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != DFINE_CHECKPOINT_FORMAT or checkpoint.get("dfine_commit") != DFINE_COMMIT:
            raise ValueError("不是本项目固定版本的 D-FINE 五通道权重")
        if checkpoint.get("preprocess") not in {DFINE_PREPROCESS_VERSION, FLOAT_PREPROCESS_VERSION} or list(checkpoint["classes"]) != list(CLASS_NAMES):
            raise ValueError("权重的预处理或类别顺序与当前预测代码不符")
        if checkpoint["config"]["imgsz"] != imgsz:
            raise ValueError("首版 D-FINE 预测必须与训练 imgsz 一致，避免隐式更改位置编码与锚点")
        self.preprocess = checkpoint["preprocess"]
        self.continuous_depth = self.preprocess == FLOAT_PREPROCESS_VERSION
        self.training_sensor_augmentation = checkpoint["config"].get("sensors")
        self.input_geometry = "float_dfine" if self.continuous_depth else "fixed_rect"
        self.model, _ = build_dfine_runtime_model(imgsz)
        self.model.load_state_dict(checkpoint["ema"]["module"], strict=True)
        self.model.to(self.device).float().eval()
        self.names: dict[int, str] = dict(enumerate(CLASS_NAMES))
        self.epoch: int = checkpoint["epoch"] + 1

    @torch.inference_mode()
    def predict_batch(self, images: torch.Tensor, targets: list[dict[str, torch.Tensor]],
                      conf: float, max_det: int = 100) -> list[dict[str, torch.Tensor]]:
        """整批传入 GPU，以 FP32 前向并分别还原原图坐标。

        Args:
            images: CPU NCHW五通道；新协议float32，历史协议uint8，可使用锁页内存。
            targets: 每张原图的尺寸与缩放、填充信息，顺序必须与批次一致。
            conf: 提交候选的最低置信度。
            max_det: 每图保留的候选框上限。

        Returns:
            与输入顺序一致的原图像素坐标、置信度和类别。
        """
        expected_dtype = torch.float32 if self.continuous_depth else torch.uint8
        if images.dtype != expected_dtype or images.ndim != 4 or tuple(images.shape[1:]) != (5, self.imgsz, self.imgsz):
            raise ValueError(f"D-FINE批次要求{expected_dtype}的[N,5,imgsz,imgsz]")
        if not len(images) or len(images) != len(targets):
            raise ValueError("D-FINE 批次不能为空，且图像与坐标信息数量必须一致")
        # 两套协议均只除255一次；新协议在读图/几何阶段已保留连续浮点。
        inputs = images.to(self.device, non_blocking=images.is_pinned()).float() / 255
        with torch.autocast(device_type=self.device.type, enabled=False):
            outputs = self.model(inputs)
        return decode_dfine_runtime_predictions(outputs, targets, self.imgsz, conf, max_det)


# 二、输入配对与模型检查
def read_legacy_modalities(paths: tuple[Path, Path, Path]) -> np.ndarray:
    """严格保留早期D-FINE的8位输入协议，新权重走共享连续FP32读取。"""
    def read(path: Path, flags: int) -> np.ndarray:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
        if image is None:
            raise FileNotFoundError(f"无法读取三模态图像：{path}")
        return image

    def gray(image: np.ndarray) -> np.ndarray:
        return image if image.ndim == 2 else image[:, :, 0] if image.shape[2] == 1 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rgb = cv2.cvtColor(read(paths[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    infrared = gray(read(paths[1], cv2.IMREAD_UNCHANGED))
    raw_depth = read(paths[2], cv2.IMREAD_UNCHANGED)
    depth = (np.rint(np.clip(raw_depth, 0, 20000) * 255.0 / 20000).astype(np.uint8)
             if raw_depth.dtype == np.uint16 else gray(raw_depth))
    if not rgb.shape[:2] == infrared.shape[:2] == depth.shape[:2]:
        raise ValueError(f"三模态尺寸不一致：{paths[0].name}")
    return np.dstack((rgb, infrared, depth))


def load_sample(paths: tuple[Path, Path, Path], model: DFinePredictor) -> PredictionSample:
    """按检查点输入协议读取三种模态，同步缩放并保留原图坐标。"""
    image = read_float_modalities(*paths)[0] if model.continuous_depth else read_legacy_modalities(paths)
    dtype = np.float32 if model.continuous_depth else np.uint8
    if image.dtype != dtype or image.ndim != 3 or image.shape[2] != 5:
        raise ValueError("D-FINE输入的精度或五通道结构与权重不符")
    canvas, geometry = (letterbox_float(image, model.imgsz, native=False) if model.continuous_depth
                        else resize_dfine_fused(image, model.imgsz))
    height, width = image.shape[:2]
    visible = np.ascontiguousarray(image[:, :, :3][:, :, ::-1]).astype(np.uint8)
    target = {"orig_size": torch.tensor([width, height]), "geometry": torch.tensor(geometry, dtype=torch.float32)}
    return PredictionSample(paths[0], np.ascontiguousarray(canvas.transpose(2, 0, 1)), visible, target)


def forward_batch(batch: list[PredictionSample], model: DFinePredictor, config: PredictionConfig) -> list[Results]:
    """只在主线程调用GPU，后台仅保存CPU结果。"""
    images = torch.from_numpy(np.stack([sample.image for sample in batch]))
    if config.pin_memory and model.device.type == "cuda":
        images = images.pin_memory()
    predictions = model.predict_batch(images, [sample.target for sample in batch], config.conf, config.max_det)
    results = []
    for sample, prediction in zip(batch, predictions, strict=True):
        boxes = torch.cat((prediction["boxes"], prediction["scores"][:, None], prediction["labels"][:, None].float()), dim=1).cpu()
        results.append(Results(sample.visible, path=str(sample.path), names=model.names, boxes=boxes))
    return results


def create_backend(config: PredictionConfig):
    """返回D-FINE读图、前向及元数据，共用输出层不负责模型选择。"""
    model = DFinePredictor(config.weights, config.device, config.imgsz)
    metadata = {
        "backend": "dfine", "dfine_commit": DFINE_COMMIT, "epoch": model.epoch,
        "nms": False, "iou": None, "multi_label": None, "rect": False,
        "postprocess": "native_query_class_topk", "preprocess": model.preprocess, "weights_kind": "ema",
        "input_dtype": "float32" if model.continuous_depth else "uint8",
        "continuous_modalities": model.continuous_depth,
        "training_sensor_augmentation": model.training_sensor_augmentation,
        "effective_model_batch": config.batch, "pin_memory": config.pin_memory and model.device.type == "cuda",
    }
    return partial(load_sample, model=model), partial(forward_batch, model=model, config=config), metadata


def predict(config: PredictionConfig, argv: list[str] | None = None) -> None:
    """只加载本项目D-FINE权重；尺寸在命令行覆盖后从最终权重读取。"""
    if argv is not None:
        config = parse_arguments(config, argv)
    if config.weights.suffix.lower() != ".pth":
        raise ValueError("predict2.py仅接收本项目D-FINE .pth权重")
    if config.imgsz == 0:
        checkpoint = torch.load((PROJECT_ROOT / config.weights).resolve(strict=True), map_location="cpu", weights_only=False)
        size = checkpoint.get("config", {}).get("imgsz") if isinstance(checkpoint, dict) else None
        if type(size) is not int or size <= 0 or size % 32:
            raise ValueError("权重缺少合法训练imgsz，不能猜测输入尺寸")
        config = replace(config, imgsz=size, height=size)
        del checkpoint
    run_prediction(config, create_backend, SOURCE_PATHS, "predict2.py")
