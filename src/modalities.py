"""提供不依赖检测框架的连续浮点三模态输入、几何和传感器增强。

内部保持float32的0–255连续值以兼容YOLO格式化，网络只除255一次。
这里只操作内存，不修改官方图像、标签或历史权重的预处理。
"""

from __future__ import annotations

# 内置库
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# 三方库
import cv2
import numpy as np
import torch


# 新权重必须记录此协议，旧权重不能静默采用新深度插值。
FLOAT_PREPROCESS_VERSION: str = "rgbirdepth_continuous_fp32_support_v25_1"
DEPTH_MAX_MM: float = 20_000.0
CLASS_NAMES: tuple[str, ...] = ("person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
                               "light", "garbage can", "uav", "tricycle")
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class SensorAugment:
    """记录温和传感器扰动幅度，所有噪声尺度按0–1输入定义。"""

    ir_probability: float = 0.3
    ir_gain: float = 0.05
    ir_bias: float = 0.01
    ir_noise_probability: float = 0.1
    ir_noise_std: float = 0.003
    depth_probability: float = 0.2
    depth_scale: float = 0.01
    depth_noise_base: float = 0.0005
    depth_noise_distance: float = 0.0015
    depth_dropout_probability: float = 0.1
    depth_dropout_fraction: float = 0.005
    # 新增增强默认关闭，由待训练v25入口显式开启。
    ir_gamma_probability: float = 0.0
    ir_local_probability: float = 0.0
    rgb_exposure_probability: float = 0.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"传感器增强参数{name}须为[0,1]有限值")


def configure_fp32() -> None:
    """禁用TF32；调用方另行禁用autocast，权重和输入保持FP32。"""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def read_plane(path: Path, color: bool = False) -> np.ndarray:
    """兼容中文路径读图，保留深度位宽并转单通道。"""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法解码图像：{path}")
    if color:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3:
        image = image[:, :, 0] if image.shape[2] == 1 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def read_float_modalities(visible: Path, infrared: Path, depth: Path) -> tuple[np.ndarray, bool]:
    """读取连续五通道及毫米深度标志，不猜测8位代理图的物理距离。"""
    if len({visible.name, infrared.name, depth.name}) != 1:
        raise ValueError("三模态必须同名配对")
    rgb, ir, raw = read_plane(visible, True), read_plane(infrared), read_plane(depth)
    if rgb.shape[:2] != ir.shape or ir.shape != raw.shape:
        raise ValueError(f"三模态尺寸不一致：{visible.name}")
    if ir.dtype not in (np.uint8, np.uint16) or raw.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"不支持的IR/深度位宽：{ir.dtype}/{raw.dtype}")
    metric = raw.dtype == np.uint16
    ir_float = ir.astype(np.float32) * (255.0 / np.iinfo(ir.dtype).max)
    depth_float = raw.astype(np.float32)
    if metric:
        valid = (raw > 0) & (raw <= DEPTH_MAX_MM)
        depth_float = np.where(valid, depth_float * (255.0 / DEPTH_MAX_MM), 0.0)
    # 8位代理没有可恢复的毫米精度；不把插值后的浮点值宣传为新增传感器信息。
    result = np.empty((*ir.shape, 5), dtype=np.float32)
    result[:, :, :3] = rgb
    result[:, :, 3] = ir_float
    result[:, :, 4] = depth_float
    return result, metric


def transform_float_image(image: np.ndarray,
                          operation: Callable[[np.ndarray, float, int], np.ndarray]) -> np.ndarray:
    """共享线性变换与最近邻有效性，避免缺失零值拉低有效深度。"""
    if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 5:
        raise ValueError("新协议要求float32五通道")
    support = (image[:, :, 4] > 0).astype(np.float32)
    mass = operation(support, 0.0, cv2.INTER_LINEAR)
    valid = operation(support, 0.0, cv2.INTER_NEAREST) > 0.5
    numerator = operation(image[:, :, 4] * support, 0.0, cv2.INTER_LINEAR)
    result = np.empty((*numerator.shape, 5), dtype=np.float32)
    result[:, :, 4] = 0.0
    np.divide(numerator, mass, out=result[:, :, 4], where=(mass > 1e-6) & valid)
    # 深度中间量先释放，再逐通道写入，避免dstack与clip同时持有两份五通道画布。
    del support, mass, valid, numerator
    for channel in range(4):
        result[:, :, channel] = operation(image[:, :, channel], 114.0 if channel < 3 else 0.0, cv2.INTER_LINEAR)
    np.clip(result, 0.0, 255.0, out=result)
    return result


def resize_float_image(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """逐通道浮点缩放，避开OpenCV五通道限制。"""
    if image.shape[:2] == size_wh[::-1]:
        return image
    return transform_float_image(image, lambda plane, border, mode: cv2.resize(plane, size_wh, interpolation=mode))


def float_canvas(height: int, width: int) -> np.ndarray:
    """RGB填114，辅助模态填0，画布始终FP32。"""
    canvas = np.zeros((height, width, 5), dtype=np.float32)
    canvas[:, :, :3] = 114.0
    return canvas


def letterbox_float(image: np.ndarray, imgsz: int, *, native: bool = False,
                    scale: float = 1.0) -> tuple[np.ndarray, tuple[float, float, int, int]]:
    """统一正方形填充；YOLO沿用ceil，D-FINE沿用round取整。"""
    h, w = image.shape[:2]
    if not 0 < scale <= 1 or imgsz < 1:
        raise ValueError("填充尺度须位于(0,1]且imgsz为正数")
    ratio = imgsz / max(h, w) * scale
    rounding = np.ceil if native else round
    nw, nh = min(imgsz, max(1, int(rounding(w * ratio)))), min(imgsz, max(1, int(rounding(h * ratio))))
    left, top = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas = float_canvas(imgsz, imgsz)
    canvas[top:top + nh, left:left + nw] = resize_float_image(image, (nw, nh))
    return canvas, (nw / w, nh / h, left, top)


def enhance_ir_local(ir: np.ndarray) -> np.ndarray:
    """限制局部细节增益及幅度，保持FP32，不使用整数直方图均衡。

    Args:
        ir: 0–1范围单通道红外图。

    Returns:
        连续浮点局部对比度增强结果；不是温度校正或热伪目标检测。
    """
    mean = cv2.GaussianBlur(ir, (0, 0), sigmaX=5.0)
    variance = np.maximum(cv2.GaussianBlur(ir * ir, (0, 0), sigmaX=5.0) - mean * mean, 0.0)
    std = np.sqrt(variance)
    # 平坦区接近零增益；有纹理时温和增强，避免无上限放大暗区噪声。
    reliability = std / (std + 0.02)
    detail = np.clip((ir - mean) * (0.3 * reliability), -0.05, 0.05)
    return np.clip(ir + detail, 0.0, 1.0).astype(np.float32, copy=False)


def augment_sensors(image: np.ndarray, metric_depth: bool, config: SensorAugment) -> np.ndarray:
    """只在训练使用，保留缺失深度，不改变三模态几何或标签。"""
    result = image.copy()
    if config.rgb_exposure_probability and np.random.random() < config.rgb_exposure_probability:
        rgb = np.clip(result[:, :, :3] / 255.0, 0.0, 1.0)
        # 整张RGB共享色调参数，保留色彩关系；不对IR或Depth套用RGB曝光。
        rgb = np.power(rgb, np.float32(np.random.uniform(0.7, 1.8)))
        rgb *= np.float32(np.random.uniform(0.6, 1.05))
        rgb += np.random.normal(0.0, 0.003, rgb.shape).astype(np.float32)
        result[:, :, :3] = np.clip(rgb, 0.0, 1.0) * 255.0
    ir = result[:, :, 3] / 255.0
    if config.ir_gamma_probability and np.random.random() < config.ir_gamma_probability:
        ir = np.power(np.clip(ir, 0.0, 1.0), np.float32(np.random.uniform(0.8, 1.2)))
    if config.ir_local_probability and np.random.random() < config.ir_local_probability:
        ir = enhance_ir_local(ir)
    if np.random.random() < config.ir_probability:
        ir = ir * np.random.uniform(1 - config.ir_gain, 1 + config.ir_gain) + np.random.uniform(-config.ir_bias, config.ir_bias)
    if np.random.random() < config.ir_noise_probability:
        ir = ir + np.random.normal(0, config.ir_noise_std, ir.shape).astype(np.float32)
    result[:, :, 3] = np.clip(ir, 0, 1) * 255.0
    depth = result[:, :, 4] / 255.0
    valid = depth > 0
    if metric_depth and np.random.random() < config.depth_probability:
        sigma = config.depth_noise_base + config.depth_noise_distance * depth ** 2
        depth = depth * np.random.uniform(1 - config.depth_scale, 1 + config.depth_scale)
        depth += np.random.normal(0, 1, depth.shape).astype(np.float32) * sigma
    if np.random.random() < config.depth_dropout_probability:
        valid &= np.random.random(depth.shape) >= config.depth_dropout_fraction
    result[:, :, 4] = np.where(valid, np.clip(depth, 1e-8, 1.0), 0.0) * 255.0
    return result.astype(np.float32, copy=False)
