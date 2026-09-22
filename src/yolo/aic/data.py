"""共享赛事类别、三模态读图和文件指纹，不依赖训练器或数据清洗工具。"""

# 内置库
import hashlib
from collections.abc import Callable
from pathlib import Path

# 三方库
import cv2
import numpy as np


IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
CLASS_NAMES: tuple[str, ...] = (
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage can",
    "uav",
    "tricycle",
)

# 沿用既有毫米深度上限，训练与预测使用同一换算。
DEPTH_MAX_MM: int = 20_000
# 仅用于v19原始毫米深度支持判断；JPG灰度不套用距离阈值。
DEPTH_MIN_MM: int = 300
# 六个张量通道仍来自三种传感器，第六通道是深度派生的支持比例。
QUALITY_PREPROCESS_VERSION: str = "rgbirdepth_support_weighted_uint8_v19_1"


def file_hash(path: Path) -> str:
    """计算文件内容指纹，用于审计和恢复校验。"""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_source_images(root: Path, samples: list[dict[str, object]]) -> None:
    """核对审计中的三模态内容指纹，拒绝同名文件被静默替换。

    Args:
        root: 含 train/val 的数据根目录。
        samples: 已落位审计的样本记录，必须包含官方来源哈希。

    Raises:
        ValueError: 记录缺少指纹，或图像内容与已审阅来源不同。
    """
    for sample in samples:
        hashes = sample.get("source_hashes", {})
        for folder, modality in (("images", "visible"), ("infrared", "infrared"), ("depth", "depth")):
            path = root / str(sample["after_split"]) / folder / str(sample["image"])
            expected = hashes.get(modality)
            if not expected or file_hash(path) != expected:
                raise ValueError(f"三模态来源指纹不一致或缺失：{modality}/{sample['image']}")


def read_image(path: Path, flags: int) -> np.ndarray:
    """读取单张图像，并兼容 Windows 中文路径。

    Args:
        path: 待读取的图像文件。
        flags: OpenCV 图像读取模式。

    Returns:
        成功解码的图像数组。

    Raises:
        FileNotFoundError: 图像不存在或无法解码。
    """
    # Windows 下 OpenCV 的 imread 可能无法处理中文路径，改由 NumPy 读取文件字节后再解码。
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise FileNotFoundError(f"无法读取三模态图像：{path}") from error
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise FileNotFoundError(f"无法读取三模态图像：{path}")
    return image


def convert_to_single_channel(image: np.ndarray) -> np.ndarray:
    """将红外或深度图转换为单通道表示。

    Args:
        image: 红外或低分辨率深度图。

    Returns:
        输入图像的单通道表示。
    """
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_depth_channel(depth: np.ndarray) -> np.ndarray:
    """将深度图转换为模型使用的 8 位单通道。

    PNG 深度图保留原始 uint16 毫米值读取，并在 0 至 20000 毫米内线性归一化；
    官方 JPG 深度图没有毫米值，因此仅转为灰度，不假设其距离尺度。

    Args:
        depth: 读取后的官方深度图。

    Returns:
        范围为 0 至 255 的 uint8 深度通道。
    """
    if depth.dtype == np.uint16:
        clipped = np.clip(depth, 0, DEPTH_MAX_MM)
        return np.rint(clipped * 255.0 / DEPTH_MAX_MM).astype(np.uint8)
    return convert_to_single_channel(depth)


def fuse_modalities(visible_path: Path, infrared_path: Path, depth_path: Path) -> np.ndarray:
    """将同一词干的 RGB、红外和深度图合成为五通道输入。

    Args:
        visible_path: 官方可见光 RGB 图路径。
        infrared_path: 官方红外图路径。
        depth_path: 官方深度图路径。

    Returns:
        RGB、红外、归一化深度顺序的 uint8 五通道图像。

    Raises:
        ValueError: 三张图尺寸不一致。
    """
    if len({visible_path.name, infrared_path.name, depth_path.name}) != 1:
        raise ValueError("三模态必须使用同名样本，不能混合不同帧")
    visible = cv2.cvtColor(read_image(visible_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    infrared = read_image(infrared_path, cv2.IMREAD_UNCHANGED)
    depth = read_image(depth_path, cv2.IMREAD_UNCHANGED)
    infrared_channel = convert_to_single_channel(infrared)
    depth_channel = normalize_depth_channel(depth)

    # 同名同尺寸只确认样本配对，不能据此证明像素级对齐；异常不能靠拉伸掩盖。
    if not visible.shape[:2] == infrared_channel.shape[:2] == depth_channel.shape[:2]:
        raise ValueError(f"三模态尺寸不一致：{visible_path.name}")
    return np.dstack((visible, infrared_channel, depth_channel))


def fuse_quality_modalities(visible_path: Path, infrared_path: Path, depth_path: Path) -> np.ndarray:
    """从原始深度生成支持掩码，再组成RGB、IR、Depth、支持比例六通道。

    Note:
        原始PNG有效范围为[300,20000]毫米；JPG非零仅是可用性代理。
        不更改源图，不对全零深度伪造数据，旧权重仍使用fuse_modalities。
    """
    if len({visible_path.name, infrared_path.name, depth_path.name}) != 1:
        raise ValueError("三模态必须使用同名样本，不能混合不同帧")
    visible = cv2.cvtColor(read_image(visible_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    infrared = convert_to_single_channel(read_image(infrared_path, cv2.IMREAD_UNCHANGED))
    raw_depth = convert_to_single_channel(read_image(depth_path, cv2.IMREAD_UNCHANGED))
    if not visible.shape[:2] == infrared.shape == raw_depth.shape or infrared.dtype != np.uint8:
        raise ValueError(f"v19要求同尺寸RGB/8位IR/深度：{visible_path.name}")
    if raw_depth.dtype == np.uint16:
        support = (raw_depth >= DEPTH_MIN_MM) & (raw_depth <= DEPTH_MAX_MM)
    elif raw_depth.dtype == np.uint8:
        support = raw_depth > 0
    else:
        raise ValueError(f"不支持的深度类型：{raw_depth.dtype}")
    depth = normalize_depth_channel(raw_depth)
    depth = np.where(support, depth, 0).astype(np.uint8)
    return np.dstack((visible, infrared, depth, support.astype(np.uint8) * 255))


def transform_quality_image(image: np.ndarray, operation: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
    """同步变换六通道，以支持加权插值避免缺测零拉低有效深度。

    Args:
        image: uint8 RGB3+IR1+Depth1+Support1，支持通道255表示完全支持。
        operation: 接受单通道浮点图与边界值的同一几何变换。

    Returns:
        六通道uint8；支持是插值后的比例而非重新阈值化的二值掩码。
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 6:
        raise ValueError("v19几何处理要求uint8六通道输入")
    support = image[:, :, 5].astype(np.float32) / 255.0
    transformed_support = np.clip(operation(support, 0.0), 0.0, 1.0)
    numerator = operation(image[:, :, 4].astype(np.float32) * support, 0.0)
    depth = np.divide(numerator, transformed_support, out=np.zeros_like(numerator), where=transformed_support > 1e-6)
    mask = np.rint(transformed_support * 255).astype(np.uint8)
    depth[mask == 0] = 0
    planes = [np.clip(operation(image[:, :, i].astype(np.float32), 114.0 if i < 3 else 0.0), 0, 255)
              for i in range(4)]
    return np.dstack((*[np.rint(plane).astype(np.uint8) for plane in planes],
                      np.rint(np.clip(depth, 0, 255)).astype(np.uint8), mask))


def resize_quality_image(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """使用共同双线性几何缩放RGB/IR与支持加权深度。"""
    if image.shape[:2] == size_wh[::-1]:
        return image
    return transform_quality_image(image, lambda plane, border: cv2.resize(plane, size_wh, interpolation=cv2.INTER_LINEAR))
