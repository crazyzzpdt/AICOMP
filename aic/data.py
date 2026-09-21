"""共享赛事类别、三模态读图和文件指纹，不依赖训练器或数据清洗工具。"""

# 内置库
import hashlib
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

    # 官方同名三模态已对齐；尺寸异常应报告，不能通过拉伸掩盖错配。
    if not visible.shape[:2] == infrared_channel.shape[:2] == depth_channel.shape[:2]:
        raise ValueError(f"三模态尺寸不一致：{visible_path.name}")
    return np.dstack((visible, infrared_channel, depth_channel))
