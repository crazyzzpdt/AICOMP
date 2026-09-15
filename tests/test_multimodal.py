"""验证 RGB、红外和深度图的五通道融合契约。"""

# 内置库
from pathlib import Path

# 三方库
import cv2
import numpy as np
import pytest

# 自己的模块
from 三模态训练 import DEPTH_MAX_MM, fuse_modalities


ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = ROOT / "数据集" / "训练集" / "AIC2026_Train_2000"


def read_image(path: Path, flags: int) -> np.ndarray:
    """通过字节解码读取含中文路径的测试图像。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def test_fuse_png_triplet_keeps_rgb_infrared_and_metric_depth() -> None:
    """同名 PNG 三模态图必须融合为 RGB、红外和深度五个通道。"""
    stem = "000002_080_00000048"
    image = fuse_modalities(
        RAW_DIR / "visible" / f"{stem}.png",
        RAW_DIR / "infrared" / f"{stem}.png",
        RAW_DIR / "depth" / f"{stem}.png",
    )

    visible = read_image(RAW_DIR / "visible" / f"{stem}.png", cv2.IMREAD_COLOR)
    infrared = read_image(RAW_DIR / "infrared" / f"{stem}.png", cv2.IMREAD_UNCHANGED)
    depth = read_image(RAW_DIR / "depth" / f"{stem}.png", cv2.IMREAD_UNCHANGED)
    expected_infrared = cv2.resize(
        cv2.cvtColor(infrared, cv2.COLOR_BGR2GRAY), (visible.shape[1], visible.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    expected_depth = cv2.resize(
        np.rint(np.clip(depth, 0, DEPTH_MAX_MM) * 255.0 / DEPTH_MAX_MM).astype(np.uint8),
        (visible.shape[1], visible.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    assert image.dtype == np.uint8
    assert image.shape == (*visible.shape[:2], 5)
    np.testing.assert_array_equal(image[:, :, :3], visible[:, :, ::-1])
    np.testing.assert_array_equal(image[:, :, 3], expected_infrared)
    np.testing.assert_array_equal(image[:, :, 4], expected_depth)


def test_fuse_jpg_triplet_uses_single_depth_channel_without_metric_assumption() -> None:
    """8 位 JPG 深度图没有毫米值，融合器必须保留单通道图像值。"""
    stem = "00000004"
    image = fuse_modalities(
        RAW_DIR / "visible" / f"{stem}.jpg",
        RAW_DIR / "infrared" / f"{stem}.jpg",
        RAW_DIR / "depth" / f"{stem}.jpg",
    )

    depth = read_image(RAW_DIR / "depth" / f"{stem}.jpg", cv2.IMREAD_UNCHANGED)
    expected_depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    assert image.shape[2] == 5
    np.testing.assert_array_equal(image[:, :, 4], expected_depth)


def test_fuse_rejects_different_frames() -> None:
    """即使可以拉伸尺寸，也不能将不同帧融合为一个样本。"""
    stem = "000002_080_00000048"
    with pytest.raises(ValueError, match="同名"):
        fuse_modalities(
            RAW_DIR / "visible" / f"{stem}.png",
            RAW_DIR / "infrared" / f"{stem}.png",
            RAW_DIR / "depth" / "00000004.jpg",
        )
