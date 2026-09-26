"""当前浮点训练的数据几何及检查点边界，不运行额外网络前向。"""

from copy import deepcopy

import numpy as np
import pytest
import torch

from src.modalities import letterbox_float, resize_float_image
from src.yolo.aic.training import FloatLetterBox, FloatPerspective, FloatYOLODataset, copy_ema_to_cpu


@pytest.mark.parametrize("shape", [(108, 192), (191, 107), (77, 128)])
def test_float_validation_matches_prediction(shape):
    """实际验证的两段缩放与预测预处理逐像素一致，包括奇数补边。"""
    rng = np.random.default_rng(0)
    image = rng.uniform(1, 254, (*shape, 5)).astype(np.float32)
    image[::3, ::2, 4] = 0
    dataset = FloatYOLODataset.__new__(FloatYOLODataset)
    dataset.imgsz = 128
    actual = FloatLetterBox(new_shape=(128, 128), scaleup=False)(image=dataset.resize_native_image(image))
    expected, _ = letterbox_float(image, 128, native=True)
    np.testing.assert_array_equal(actual, expected)


def test_float_geometry_keeps_depth_and_modalities_aligned():
    """变换标签使用同一矩阵；辅助通道不补114，深度不降至uint8。"""
    image = np.zeros((8, 8, 5), np.float32)
    image[2:4, 2:4] = [101.25, 102.25, 103.25, 104.25, 105.25]
    transform = FloatPerspective(degrees=0, translate=0, scale=0, shear=0, perspective=0, size=(8, 8))
    matrix = np.eye(3)
    matrix[0, 2], matrix[1, 2] = 2, 1
    result = transform.apply_image({"img": image}, {"M": matrix, "size": (8, 8)})["img"]
    np.testing.assert_array_equal(result[3:5, 4:6], image[2:4, 2:4])
    np.testing.assert_array_equal(result[0, 0], [114, 114, 114, 0, 0])
    assert result.dtype == np.float32
    resized = resize_float_image(image, (16, 16))
    depth = resized[:, :, 4]
    np.testing.assert_allclose(depth[depth > 0], 105.25)


def test_cpu_snapshot_does_not_mutate_live_model():
    """保存时移除损失对象并复制动态缓存，不能共享/修改训练中的参数。"""
    model = torch.nn.Sequential(torch.nn.Conv2d(5, 4, 3), torch.nn.BatchNorm2d(4))
    model.criterion = {"temporary": torch.ones(3)}
    model[0].anchors = torch.ones(2, 3)
    original = deepcopy(model.state_dict())
    snapshot = copy_ema_to_cpu(model)
    assert snapshot.criterion is None and model.criterion is not None
    assert snapshot[0].anchors.data_ptr() != model[0].anchors.data_ptr()
    with torch.no_grad():
        next(snapshot.parameters()).zero_()
    for name, value in original.items():
        torch.testing.assert_close(model.state_dict()[name], value)
