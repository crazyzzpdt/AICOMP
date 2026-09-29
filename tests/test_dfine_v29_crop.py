"""D-FINE v29数据边界检查：合成五通道CPU图，无模型前向或正式训练。"""

from pathlib import Path
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from src.augmentation import SmallObjectCrop, choose_small_object_crop
from src.dfine.training import MultimodalDFineDataset, collate_samples
from src.modalities import SensorAugment, letterbox_float


def make_dataset(monkeypatch, split="train", epoch=0, probability=1):
    dataset = MultimodalDFineDataset.__new__(MultimodalDFineDataset)
    dataset.config = SimpleNamespace(
        small_crop=SmallObjectCrop(probability=probability), imgsz=128, polish_epoch=40,
        sensors=SensorAugment(), scale_min=0.9, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0,
    )
    dataset.split, dataset.epoch = split, epoch
    dataset.images = [Path("synthetic.png")]
    dataset.infrared = dataset.depth = Path(".")
    dataset.labels = [np.array([[7, 0.5, 0.5, 0.04, 0.04]], np.float32)]
    plane = (np.arange(80 * 160, dtype=np.float32).reshape(80, 160) % 200) + 1
    image = np.stack([plane + i for i in range(5)], axis=-1)
    monkeypatch.setattr("src.dfine.training.read_float_modalities", lambda *args: (image.copy(), True))
    monkeypatch.setattr("src.dfine.training.augment_sensors", lambda img, *args: img)
    return dataset, image


def test_dfine_raw_crop_pixels_boxes_and_batch_match_shared_geometry(monkeypatch):
    dataset, image = make_dataset(monkeypatch)
    original = dataset.labels[0].copy()
    random.seed(29)
    random.random()  # 与数据集的尝试概率抽样对齐。
    (left, top, w, h), keep, boxes = choose_small_object_crop(
        original[:, 1:], original[:, :1], image.shape[:2], 128, dataset.config.small_crop)
    expected_image, geometry = letterbox_float(image[top:top + h, left:left + w].copy(), 128)
    sx, sy, pad_x, pad_y = geometry
    expected_boxes = boxes * np.array([w * sx, h * sy, w * sx, h * sy], np.float32)
    expected_boxes[:, :2] += [pad_x, pad_y]
    random.seed(29)
    actual, target = dataset[0]
    np.testing.assert_allclose(actual.numpy(), expected_image.transpose(2, 0, 1) / 255, atol=1e-7)
    np.testing.assert_allclose(target["boxes"], expected_boxes / 128, atol=1e-7)
    np.testing.assert_array_equal(dataset.labels[0], original)
    assert target["small_crop_attempted"] and target["small_crop_applied"]
    assert target["orig_size"].tolist() == [w, h]
    assert actual.dtype == torch.float32 and actual.shape == (5, 128, 128)
    batch, targets = collate_samples([(actual, target), (actual, target)])
    assert batch.shape == (2, 5, 128, 128) and len(targets) == 2


@pytest.mark.parametrize("split,epoch", [("val", 0), ("train", 40)])
def test_dfine_validation_and_polish_remain_full_image(monkeypatch, split, epoch):
    dataset, image = make_dataset(monkeypatch, split=split, epoch=epoch)
    monkeypatch.setattr("src.dfine.training.choose_small_object_crop", lambda *args: pytest.fail("验证/收尾不得裁剪"))
    actual, target = dataset[0]
    expected, _ = letterbox_float(image, 128)
    np.testing.assert_allclose(actual.numpy(), expected.transpose(2, 0, 1) / 255, atol=1e-7)
    assert target["orig_size"].tolist() == [160, 80]
    assert not target["small_crop_attempted"] and not target["small_crop_applied"]


def test_dfine_crop_failure_falls_back_without_dropping_objects(monkeypatch):
    dataset, _ = make_dataset(monkeypatch)
    dataset.labels[0] = np.array([[7, 0.5, 0.5, 0.02, 0.02], [0, 0.5, 0.5, 1, 1]], np.float32)
    _, target = dataset[0]
    assert target["small_crop_attempted"] and not target["small_crop_applied"]
    assert len(target["boxes"]) == 2 and target["orig_size"].tolist() == [160, 80]
