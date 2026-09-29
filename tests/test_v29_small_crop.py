"""v29有界几何检查：只处理CPU合成图，不训练模型或读取复赛测试图。"""

from copy import deepcopy
from dataclasses import replace
import random

import numpy as np
import pytest
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset

from src.modalities import SensorAugment
from src.yolo.aic.training import FloatYOLODataset
from src.augmentation import SmallObjectCrop, choose_small_object_crop


def test_crop_preserves_target_and_keeps_source_labels():
    random.seed(29)
    boxes = np.array([[0.5, 0.5, 0.02, 0.02], [0.53, 0.48, 0.01, 0.01]], np.float32)
    original = boxes.copy()
    window, keep, changed = choose_small_object_crop(boxes, np.array([[1], [7]]), (1080, 1920), 1536, SmallObjectCrop())
    left, top, w, h = window
    assert keep.all() and 0 <= left <= 1920 - w and 0 <= top <= 1080 - h
    restored = changed * np.array([w, h, w, h])
    restored[:, :2] += [left, top]
    np.testing.assert_allclose(restored, boxes * [1920, 1080, 1920, 1080], atol=1e-4)
    np.testing.assert_array_equal(boxes, original)
    assert np.all(changed[:, 2:] > original[:, 2:])


def test_crop_rejects_severely_cut_neighbour_instead_of_dropping_its_label():
    # 全图大框与任意0.6倍窗口相交后只剩36%，必须整体回退。
    boxes = np.array([[0.5, 0.5, 0.01, 0.01], [0.5, 0.5, 1, 1]], np.float32)
    config = SmallObjectCrop(min_fraction=0.6, max_fraction=0.6)
    assert choose_small_object_crop(boxes, np.array([[0], [2]]), (1080, 1920), 1536, config) is None
    assert choose_small_object_crop(boxes[1:], np.array([[2]]), (1080, 1920), 1536, config) is None
    assert choose_small_object_crop(np.empty((0, 4)), np.empty((0, 1)), (1080, 1920), 1536, config) is None


@pytest.mark.parametrize("settings", [{"probability": 2}, {"min_fraction": 0}, {"attempts": 0},
                                     {"max_fraction": 1}, {"max_object_size": float("nan")}])
def test_invalid_crop_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        SmallObjectCrop(**settings)


def make_dataset():
    dataset = FloatYOLODataset.__new__(FloatYOLODataset)
    dataset.small_crop = SmallObjectCrop(probability=1)
    dataset.crop_enabled = dataset.augment = True
    dataset.imgsz = 128
    dataset.rect = dataset.use_segments = dataset.use_keypoints = dataset.use_obb = False
    dataset.cache = False
    dataset.data = {"channels": 5}
    dataset.format_class = Format
    dataset.metric_depth = {0: True}
    dataset.sensors = SensorAugment()
    dataset.labels = [{"im_file": "synthetic.png", "shape": (80, 160), "normalized": True,
                       "bbox_format": "xywh", "bboxes": np.array([[0.5, 0.5, 0.04, 0.04]], np.float32),
                       "cls": np.array([[7]], np.float32), "segments": []}]
    dataset.hyp = get_cfg(overrides={"mosaic": 0.25, "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,
                                    "scale": 0.2, "translate": 0.1, "fliplr": 0.5})
    dataset.transforms = dataset.build_transforms(deepcopy(dataset.hyp))
    return dataset


def test_raw_crop_keeps_five_planes_aligned_and_uses_separate_transforms(monkeypatch):
    random.seed(29)
    dataset = make_dataset()
    plane = np.arange(80 * 160, dtype=np.float32).reshape(80, 160) % 200
    image = np.stack([plane + i for i in range(5)], axis=-1)
    dataset.load_fused_image = lambda _: image.copy()
    monkeypatch.setattr("src.yolo.aic.training.augment_sensors", lambda img, *args: img)
    original = deepcopy(dataset.labels)
    cropped = dataset.get_small_crop(0)
    assert cropped["ori_shape"] != image.shape[:2]
    assert cropped["img"].dtype == np.float32
    np.testing.assert_allclose(cropped["img"][:, :, 1] - cropped["img"][:, :, 0], 1, atol=2e-5)
    np.testing.assert_array_equal(dataset.labels[0]["bboxes"], original[0]["bboxes"])
    # 若实现误走普通增强链，直接失败；正常路径仅LetterBox、水平翻转、Format。
    dataset.transforms = lambda _: pytest.fail("裁剪分支不应叠加Mosaic或RandomPerspective")
    output = dataset[0]
    assert output["img"].shape == (5, 128, 128)
    assert output["small_crop_attempted"] and output["small_crop_applied"]
    assert output["bboxes"].min() >= 0 and output["bboxes"].max() <= 1
    batch = YOLODataset.collate_fn([output, output])
    assert batch["img"].shape == (2, 5, 128, 128)
    assert batch["small_crop_applied"] == (True, True)


def test_validation_and_polish_disable_crop(monkeypatch):
    dataset = make_dataset()
    dataset.get_small_crop = lambda _: pytest.fail("验证或收尾阶段不能裁剪")
    monkeypatch.setattr("src.yolo.aic.training.MultimodalYOLODataset.__getitem__", lambda *args: {})
    dataset.augment = False
    assert not dataset[0]["small_crop_attempted"]
    dataset.augment = True
    dataset.polish_scale, dataset.polish_translate = 0.1, 0.025
    dataset.close_mosaic(dataset.hyp)
    assert not dataset[0]["small_crop_applied"]
    assert dataset.hyp.mosaic == 0 and dataset.hyp.scale == 0.1


def test_disabled_recipe_does_not_consume_extra_random_draws(monkeypatch):
    dataset = make_dataset()
    dataset.small_crop = replace(dataset.small_crop, probability=0)
    monkeypatch.setattr("src.yolo.aic.training.MultimodalYOLODataset.__getitem__", lambda *args: {})
    random.seed(29)
    state = random.getstate()
    dataset[0]
    assert random.getstate() == state
