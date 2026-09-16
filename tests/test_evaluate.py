"""验证五通道评估读取真实标签内容、计算官方 AP，并保护已有结果。"""

# 内置库
import os
import json
from pathlib import Path

# 三方库
import cv2
import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

# 自己的模块
from 准备三模态数据集 import CLASS_NAMES


def make_validation_data(root: Path) -> Path:
    """创建两张三模态验证图和一张训练图，只用于离线测试。"""
    for folder in ("train/images", "val/images", "val/labels", "infrared", "depth"):
        (root / folder).mkdir(parents=True)
    (root / "train/images/train.png").touch()
    for name in ("a.png", "b.png"):
        for folder, value in (("val/images", 40), ("infrared", 80), ("depth", 10000)):
            image = np.full((48, 64, 3) if folder == "val/images" else (48, 64), value, dtype=np.uint16 if folder == "depth" else np.uint8)
            cv2.imencode(".png", image)[1].tofile(root / folder / name)
        (root / "val/labels" / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    data = root / "data.yaml"
    data.write_text(f"train: train/images\nval: val/images\ninfrared: infrared\ndepth: depth\nchannels: 5\nnames:\n{names}\n", encoding="utf-8")
    return data


def test_validation_fingerprints_detect_same_size_same_mtime_label_change(tmp_path: Path) -> None:
    """清洗前后等长标签不能因框架缓存只比较大小而被误认为相同。"""
    from 评估模型 import inspect_validation

    data = make_validation_data(tmp_path)
    before = inspect_validation(data, expected_count=2)
    label = tmp_path / "val/labels/a.txt"
    old_stat = label.stat()
    label.write_text("7 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    os.utime(label, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    after = inspect_validation(data, expected_count=2)
    assert before.fingerprints["validation_labels_sha256"] != after.fingerprints["validation_labels_sha256"]
    assert before.fingerprints["image_split_sha256"] == after.fingerprints["image_split_sha256"]
    assert after.samples[0].labels[0, 0] == 7
    assert not list(tmp_path.rglob("*.cache"))
    assert not list(tmp_path.rglob("*.npy"))


def make_result(boxes: list[list[float]]) -> Results:
    """使用原图坐标构造检测结果，避免网络随机性掩盖 AP 匹配问题。"""
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    return Results(image, path="a.png", names=dict(enumerate(CLASS_NAMES)), boxes=torch.tensor(boxes).reshape(-1, 6))


def test_metrics_match_original_coordinates_and_include_absent_classes(tmp_path: Path) -> None:
    """正确框获得高 AP，空图中的高分误检会扣分，无真值类不能伪造 AP。"""
    from 评估模型 import score_predictions

    labels = np.array([[0, 0.4, 0.4, 0.4, 0.4]], dtype=np.float32)
    correct = make_result([[20, 10, 60, 30, 0.9, 0]])
    clean = score_predictions(iter([("a.png", correct, labels)]), tmp_path / "clean")
    polluted = score_predictions(iter([
        ("a.png", correct, labels),
        ("empty.png", make_result([[20, 10, 60, 30, 0.99, 0]]), np.empty((0, 5), dtype=np.float32)),
    ]), tmp_path / "polluted")
    assert clean["metrics"]["metrics/mAP50-95(B)"] > 0.99
    assert polluted["metrics"]["metrics/mAP50-95(B)"] < clean["metrics"]["metrics/mAP50-95(B)"]
    assert len(clean["per_class"]) == 12
    assert clean["per_class"][0]["instances"] == 1
    assert clean["per_class"][1]["instances"] == 0
    assert clean["per_class"][1]["ap50_95"] is None
    missed = score_predictions(iter([("a.png", make_result([]), labels)]), tmp_path / "missed")
    assert missed["metrics"]["metrics/mAP50-95(B)"] == 0
    assert missed["per_class"][0]["ap50_95"] == 0


def test_evaluate_checkpoint_runs_five_channel_model_without_caches_or_overwrite(tmp_path: Path) -> None:
    """小型真实五通道网络走通整套评估，不改检查点、源数据或已有结果。"""
    from ultralytics.nn.tasks import DetectionModel
    from 评估模型 import evaluate_checkpoint, file_sha256

    data = make_validation_data(tmp_path / "dataset")
    config = data.read_text(encoding="utf-8")
    data.write_text(config.replace("infrared: infrared", "infrared: {train: infrared, val: infrared}").replace("depth: depth", "depth: {train: depth, val: depth}"), encoding="utf-8")
    cache = data.parent / "val/labels.cache"
    cache.write_bytes(b"invalid stale cache")
    np.save(data.parent / "val/images/a.npy", np.ones((48, 64, 3), dtype=np.uint8))
    cfg = {
        "nc": 12,
        "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [32, 3, 2]]],
        "head": [[[0, 1, 2], 1, "Detect", [12]]],
    }
    model = DetectionModel(cfg, ch=5, nc=12, verbose=False)
    model.names = dict(enumerate(CLASS_NAMES))
    weights = tmp_path / "tiny.pt"
    torch.save({"model": model, "train_args": {"task": "detect"}}, weights)
    weight_hash = file_sha256(weights)
    output = tmp_path / "evaluation"
    report = evaluate_checkpoint(weights, 64, output, "cpu", data=data, expected_count=2)
    assert report["status"] == "complete"
    assert report["images"] == 2 and report["instances"] == 2
    assert len(report["per_class"]) == 12
    assert report["metadata"]["weights_sha256"] == weight_hash == file_sha256(weights)
    assert report["metadata"]["parameters"]["quantize"] == 32
    assert report["metadata"]["parameters"]["nms"] is True
    assert json.loads((output / "evaluation.json").read_text(encoding="utf-8"))["metadata"] == report["metadata"]
    assert (output / "per_class.csv").is_file() and (output / "summary.csv").is_file()
    assert cache.read_bytes() == b"invalid stale cache"
    assert np.load(data.parent / "val/images/a.npy").shape[-1] == 3
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(weights, 64, output, "cpu", data=data, expected_count=2)
