"""验证三模态预测的通道、坐标、提交格式与防覆盖约束。"""

# 内置库
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

# 三方库
import cv2
import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

# 自己的模块
from 准备三模态数据集 import CLASS_NAMES


def make_result(boxes: list[list[float]]) -> Results:
    """构造原图为 100×50 的检测结果，坐标期望由测试手工给定。"""
    image = np.full((50, 100, 3), (10, 20, 30), dtype=np.uint8)
    return Results(image, path="样本.png", names=dict(enumerate(CLASS_NAMES)), boxes=torch.tensor(boxes).reshape(-1, 6))


def test_prediction_module_exists() -> None:
    """交付的入口必须使用用户指定的英文文件名。"""
    assert (Path(__file__).resolve().parents[1] / "predict.py").is_file()


def test_collect_samples_rejects_missing_modality_and_stem_collision(tmp_path: Path) -> None:
    """避免缺失模态或两个图片共享同一个提交 TXT。"""
    from predict import collect_samples

    for modality in ("visible", "infrared", "depth"):
        (tmp_path / modality).mkdir()
        (tmp_path / modality / "样本.png").touch()
    samples = collect_samples(tmp_path)
    assert len(samples) == 1
    assert tuple(p.parent.name for p in samples[0]) == ("visible", "infrared", "depth")
    (tmp_path / "depth" / "样本.png").unlink()
    with pytest.raises(ValueError, match="配对"):
        collect_samples(tmp_path)
    (tmp_path / "depth" / "样本.png").touch()
    for modality in ("visible", "infrared", "depth"):
        (tmp_path / modality / "样本.jpg").touch()
    with pytest.raises(ValueError, match="同名"):
        collect_samples(tmp_path)


def test_prepare_output_never_overwrites(tmp_path: Path) -> None:
    """第二次运行必须拒绝已有目录，并保留第一次的产物。"""
    from predict import prepare_output

    output = tmp_path / "predict"
    prepare_output(output)
    assert {p.name for p in output.iterdir()} == {"images", "labels", "比赛提交内容"}
    sentinel = output / "labels" / "existing.txt"
    sentinel.write_text("保留", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_output(output)
    assert sentinel.read_text(encoding="utf-8") == "保留"


def test_rows_normalize_sort_limit_and_drop_degenerate_boxes() -> None:
    """错用 LetterBox 尺寸归一化、未排序或超限都应被发现。"""
    from predict import prediction_rows

    result = make_result([[20, 10, 60, 30, 0.2, 7], [0, 0, 100, 50, 0.9, 0], [10, 10, 10, 20, 0.8, 2]])
    rows = prediction_rows(result)
    np.testing.assert_allclose(rows, [[0, 0.5, 0.5, 1, 1, 0.9], [7, 0.4, 0.4, 0.4, 0.4, 0.2]], atol=1e-7)
    assert len(prediction_rows(make_result([[0, 0, 10, 10, 0.5, 0]] * 110))) == 100


@pytest.mark.parametrize("box", [[0, 0, 10, 10, float("nan"), 0], [0, 0, 10, 10, 0.5, 12], [0, 0, 10, 10, 0.5, 1.5]])
def test_rows_reject_invalid_predictions(box: list[float]) -> None:
    """非法置信度或类别不能默默写进提交包。"""
    from predict import prediction_rows

    with pytest.raises(ValueError):
        prediction_rows(make_result([box]))


def test_submission_includes_empty_label_and_only_root_txt(tmp_path: Path) -> None:
    """低分候选保留在标签内，空图同样有 TXT，压缩包不夹带图片。"""
    from predict import build_submission, prepare_output, save_prediction

    output = tmp_path / "predict"
    prepare_output(output)
    save_prediction(make_result([[20, 10, 60, 30, 0.01, 7]]), "样本.png", output, 0.25)
    save_prediction(make_result([]), "空图.png", output, 0.25)
    image = cv2.imdecode(np.fromfile(output / "images" / "样本.png", dtype=np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(image, make_result([]).orig_img)
    assert len((output / "labels" / "样本.txt").read_text().split()) == 6
    archive = build_submission(output, ["样本.png", "空图.png"])
    with ZipFile(archive) as saved:
        assert set(saved.namelist()) == {"样本.txt", "空图.txt"}
        assert saved.read("空图.txt") == b""
    with pytest.raises(FileExistsError):
        save_prediction(make_result([]), "样本.png", output, 0.25)
    with pytest.raises(FileExistsError):
        build_submission(output, ["样本.png", "空图.png"])


@pytest.mark.parametrize("content", ["0 0.5 0.5 0.5 0.5", "12 0.5 0.5 0.5 0.5 0.7", "0 0.99 0.5 0.5 0.5 0.7", "0 nan 0.5 0.5 0.5 0.7"])
def test_submission_revalidates_txt(tmp_path: Path, content: str) -> None:
    """手动损坏标签后不能仍生成正式 ZIP。"""
    from predict import build_submission, prepare_output, save_prediction

    output = tmp_path / "predict"
    prepare_output(output)
    save_prediction(make_result([]), "样本.png", output, 0.25)
    (output / "labels" / "样本.txt").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        build_submission(output, ["样本.png"])
    assert not (output / "比赛提交内容" / "submission.zip").exists()


def test_predictor_preserves_channels_and_restores_original_coordinates() -> None:
    """五通道禁止 RGB/BGR 整体翻转，去填充后才能按原图归一化。"""
    from predict import MultimodalDetectionPredictor

    predictor = MultimodalDetectionPredictor(overrides={"imgsz": 128, "rect": False, "save": False})
    predictor.device = torch.device("cpu")
    predictor.imgsz = [128, 128]
    predictor.model = SimpleNamespace(fp16=False, format="pt", stride=32, names=dict(enumerate(CLASS_NAMES)))
    fused = np.full((50, 100, 5), (10, 20, 30, 40, 50), dtype=np.uint8)
    tensor = predictor.preprocess([fused])
    np.testing.assert_allclose(tensor[0, :, 64, 64], np.array([10, 20, 30, 40, 50]) / 255, atol=1e-7)
    pred = torch.tensor([[25.6, 44.8, 76.8, 70.4, 0.8, 7]])
    result = predictor.construct_result(pred, tensor, fused, "样本.png")
    np.testing.assert_allclose(result.boxes.xyxy, [[20, 10, 60, 30]], atol=1e-5)
    assert result.orig_img[0, 0].tolist() == [30, 20, 10]
    with pytest.raises(ValueError):
        predictor.preprocess([fused[:, :, :3]])


def test_validate_model_rejects_rgb_weights_and_wrong_class_order() -> None:
    """不能误用 COCO 权重或类别编号不一致的检查点。"""
    from predict import validate_model

    model = SimpleNamespace(task="detect", names=dict(enumerate(CLASS_NAMES)), model=torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1)))
    with pytest.raises(ValueError, match="五通道"):
        validate_model(model)
    model.model = torch.nn.Sequential(torch.nn.Conv2d(5, 4, 1))
    model.model.yaml = {"channels": 5}
    validate_model(model)
    model.names[7] = "sports ball"
    with pytest.raises(ValueError, match="类别"):
        validate_model(model)


def test_cli_runs_real_five_channel_model_offline(tmp_path: Path) -> None:
    """用小型随机五通道网络走通真实推理、图片保存与 ZIP，不依赖正式训练。"""
    from ultralytics.nn.tasks import DetectionModel
    from predict import main

    cfg = {
        "nc": 12,
        "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [32, 3, 2]]],
        "head": [[[0, 1, 2], 1, "Detect", [12]]],
    }
    model = DetectionModel(cfg, ch=5, nc=12, verbose=False)
    model.names = dict(enumerate(CLASS_NAMES))
    weights = tmp_path / "tiny.pt"
    torch.save({"model": model, "train_args": {"task": "detect"}}, weights)
    source = tmp_path / "测试集"
    for modality, value in (("visible", 40), ("infrared", 80), ("depth", 10000)):
        folder = source / modality
        folder.mkdir(parents=True)
        image = np.full((48, 64, 3) if modality == "visible" else (48, 64), value, dtype=np.uint16 if modality == "depth" else np.uint8)
        cv2.imencode(".png", image)[1].tofile(folder / "样本.png")
    output = tmp_path / "predict"
    main(["--weights", str(weights), "--source", str(source), "--output", str(output), "--device", "cpu", "--imgsz", "64", "--expected-count", "1", "--conf", "0.99"])
    with ZipFile(output / "比赛提交内容" / "submission.zip") as saved:
        assert saved.namelist() == ["样本.txt"]
    assert (output / "images" / "样本.png").is_file()
    assert (output / "prediction.json").is_file()
