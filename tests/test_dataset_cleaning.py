"""验证定向标签清洗不会修改原数据、误删目标或静默接受异常标注。"""

# 三方库
import pytest
from pathlib import Path
from PIL import Image

# 自己的模块
import 准备三模态数据集 as preparation


def test_clip_visible_box_and_remove_only_exact_duplicate() -> None:
    """边界裁剪应重算中心和宽度，相邻同类框必须保留。"""
    original = "0 0.98 0.5 0.1 0.4\n0 0.98 0.5 0.1 0.4\n0 0.97 0.5 0.1 0.4\n"
    cleaned, changes = preparation.clean_label_text(original)
    rows = [[float(value) for value in row.split()] for row in cleaned.splitlines()]
    assert len(rows) == 2
    assert rows[0] == pytest.approx([0, 0.965, 0.5, 0.07, 0.4])
    assert rows[1] == pytest.approx([0, 0.96, 0.5, 0.08, 0.4])
    assert [change["action"] for change in changes] == ["clip", "duplicate", "clip"]
    assert changes[1]["line"] == 2


def test_cleaning_preserves_roundoff_and_is_idempotent() -> None:
    """舍入量级边缘误差不被当作需人工清理的问题。"""
    original = "7 0.5 0.5 1.000001 0.4\n"
    assert preparation.clean_label_text(original) == (original, [])
    cleaned, _ = preparation.clean_label_text("0 0.98 0.5 0.1 0.4\n")
    assert preparation.clean_label_text(cleaned) == (cleaned, [])
    assert preparation.clean_label_text("") == ("", [])


@pytest.mark.parametrize("line", ["12 0.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1", "0 0.5 0.5 -0.1 0.1", "0 2 0.5 0.1 0.1", "0.5 0.5 0.5 0.1 0.1"])
def test_invalid_labels_fail_before_materialization(line: str) -> None:
    """非法类别、非有限值、负尺寸和完全落在图外的框不能被静默删除。"""
    with pytest.raises(ValueError):
        preparation.clean_label_text(line + "\n")


def test_reviewed_class_correction_requires_exact_source() -> None:
    """客车修订仅作用于审阅过的原行，源版本改变后不能误套。"""
    original = "0 0.623698 0.336574 0.752604 0.678704\n0 0.352083 0.218056 0.0208333 0.0935185\n"
    corrected, changes = preparation.correct_reviewed_classes("000003_026_00000001.txt", original)
    assert corrected.startswith("6 0.623698 0.336574 0.752604 0.678704\n")
    assert corrected.splitlines()[1] == original.splitlines()[1]
    assert changes[0]["action"] == "class_correction"
    with pytest.raises(ValueError):
        preparation.correct_reviewed_classes("000003_026_00000001.txt", "0 0.5 0.5 0.2 0.2\n")


def test_materialization_preserves_originals_and_detects_edits(tmp_path: Path) -> None:
    """落位保留原始硬链接标签内容，并拒绝审阅后的未记录改动。"""
    dataset, raw, labels = tmp_path / "datasets", tmp_path / "raw", tmp_path / "official"
    labels.mkdir()
    for modality in ("visible", "infrared", "depth"):
        (raw / modality).mkdir(parents=True)
    for split in ("train", "val"):
        for folder in ("images", "labels"):
            (dataset / split / folder).mkdir(parents=True)
        for modality in ("visible", "infrared", "depth"):
            Image.new("RGB", (32, 24)).save(raw / modality / f"{split}.png")
        (dataset / split / "images" / f"{split}.png").hardlink_to(raw / "visible" / f"{split}.png")
        (labels / f"{split}.txt").write_text("0 0.98 0.5 0.1 0.4\n", encoding="utf-8")
        (dataset / split / "labels" / f"{split}.txt").hardlink_to(labels / f"{split}.txt")
    (dataset / "data.yaml").write_text("channels: 5\n", encoding="utf-8")
    records = preparation.inspect_cleaning(dataset, raw, labels)
    report = tmp_path / "report"
    report.mkdir()
    preparation.apply_cleaning(dataset, raw, labels, report, records)
    assert (labels / "train.txt").read_text() == "0 0.98 0.5 0.1 0.4\n"
    assert not (dataset / "train/labels/train.txt").samefile(labels / "train.txt")
    assert (report / "before/train/labels/train.txt").read_text() == "0 0.98 0.5 0.1 0.4\n"
    assert (dataset / "val/depth/val.png").samefile(raw / "depth/val.png")
    rebuilt = preparation.inspect_cleaning(dataset, raw, labels)
    assert rebuilt[0]["cleaned"] == (dataset / "train/labels/train.txt").read_text()
    (dataset / "val/labels/val.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    next_report = tmp_path / "next_report"
    next_report.mkdir()
    with pytest.raises(ValueError):
        preparation.apply_cleaning(dataset, raw, labels, next_report, rebuilt)
    assert not (next_report / "before").exists()
