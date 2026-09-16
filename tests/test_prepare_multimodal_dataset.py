"""验证新版标注三模态训练集的非破坏性构建。"""

# 内置库
from pathlib import Path

# 自己的模块
from 准备三模态数据集 import prepare_multimodal_dataset


def test_prepare_multimodal_dataset_uses_new_labels_and_ignores_old_cache(tmp_path: Path) -> None:
    """构建器只链接图像、复制新版标签，且不会改变现有旧数据集。"""
    old_dataset = tmp_path / "datasets"
    raw_dataset = tmp_path / "raw"
    new_labels = tmp_path / "new_labels"
    for split in ("train", "val"):
        (old_dataset / split / "images").mkdir(parents=True)
        (old_dataset / split / "labels").mkdir(parents=True)
    for modality in ("infrared", "depth"):
        (raw_dataset / modality).mkdir(parents=True)
    new_labels.mkdir()

    for split, stem in (("train", "sample_train"), ("val", "sample_val")):
        (old_dataset / split / "images" / f"{stem}.png").write_bytes(b"visible")
        (old_dataset / split / "images" / f"{stem}.npy").write_bytes(b"stale cache")
        (old_dataset / split / "labels" / f"{stem}.txt").write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")
        (raw_dataset / "infrared" / f"{stem}.png").write_bytes(b"infrared")
        (raw_dataset / "depth" / f"{stem}.png").write_bytes(b"depth")
        (new_labels / f"{stem}.txt").write_text("1 0.2 0.2 0.2 0.2\n", encoding="utf-8")

    output = prepare_multimodal_dataset(old_dataset, raw_dataset, new_labels, old_dataset)

    assert (output / "train" / "images" / "sample_train.png").is_file()
    assert not (output / "train" / "images" / "sample_train.npy").exists()
    assert (output / "val" / "labels" / "sample_val.txt").read_text(encoding="utf-8").startswith("1 ")
    assert (old_dataset / "val" / "labels" / "sample_val.txt").read_text(encoding="utf-8").startswith("1 ")
    assert "channels: 5" in (output / "data.yaml").read_text(encoding="utf-8")
