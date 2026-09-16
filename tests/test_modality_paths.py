"""验证自包含 datasets 的模态路径不会串用训练和验证数据。"""

# 内置库
from pathlib import Path

# 三方库
import pytest

# 自己的模块
import 三模态训练 as multimodal


def test_split_paths_and_legacy_paths(tmp_path: Path) -> None:
    """各划分读取自己的模态目录，旧共享目录配置仍可验证历史模型。"""
    data = {"path": tmp_path, "infrared": {"train": "train/infrared", "val": "val/infrared"}}
    assert multimodal.resolve_modality_directory(data, "infrared", str(tmp_path / "val/images")) == tmp_path / "val/infrared"
    data["infrared"] = "../raw/infrared"
    assert multimodal.resolve_modality_directory(data, "infrared", str(tmp_path / "train/images")) == (tmp_path / "../raw/infrared").resolve()


def test_unknown_split_is_not_silently_mapped(tmp_path: Path) -> None:
    """未知拆分必须报错，不能误用验证模态。"""
    data = {"path": tmp_path, "infrared": {"train": "train/infrared", "val": "val/infrared"}}
    with pytest.raises(ValueError):
        multimodal.resolve_modality_directory(data, "infrared", str(tmp_path / "test/images"))
