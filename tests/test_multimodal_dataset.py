"""验证三模态数据集向 Ultralytics 输出五通道训练样本。"""

# 内置库
from pathlib import Path

# 三方库
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.utils import YAML

# 自己的模块
from 三模态训练 import MultimodalYOLODataset


ROOT: Path = Path(__file__).resolve().parents[1]


def test_multimodal_dataset_emits_five_channel_tensor() -> None:
    """真实训练样本须按新版标签和三模态路径生成五通道张量。"""
    data = YAML.load(ROOT / "datasets" / "data.yaml")
    data["path"] = ROOT / "datasets"
    dataset = MultimodalYOLODataset(
        img_path=str(ROOT / "datasets" / "train" / "images"),
        imgsz=64,
        batch_size=1,
        augment=False,
        hyp=DEFAULT_CFG,
        rect=False,
        cache=False,
        data=data,
        prefix="test: ",
    )

    sample = dataset[0]

    assert sample["img"].shape[0] == 5
    assert sample["img"].shape[1:] == (64, 64)
    assert sample["cls"].shape[0] == sample["bboxes"].shape[0]
