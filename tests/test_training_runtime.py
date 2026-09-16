"""验证长训练上限下的收敛计划与加载器内存限制。"""

# 内置库
import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace

# 三方库
import pytest
import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.utils import YAML

# 自己的模块
from 三模态训练 import MultimodalDetectionTrainer, MultimodalYOLODataset, learning_rate_factor


def test_scheduler_reaches_floor_before_large_epoch_limit() -> None:
    """1000 轮仅作为上限，学习率需在 200 轮降到底且后续不反弹。"""
    trainer = object.__new__(MultimodalDetectionTrainer)
    trainer.args = SimpleNamespace(cos_lr=True, lrf=0.01)
    trainer.epochs = 1000
    trainer.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.0003)
    trainer._setup_scheduler()
    assert trainer.lf(0) == pytest.approx(1.0)
    assert trainer.lf(100) == pytest.approx(0.505)
    assert trainer.lf(200) == pytest.approx(0.01)
    assert trainer.lf(1000) == pytest.approx(0.01)


@pytest.mark.parametrize("cosine", [False, True])
def test_schedule_clamps_at_horizon_and_supports_short_runs(cosine: bool) -> None:
    """短训和续训越过衰减周期时不出现负学习率或周期反弹。"""
    assert learning_rate_factor(-1, 2, 0.1, cosine) == pytest.approx(1.0)
    assert learning_rate_factor(1, 2, 0.1, cosine) == pytest.approx(0.55)
    assert learning_rate_factor(2, 2, 0.1, cosine) == pytest.approx(0.1)
    assert learning_rate_factor(4, 2, 0.1, cosine) == pytest.approx(0.1)


def test_training_image_buffer_evicts_old_five_channel_images() -> None:
    """连续读取训练图后及时释放旧图，避免每进程保留原来的 32 张缓冲。"""
    root = Path(__file__).resolve().parents[1]
    data = YAML.load(root / "datasets/data.yaml")
    data["path"] = root / "datasets"
    hyp = copy(DEFAULT_CFG)
    hyp.hsv_h = hyp.hsv_s = hyp.hsv_v = 0.0
    dataset = MultimodalYOLODataset(
        img_path=str(root / "datasets/train/images"), imgsz=64, batch_size=4,
        augment=True, hyp=hyp, rect=False, cache=False, data=data,
    )
    for index in range(20):
        image, _, _ = dataset.load_image(index)
        assert image.shape[2] == 5
    assert sum(image is not None for image in dataset.ims) <= 8
    assert dataset.ims[0] is None


@pytest.mark.parametrize("workers", [0, 1])
def test_loader_bounds_prefetch_and_keeps_validation_worker_count(workers: int) -> None:
    """完整五通道验证加载不翻倍进程，预取至多一批且不锁页。"""
    root = Path(__file__).resolve().parents[1]
    trainer = object.__new__(MultimodalDetectionTrainer)
    trainer.args = copy(DEFAULT_CFG)
    trainer.args.workers = workers
    trainer.args.imgsz = 64
    trainer.args.cache = False
    trainer.data = YAML.load(root / "datasets/data.yaml")
    trainer.data["path"] = root / "datasets"
    trainer.model = SimpleNamespace(stride=torch.tensor([32]))
    trainer.device = torch.device("cpu")
    loader = trainer.get_dataloader(str(root / "datasets/val/images"), batch_size=4, rank=-1, mode="val")
    try:
        assert loader.num_workers == workers
        assert loader.prefetch_factor == (1 if workers else None)
        assert loader.pin_memory is False
        batch = next(iter(loader))
        assert batch["img"].shape[:2] == (4, 5)
    finally:
        loader.close()


@pytest.mark.parametrize("epochs", [200, 1000, 5000])
def test_main_mosaic_closes_after_same_number_of_augmented_epochs(epochs: int) -> None:
    """训练上限变化后，入口传给框架的关闭时间仍为第 101 轮。"""
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and any(keyword.arg == "close_mosaic" for keyword in node.keywords))
    expression = next(keyword.value for keyword in call.keywords if keyword.arg == "close_mosaic")
    namespace = {"MAX_EPOCHS": epochs, "MOSAIC_EPOCHS": 100}
    close_mosaic = eval(compile(ast.Expression(expression), "main.py", "eval"), namespace)
    assert epochs - close_mosaic == 100
