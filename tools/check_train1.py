"""有界检查真实train1入口：两轮少量批次，覆盖验证、保存、下一轮前向。

运行：uv run python -m tools.check_train1
只限制消费批次和运行预算，保持入口的分辨率、精度、增强及优化器参数。
结果写入独立runs/train1_checks目录，不作为模型质量评估或正式训练版本。
"""

import json
import os
import runpy
from dataclasses import replace
from datetime import datetime
from functools import partial
from itertools import islice
from pathlib import Path
from unittest.mock import patch

# 与正式入口一致，必须在导入框架前禁用自动联网行为。
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from ultralytics import YOLO

from src.yolo.aic.training import FusionDetectionTrainer


class LimitedLoader:
    """保留原始轮次长度和学习率日程，只截取少量真实批次。"""

    def __init__(self, loader, limit):
        self.loader, self.limit = loader, limit

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        yield from islice(iter(self.loader), self.limit)


class CheckTrainer(FusionDetectionTrainer):
    """复用正式训练器，额外记录输入、优化步及跨轮显存。"""

    def get_dataloader(self, *args, **kwargs):
        loader = super().get_dataloader(*args, **kwargs)
        mode = kwargs.get("mode", "train")
        return LimitedLoader(loader, 4 if mode == "train" else 2)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        image = batch["img"]
        assert image.dtype == torch.float32 and image.shape[1:] == (5, 1280, 1280)
        assert torch.isfinite(image).all() and 0 <= image.min() <= image.max() <= 1
        assert image[:, 3].count_nonzero() > 0, "红外输入缺失"
        self.check_batches = getattr(self, "check_batches", 0) + 1
        return batch

    def optimizer_step(self):
        super().optimizer_step()
        self.check_steps = getattr(self, "check_steps", 0) + 1

    def save_model(self):
        result = super().save_model()
        print(f"CHECK epoch={self.epoch + 1} saved; GPU allocated={torch.cuda.memory_allocated()/2**30:.3f} GiB, "
              f"peak={torch.cuda.max_memory_allocated()/2**30:.3f} GiB", flush=True)
        return result

    def train(self):
        try:
            super().train()
        finally:
            for name in ("train_loader", "test_loader"):
                loader = getattr(self, name, None)
                if isinstance(loader, LimitedLoader):
                    loader.close()


def main():
    root = Path(__file__).resolve().parents[1]
    original_train = YOLO.train
    record = {}

    def checked_train(self, *args, **kwargs):
        factory = kwargs["trainer"]
        recipe = replace(factory.keywords["recipe"], budget_epochs=2)
        kwargs.update(trainer=partial(CheckTrainer, recipe=recipe),
                      project=str(root / "runs" / "train1_checks"),
                      name=datetime.now().strftime("entry_%Y%m%d_%H%M%S"))
        original_train(self, *args, **kwargs)
        trainer = self.trainer
        assert trainer.epoch == 1 and trainer.check_batches == 8
        assert trainer.check_steps > 0
        checkpoint = torch.load(trainer.last, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 1 and checkpoint["optimizer"]["state"]
        assert all(p.dtype == torch.float32 for p in checkpoint["ema"].parameters())
        assert next(checkpoint["ema"].parameters()).shape[1] == 5
        stem = next(checkpoint["ema"].parameters())
        assert stem[:, 3].count_nonzero() > 0 and stem[:, 4].count_nonzero() > 0, "辅助模态首层未学习"
        assert all(p.device.type == "cpu" for p in checkpoint["ema"].parameters())
        record.update(status="passed", epochs=2, train_batches=trainer.check_batches,
                      optimizer_steps=trainer.check_steps, batch=trainer.args.batch,
                      imgsz=trainer.args.imgsz, amp=trainer.args.amp,
                      peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                      limitation="少量真实批次；不是完整两轮、长期稳定性或模型质量验证")
        (trainer.save_dir / "check_report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    with patch.object(YOLO, "train", checked_train):
        runpy.run_path(str(root / "train1.py"), run_name="__main__")


if __name__ == "__main__":
    main()
