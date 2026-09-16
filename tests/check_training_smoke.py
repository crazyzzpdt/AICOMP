"""按 main.py 参数执行短时 GPU 训练，检查正式尺寸下的运行能力。"""

# 内置库
import ast
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 必须在导入 Ultralytics 前导入 main，复用离线环境设置。
import main

# 三方库
import torch
import psutil
from ultralytics import YOLO

# 自己的模块
from 三模态训练 import MultimodalDetectionTrainer


def main_check() -> None:
    """复用正式超参数，覆盖两轮训练、完整验证和 Mosaic 关闭。"""
    parser = argparse.ArgumentParser(description="使用真实三模态数据检查训练资源上限")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--name", default="training_v2_smoke")
    options = parser.parse_args()
    tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "train" and len(node.keywords) > 20)
    args = {item.arg: eval(compile(ast.Expression(item.value), "main.py", "eval"), vars(main)) for item in call.keywords}
    args.update(
        epochs=options.epochs,
        fraction=options.fraction,
        close_mosaic=1 if options.epochs > 1 else 0,
        project=str(Path(main.__file__).parent / "runs" / "config_checks"),
        name=options.name,
        save_period=-1,
        plots=False,
    )
    for key in ("batch", "imgsz", "workers"):
        if (value := getattr(options, key)) is not None:
            args[key] = value
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(main.MODEL_PATH)
    model.add_callback("on_train_epoch_end", report_memory)
    model.add_callback("on_train_batch_end", report_resource_sample)
    model.train(**args)
    if model.trainer.batch_size != args["batch"]:
        raise RuntimeError(f"探测触发了自动降批次：{args['batch']} -> {model.trainer.batch_size}，该配置不能作为通过")
    print(f"PEAK_ALLOCATED_GIB={torch.cuda.max_memory_allocated() / 2**30:.2f}")
    print(f"PEAK_RESERVED_GIB={torch.cuda.max_memory_reserved() / 2**30:.2f}")


def report_resource_sample(trainer: MultimodalDetectionTrainer) -> None:
    """每 64 批采样整卡占用和系统剩余内存，观察是否使用了过多资源。"""
    step: int = getattr(trainer, "resource_sample_step", 0)
    trainer.resource_sample_step = step + 1
    if step % 64:
        return
    free, total = torch.cuda.mem_get_info()
    print(f"RESOURCE batch={step} GPU_USED_GIB={(total-free)/2**30:.2f} RAM_AVAILABLE_GIB={psutil.virtual_memory().available/2**30:.2f}")


def report_memory(trainer: MultimodalDetectionTrainer) -> None:
    """记录训练进程树在每轮结束时的内存快照。"""
    process = psutil.Process()
    processes = [process, *process.children(recursive=True)]
    rss: int = 0
    for child in processes:
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    print(f"第 {trainer.epoch + 1} 轮进程树 RSS 快照：{rss / 2**30:.2f} GiB（含共享页重复计数，非峰值）")


if __name__ == "__main__":
    main_check()
