"""按 main.py 参数执行短时 GPU 训练，检查正式尺寸下的运行能力。"""

# 内置库
import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 三方库
import torch
from ultralytics import YOLO

# 自己的模块
import main
from 三模态训练 import MultimodalDetectionTrainer


def main_check() -> None:
    """复用正式超参数，缩减训练样本并保留完整验证集。"""
    tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "train" and len(node.keywords) > 20)
    args = {item.arg: eval(compile(ast.Expression(item.value), "main.py", "eval"), vars(main)) for item in call.keywords}
    args.update(epochs=1, fraction=0.005, project=str(Path(main.__file__).parent / "outputs" / "training_checks"), name="rgb_order_fixed", save_period=-1)
    torch.cuda.reset_peak_memory_stats()
    YOLO(main.MODEL_PATH).train(**args)
    print(f"PEAK_ALLOCATED_GIB={torch.cuda.max_memory_allocated() / 2**30:.2f}")
    print(f"PEAK_RESERVED_GIB={torch.cuda.max_memory_reserved() / 2**30:.2f}")


if __name__ == "__main__":
    main_check()
