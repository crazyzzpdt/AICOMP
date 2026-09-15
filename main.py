"""YOLO26m RGB-only 基线训练入口。

从官方 2000 组训练集的 visible 模态训练 12 类目标检测模型，
结果输出到 runs/detect/AI COMP/（含 best.pt / last.pt 与训练曲线）。
"""
# 内置库
import os

# 三方库
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE, "datasets", "data.yaml")


def train_baseline() -> None:
    """启动 YOLO26m 基线训练。

    Note:
        轮数给足、靠 patience 早停兜底；Ctrl+C 强退（退出码 0xC000013A 属正常）
        后加载 last.pt 并置 resume=True 即可无缝续训。
    """
    model = YOLO("yolo26m.pt")
    print("YOLO26m 基线训练启动：数据集 datasets/data.yaml，设备 GPU 0")
    result = model.train(
        data=DATA_YAML,
        epochs=10000,     # 轮数给足，配合 patience 早停兜底
        imgsz=940,        # 高分辨率输入，兼顾小目标定位
        patience=100,     # 验证指标 100 轮无改善即早停
        batch=32,
        save=True,        # 保存权重是断点续训的命根子，勿关
        save_period=100,  # 每 100 轮落一次检查点
        plots=True,       # 生成 results.png / PR 曲线，复盘必备
        val=True,
        workers=0,        # Windows 上保守取 0，避免 DataLoader 进程问题
        device=0,
        amp=True,
        name="AI COMP",
    )
    print("训练结束，产出目录 runs/detect/AI COMP/")
    print(result)


if __name__ == "__main__":
    train_baseline()
