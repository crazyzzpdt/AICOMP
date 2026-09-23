"""D-FINE 独立预测入口，使用复赛 ``datasets/test`` 三模态数据。

D-FINE 采用训练一致的正方形预处理和原生查询排序，不调用 YOLO NMS。
结果只写入 ``历史产出``；需要补充团队材料时使用 ``--materials-only``。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_WEIGHTS: Path = PROJECT_ROOT / "runs/detect/AIC_RGBIRDepth_dfine_l_1280_v6/weights/best.pth"
DEFAULT_SOURCE: Path = PROJECT_ROOT / "datasets" / "test"
DEFAULT_OUTPUT: Path = PROJECT_ROOT / "历史产出" / "复赛_v6"


def infer_imgsz(weights: Path, fallback: int = 1536) -> int:
    """从D-FINE检查点读取训练输入尺寸，读取失败时使用兼容默认值。"""
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        value = config.get("imgsz")
        if isinstance(value, int) and value > 0:
            return value
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        pass
    return fallback


def main() -> None:
    """将 D-FINE 参数交给独立的 D-FINE 预测实现。"""
    # 先固定项目根目录，保证predict1能导入仓库的src.yolo；D-FINE路径由其后端按需加载。
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(1, str(PROJECT_ROOT / "src" / "D-FINE"))
    # 预测实现与训练入口分开；predict1 不会被默认调用。
    from predict1 import PredictionConfig, predict

    # D-FINE的位置编码和锚点必须使用训练时尺寸；从权重自动读取，避免手工同步。
    model_imgsz = infer_imgsz(DEFAULT_WEIGHTS)

    predict(PredictionConfig(
        weights=DEFAULT_WEIGHTS,
        source=DEFAULT_SOURCE,
        output=DEFAULT_OUTPUT,
        backend="dfine",
        expected_count=1000,
        imgsz=model_imgsz,
        batch=4,
        workers=8,
        save_workers=8,
        prefetch_batches=2,
        pin_memory=True,
        device="0",
        conf=0.001,
        iou=0.7,
        max_det=100,
        multi_label=False,
        visual_conf=0.25,
        png_compression=1,
        log_every=25,
        yolo_profile="v4",
        height=1280,
        phase="round2",
        export_materials=True,
        team_id="AIC-2026-81588292",
        team_name="牛副队",
        captain="潘炜德",
    ), argv=sys.argv[1:])


if __name__ == "__main__":
    main()
