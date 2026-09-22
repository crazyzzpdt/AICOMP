"""D-FINE 独立预测入口，使用复赛 ``datasets/test`` 三模态数据。

D-FINE 采用训练一致的正方形预处理和原生查询排序，不调用 YOLO NMS。
结果只写入 ``历史产出``；需要补充团队材料时使用 ``--materials-only``。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_WEIGHTS: Path = PROJECT_ROOT / "runs" / "detect" / "AIC_RGBIRDepth_dfine_l_1280_v8" / "weights" / "best.pth"
DEFAULT_SOURCE: Path = PROJECT_ROOT / "datasets" / "test"
DEFAULT_OUTPUT: Path = PROJECT_ROOT / "历史产出" / "predict_dfine"


def main() -> None:
    """将 D-FINE 参数交给独立的 D-FINE 预测实现。"""
    # 让官方D-FINE的src包优先于仓库的src命名空间，再加载D-FINE预测实现。
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "D-FINE"))
    # 预测实现与训练入口分开；predict1 不会被默认调用。
    from predict1 import PredictionConfig, predict

    predict(PredictionConfig(
        weights=DEFAULT_WEIGHTS,
        source=DEFAULT_SOURCE,
        output=DEFAULT_OUTPUT,
        backend="dfine",
        expected_count=1000,
        imgsz=1280,
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
