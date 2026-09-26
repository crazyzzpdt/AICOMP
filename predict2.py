"""D-FINE 独立预测入口，使用复赛 ``datasets/test`` 三模态数据。

D-FINE 采用训练一致的正方形预处理和原生查询排序，不调用 YOLO NMS。
结果只写入 ``历史产出``；需要补充团队材料时使用 ``--materials-only``。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_WEIGHTS: Path = PROJECT_ROOT / "runs/detect/AIC_RGBIRDepth_dfine_x_1536_v27/weights/best.pth"
DEFAULT_SOURCE: Path = PROJECT_ROOT / "datasets" / "test"
DEFAULT_OUTPUT: Path = PROJECT_ROOT / "历史产出" / "复赛predict_v27_dfine_x"


def main() -> None:
    """将 D-FINE 参数交给独立的 D-FINE 预测实现。"""
    # 上游模块由独立运行时按需加载，不导入另一个预测入口或YOLO项目代码。
    from src.dfine.prediction import PredictionConfig, predict

    predict(PredictionConfig(
        weights=DEFAULT_WEIGHTS,  # 本轮v27-X；按检查点variant、尺寸和浮点协议构建模型
        source=DEFAULT_SOURCE,
        output=DEFAULT_OUTPUT,
        expected_count=1000,
        imgsz=0,  # CLI解析后从最终权重读取训练尺寸，缺失报错，不默认猜1536
        batch=1,
        workers=8,
        save_workers=8,
        prefetch_batches=2,
        pin_memory=True,
        device="0",  # 权重及前向FP32、TF32关闭；推理不加训练用随机红外/深度扰动
        conf=0.001,
        max_det=12,
        visual_conf=0.25,
        png_compression=1,
        log_every=25,
        height=1536,
        report=PROJECT_ROOT / "docs/技术方案.md",  # 由tools统一转换并生成官方复赛目录
    ), argv=sys.argv[1:])


if __name__ == "__main__":
    main()
