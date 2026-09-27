"""D-FINE 独立预测入口，使用复赛 ``datasets/test`` 三模态数据。

D-FINE 采用训练一致的正方形预处理和原生查询排序，不调用 YOLO NMS。
结果只写入 ``历史产出``，生成同名六列TXT、带框图片和复赛材料，已有结果不覆盖。
"""

from __future__ import annotations

# 内置库
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_WEIGHTS: Path = PROJECT_ROOT / "runs/detect/AIC_RGBIRDepth_dfine_x_1536_v27/weights/best.pth"
DEFAULT_SOURCE: Path = PROJECT_ROOT / "datasets" / "test"
DEFAULT_OUTPUT: Path = PROJECT_ROOT / "历史产出" / "复赛predict_v27_dfine_x_max100"


def main() -> None:
    """将 D-FINE 参数交给独立的 D-FINE 预测实现。"""
    # 上游模块由独立运行时按需加载，不导入另一个预测入口或YOLO项目代码。
    from src.dfine.prediction import PredictionConfig, predict

    predict(PredictionConfig(
        # 一、模型、输入与输出
        weights=DEFAULT_WEIGHTS,  # 保留v27-X的EMA权重，按检查点variant、尺寸和浮点协议构建模型
        source=DEFAULT_SOURCE,  # visible、infrared、depth同名配对，不改原图或生成缓存
        output=DEFAULT_OUTPUT,  # 本次结果目录，已存在则拒绝覆盖，可传--output另选目录
        expected_count=1000,  # 复赛每种模态1000张，缺失或重名立即报错

        # 二、输入几何、精度与资源
        imgsz=0,  # CLI解析后从最终权重读取训练尺寸，缺失报错，不默认猜1536
        height=1536,  # 执行前随最终imgsz统一，D-FINE始终使用检查点规定的方形画布
        batch=1,  # X模型单张前向，不自动探测显存或重试
        workers=8,  # 并行读图和预处理，不使用训练DataLoader
        save_workers=8,  # 后台绘图、编码与保存，仅持有CPU结果
        prefetch_batches=2,  # 有界预取，不将全部五通道图像放入内存
        pin_memory=True,  # 当前批次锁页传输，网络仅除255一次
        device="0",  # 权重及前向FP32、TF32关闭，不加训练用随机红外/深度扰动

        # 三、D-FINE原生候选筛选：sigmoid后联合排序query与类别，不追加NMS
        conf=0.001,  # 提交候选最低分数，不是展示阈值，不人为修改模型置信度
        max_det=100,  # 联合top300候选恢复原图、裁边和过滤后最多保留100框

        # 四、结果图片、标签与进度
        visual_conf=0.25,  # 仅筛选带框图展示，TXT仍保留conf阈值以上的最终候选
        png_compression=1,  # 输出PNG无损压缩，不压缩模型输入
        log_every=25,  # 每25张报告进度，结束记录实际TXT框数和触顶图数

        # 五、复赛材料
        report=PROJECT_ROOT / "docs/技术方案.md",  # 由tools统一转换并生成官方复赛目录
    ), argv=sys.argv[1:])


if __name__ == "__main__":
    main()
