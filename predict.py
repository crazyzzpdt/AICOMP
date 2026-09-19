"""批量预测 RGB、红外、深度五通道图像，并生成赛事提交 ZIP。

读取、推理与保存并行调度；检测保持 FP32、单模型、训练同口径预处理。
结果存入 images、labels、比赛提交内容三个子目录，已有输出目录不覆盖。

在项目根目录执行预测：
    uv run python predict.py
调整批次与输出位置：
    uv run python predict.py --batch 4 --output predict_v9_batch4
使用历史 YOLO 权重：
    uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_yolo26l_1280_v4_clean_lr1e4/weights/best.pt --imgsz 1280 --yolo-profile v4 --output predict_v4_restored
查看命令行覆盖参数：
    uv run python predict.py --help
"""

# 内置库
import os
import sys
from pathlib import Path

# 必须在导入模型组件前设置，预测不联网下载或上传遥测。
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# 自己的模块
from 三模态预测 import PredictionConfig, predict


# 相对路径以入口文件所在目录为基准，兼容 IDE 从其他位置启动。
PROJECT_ROOT: Path = Path(__file__).resolve().parent
# 本轮训练完成后使用v9 AP95最佳权重；同名训练递增目录须填写实际路径。
MODEL_PATH: Path = PROJECT_ROOT / "runs/detect/AIC_RGBIRDepth_yolo26l_1920x1080_v9_fusion/weights/best.pt"
# 官方初赛的同名 visible、infrared、depth 三模态图像。
SOURCE_PATH: Path = PROJECT_ROOT / "数据集/测试集/AIC2026_PHASE_1_1000"
# 与训练产物隔离；再次预测时修改这里或传 --output，不清空已有结果。
OUTPUT_PATH: Path = PROJECT_ROOT / "predict_v9"


# Windows 子进程或其他模块可能导入入口，正式预测必须放在入口保护内。
if __name__ == "__main__":
    predict(PredictionConfig(
        # 一、模型、数据与输出
        weights=MODEL_PATH,  # v9训练完成后使用同一训练轨迹的EMA最佳模型
        source=SOURCE_PATH,  # 只读官方三模态图像，不生成缓存或修改原始文件
        output=OUTPUT_PATH,  # 保存带框图片、六列标签和提交 ZIP，已有目录不覆盖
        backend="auto",  # 从检查点识别融合结构；仍兼容旧YOLO和D-FINE
        expected_count=1000,  # 已确认初赛为 1000 组，仅核对文件清单避免交错目录

        # 二、设备、加载与资源：由用户调整，不自动试跑探测显存
        imgsz=1920,  # 与v9训练一致的内容宽度，原1920宽图片不先缩到1280
        height=1080,  # 内容高度1080；网络画布仅补齐到1088，不拉伸
        batch=4,  # 全尺寸新结构FP32批量起点，由用户调整，不自动探测
        workers=8,  # 并行解码三模态并提前缩放，给 GPU 连续准备输入
        save_workers=8,  # 后台画框、编码和写文件，不再逐张阻塞下一次推理
        prefetch_batches=2,  # 全尺寸图片预取2批，控制五通道与待保存原图内存
        pin_memory=True,  # 当前批次uint8锁页传输，在GPU转FP32并归一化
        device="0",  # 本机 RTX 5080；保持 FP32，不默认使用半精度或 INT8

        # 三、检测候选与后处理：保持比赛口径，不用降低精度换速度
        conf=0.001,  # 保留低分候选计算 AP，与验证一致，不是图片展示阈值
        iou=0.7,  # v9与轮末验证共享单标签NMS；D-FINE仍忽略此参数
        max_det=100,  # 赛事每图最多 100 框，按置信度保留，不做多模型集成
        multi_label=False,  # YOLO 默认单标签；D-FINE 原生查询类别排序不受此开关控制
        yolo_profile="v4",  # 仅用于旧YOLO；v9自动采用训练共用的矩形批量后端

        # 四、可视化与进度：只影响输出耗时和图片，不改提交预测框
        visual_conf=0.25,  # 仅绘制较高置信度框，六列 TXT 仍保留 conf 以上候选
        png_compression=1,  # PNG 低级别无损压缩，以磁盘空间换编码速度，JPG 保持默认
        log_every=25,  # 按批次报告累计进度，结束后统计包含读图和保存的实际吞吐量
    ), argv=sys.argv[1:])
