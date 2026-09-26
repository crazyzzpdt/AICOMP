"""YOLO三模态预测入口，输出图片、六列标签与提交ZIP，已有结果不覆盖。

运行：uv run python predict1.py
覆盖：uv run python predict1.py --batch 2 --output 历史产出/复赛predict_v27_yolo26x_第二次
"""
from __future__ import annotations

# 内置库
import sys
from pathlib import Path

# 自己的模块
from src.yolo.prediction import PredictionConfig, predict


PROJECT_ROOT: Path = Path(__file__).resolve().parent
MODEL_PATH: Path = PROJECT_ROOT / "runs/detect/AIC_RGBIRDepth_yolo26x_1536_v27_bn_fix/weights/best.pt"
SOURCE_PATH: Path = PROJECT_ROOT / "datasets/test"
OUTPUT_PATH: Path = PROJECT_ROOT / "历史产出/复赛predict_v27_yolo26x_bn_fix"


if __name__ == "__main__":
    predict(PredictionConfig(
        # 一、模型、输入与输出
        weights=MODEL_PATH,  # 仅接收本项目YOLO .pt权重，保留本轮v27-X选择
        source=SOURCE_PATH,  # visible、infrared、depth同名配对，不改原图或生成缓存
        project=OUTPUT_PATH.parent,  # 预测产出根目录，与训练runs分离
        name=OUTPUT_PATH.name,  # 本次结果目录；再次运行请改名或传--output
        data=None,  # 从检查点读取并核对12类顺序，不从其他YAML替换名称
        expected_count=1000,  # 复赛每种模态1000张，缺失或重名立即报错

        # 二、输入几何、精度与资源
        imgsz=1536,  # 与v27训练一致；也支持(height, width)指定历史矩形模型内容尺寸
        height=1536,  # 整数imgsz时的内容高度，须匹配融合权重记录
        rect=None,  # 随权重自动确定：v27固定方形False，历史v4最小矩形True
        quantize=32,  # 固定FP32前向、TF32关闭，保留连续浮点IR/深度输入
        device="0",  # RTX 5080，也可指定cpu；不自动探测显存或重试
        batch=1,  # X模型单张前向；历史v4配方保留逐张前向
        workers=8,  # 并行读图和预处理，不使用训练DataLoader
        save_workers=8,  # 后台绘图、编码与保存，仅持有CPU结果
        prefetch_batches=2,  # 有界预取，不将全部五通道图像放入内存
        pin_memory=True,  # 当前批次锁页传输，网络仅除255一次
        channels_last=False,  # 保持NCHW布局，True可请求channels_last内存格式
        compile=False,  # 当前自定义路径不启用图编译，非False会明确报错
        dnn=False,  # 当前为PyTorch .pt，不使用ONNX/OpenCV DNN
        stream=False,  # 旧YOLO可返回批内生成器；外层始终逐批消费，不积累全部结果

        # 三、候选框与后处理
        conf=0.001,  # 与验证一致保留低分候选，不是展示图片的阈值
        iou=0.7,  # 一对多检测头的NMS阈值
        max_det=12,  # 用户选择每图保留12框；赛事手册允许100框，非类别数量限制
        nms=True,  # 保持验证的一对多头，本路径不开放无NMS切换
        agnostic_nms=False,  # 不跨类别压制；True会改变重叠目标筛选口径
        classes=None,  # None预测全部12类，可用编号列表过滤，正式提交保持None
        multi_label=False,  # 融合权重保持单标签；历史current路径才支持多标签
        yolo_profile="v4",  # 只控制历史权重，v27自动读取自身几何协议
        augment=False,  # 不启用TTA或训练随机增强，本路径未接RGB专用TTA

        # 四、结果图片、标签与进度
        save=True,  # 保存RGB带框图；False跳过绘图，仍生成标签和提交ZIP
        save_txt=True,  # 赛事每图要求同名TXT，无检测也创建空文件
        save_conf=True,  # 固定六列class/cx/cy/w/h/conf，不能关闭
        show=False,  # 后台批量处理不弹窗，本路径不支持GUI显示
        save_crop=False,  # 保持三个输出目录，不生成目标裁剪集
        show_labels=True,  # 带框图片显示类别名
        show_conf=True,  # 带框图片显示置信度
        show_boxes=True,  # 带框图片绘制边界框
        line_width=None,  # 随原图尺寸确定线宽，也可指定正整数
        visual_conf=0.25,  # 仅影响展示，TXT仍按conf保留候选
        png_compression=1,  # 输出PNG无损压缩，不压缩模型输入
        verbose=True,  # 显示进度，历史YOLO同时显示框架日志
        log_every=25,  # 至少每25张报告一次进度，结束统计实际吞吐量

        # 五、其他任务参数：静态图检测不启用，非默认值明确报错
        vid_stride=1,  # 视频抽帧不适用于三模态图像目录
        stream_buffer=False,  # 不使用实时视频队列
        save_frames=False,  # 静态图无视频帧保存任务
        visualize=False,  # 自定义五通道前向未接类别激活图接口
        retina_masks=False,  # 目标检测没有分割掩码
        embed=None,  # 返回检测框，不切换为中间特征向量

        # 六、复赛材料
        report=PROJECT_ROOT / "docs/技术方案.md",  # tools工具转为PDF并整合模型、源码、依赖和评估材料
    ), argv=sys.argv[1:])
