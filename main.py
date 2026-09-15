"""在 RTX 5080 16 GB 上训练 YOLO26l RGB、红外、深度融合检测模型。

保留「docs/Ultralytics训练参数参考.md」的 83 个参数，并显式设置验证阈值。
首次运行使用 datasets/multimodal_new_labels 中的新版标注，不会修改 datasets/train 与 datasets/val 的旧副本。
配置依据、数据划分局限和本机实测见「docs/训练配置与数据集复核.md」。
在项目目录执行 uv run python main.py 开始训练。

断点续训（将路径改为实际运行目录）：
    将 RESUME_PATH 设置为实际运行目录的 weights/last.pt，再运行本文件。

该模型输入为五通道，不能用只提供 visible 图的通用 yolo predict 命令推理；
提交推理也必须以同名 RGB、红外、深度图融合后再送入模型。
"""
# 内置库
import os
from pathlib import Path
# 必须在导入 Ultralytics 前设置，训练时关闭联网检查和自动安装依赖。
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
# 三方库
import torch
from ultralytics import YOLO
# 自己的模块
from 三模态训练 import MultimodalDetectionTrainer


# n/s 更适合速度优先；m 更省资源；x 计算量明显更大。
MODEL_PATH: str =  r"./orgin_models/yolo26l.pt"
# 12 类三模态数据集配置，沿用既有 1744/256 划分，标签来自 new_labels_2000。
DATA_PATH: str = r"./datasets/multimodal_new_labels/data.yaml"
# 训练结果根目录
PROJECT_PATH: str = str(Path(__file__).resolve().parent / "runs" / "detect")
RESUME_PATH: str | None = None  # 续训时填写五通道 last.pt 路径；首次训练保持 None


# Windows 创建 DataLoader 子进程时会重新导入当前脚本，因此训练代码必须放在入口保护内。
if __name__ == "__main__":

    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前配置需要支持 BF16 的 CUDA 显卡")
    if not Path(RESUME_PATH or MODEL_PATH).is_file() or not Path(DATA_PATH).is_file():
        raise FileNotFoundError("模型权重或三模态数据配置不存在，请检查路径")
    if RESUME_PATH:
        YOLO(RESUME_PATH).train(trainer=MultimodalDetectionTrainer, resume=True)
        raise SystemExit(0)
    # 实例化模型类
    model: YOLO = YOLO(str(MODEL_PATH), task="detect")
    # 训练函数
    model.train(
        trainer=MultimodalDetectionTrainer,  # 用五通道数据加载器同步融合 RGB、红外、深度
        # 一、模型、数据与训练时长
        model=str(MODEL_PATH),  # 显式记录本次训练使用的本地权重；与上方加载路径保持一致
        data=str(DATA_PATH),  # 12 类检测数据集配置
        pretrained=True,  # 初始训练从 MODEL_PATH 指向的官方预训练权重开始
        cls_remap=True,  # 按类别名称匹配并迁移预训练分类头中的对应参数
        epochs=1000,  # 最大训练轮数；为后续扩大训练集留出上限，实际可由早停提前结束
        time=None,  # 最大训练小时数；设置后覆盖 epochs 限制
        patience=100,  # 验证指标连续 100 轮未改善时提前停止，最终使用 best.pt
        batch=4,  # 本机五通道 1280/BF16/MuSGD 短训峰值已分配 10.89 GiB、保留 11.87 GiB
        imgsz=1280,  # 32 的整数倍；相比 960，提高小目标在输入图中的有效像素数
        fraction=1.0,  # 使用全部数据；也可指定比例、数量或各拆分的列表
        single_cls=False,  # 不将所有类别合并为一个类别
        classes=None,  # None 使用全部类别，也可指定类别 ID 列表

        # 二、设备、性能与可复现性
        device=0,  # PyTorch 中的 CUDA 设备编号（本机 RTX 5080）
        workers=4,  # 9950X 的 4 个加载进程，兼顾供数速度和 Windows 子进程内存开销
        cache="disk",  # 五通道完整原尺寸缓存约 18 GiB；本机可用内存约 16 GiB，采用磁盘缓存
        amp="bf16",  # RTX 5080 原生支持；本机已验证有限损失和梯度，且无需 FP16 的缩放器
        quantize=None,  # None 关闭量化感知训练；8/"int8" 开启 QAT
        seed=0,  # 随机种子
        deterministic=True,  # 使用确定性算法，尽量保证可复现性
        profile=False,  # 是否在训练时分析 ONNX/TensorRT 推理速度
        freeze=None,  # 不冻结；可设置前 N 层或层索引/模块名称列表
        compile=False,  # 关闭 torch.compile；可设 True 或支持的编译模式字符串
        channels_last=False,  # 保持已实测的 NCHW 内存布局
        verbose=True,  # 输出详细训练信息

        # 三、结果保存与恢复
        save=True,  # 保存训练检查点和最终权重
        save_period=25,  # 每 25 轮额外留档；best.pt/last.pt 按框架训练流程保存
        project=PROJECT_PATH,  # 训练结果根目录
        name="AIC_RGBIRDepth_yolo26l_1280",  # 区分三模态融合、模型规格和输入尺寸
        exist_ok=False,  # 同名目录已存在时自动递增运行目录名
        save_dir=None,  # 指定确切输出目录会覆盖 project/name，且不自动递增
        resume=False,  # 首次训练；续训在上方 RESUME_PATH 填写 last.pt，由独立分支恢复

        # 四、优化器、学习率与预热
        optimizer="MuSGD",  # 显式使用 YOLO26 长周期训练优化器，避免 auto 随轮数切换配方
        lr0=0.01,  # 与本版本长周期 auto 配方一致；MuSGD 对部分分类头参数另有学习率倍率
        lrf=0.01,  # 最终学习率比例，最终学习率为 lr0 * lrf
        momentum=0.9,  # 与 MuSGD 的长周期 auto 配方一致
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=5.0,  # 预热期间逐步调整学习率和梯度累积，适应新的 12 类检测头
        warmup_momentum=0.8,  # 预热阶段的初始动量
        warmup_bias_lr=0.0,  # 与本版本 auto 行为一致，避免预热初期偏置学习率过高
        cos_lr=True,  # 按 1000 轮规划余弦下降；早停时可能尚未达到最终学习率
        nbs=64,  # 预热后 batch=4 累积约 16 个批次再更新，名义有效批次约为 64

        # 五、检测损失与可选知识蒸馏
        box=7.5,  # 边界框损失权重
        cls=0.5,  # 分类损失权重
        cls_pw=0.25,  # 按训练集频次做温和的四次方根逆频率加权，兼顾稀有类和常见类
        dfl=1.5,  # 框距离回归损失权重，YOLO26 无 DFL 检测头使用 L1 损失
        distill_model=None,  # 不使用教师模型；可填本地教师权重的路径字符串
        dis=6.0,  # 开启知识蒸馏后的蒸馏损失权重

        # 六、训练期间验证
        val=True,  # 开启验证
        split="val",  # 使用独立验证集，不把官方测试集用于选权重
        conf=0.001,  # 保留低分候选用于计算完整 PR 曲线；不是展示图片时的置信度阈值
        iou=0.7,  # NMS 去重阈值，与 mAP50-95 的评测 IoU 阈值区间不是同一含义
        nms=None,  # 验证、早停和权重选择使用带 NMS 的检测头；False 时优先使用可用的无 NMS 头
        max_det=100,  # 对齐比赛每图最多 100 个预测框；当前官方训练标注最大为 66 个
        plots=True,  # 保存训练曲线、验证指标和预测示例图

        # 七、检测数据增强
        rect=False,  # 矩形批次最小填充；False 使用正方形输入
        multi_scale=0.0,  # 多尺度尺寸变化比例；0 关闭，例如 0.25 表示 0.75~1.25 倍
        close_mosaic=40,  # 最后 40 轮回到非 Mosaic 图像分布，配合较低学习率细化定位
        hsv_h=0.0,  # 五通道图像不能安全套用 RGB HSV；关闭以保护红外和深度物理含义
        hsv_s=0.0,  # 同上
        hsv_v=0.0,  # 同上
        degrees=0.0,  # 随机旋转角度范围 ±degrees
        translate=0.1,  # 水平/垂直平移比例范围
        scale=0.3,  # 缩放系数约 0.7~1.3，减轻小目标在强缩小时消失的问题
        shear=0.0,  # 随机剪切角度
        perspective=0.0,  # 透视变换幅度，通常为 0~0.001
        flipud=0.0,  # 上下翻转概率
        fliplr=0.5,  # 左右翻转概率
        bgr=0.0,  # RGB/BGR 通道顺序翻转概率
        mosaic=1.0,  # 四图 Mosaic 增强概率
        mixup=0.0,  # MixUp 图像混合概率
        cutmix=0.0,  # CutMix 局部区域混合概率
        augmentations=None,  # 自定义 Albumentations 变换对象列表，仅 Python API 支持

        # 八、参考表中的其他任务专用参数：完整列出，当前 detect 任务不使用
        pose=12.0,  # 姿态估计：姿态损失权重
        kobj=1.0,  # 姿态估计：关键点目标性损失权重
        rle=1.0,  # 姿态估计：残差对数似然损失权重
        angle=1.0,  # OBB 旋转框：角度损失权重
        dlog=1.0,  # 深度估计：SILog 损失权重
        dgrad=0.5,  # 深度估计：梯度损失权重
        dlam=1.0,  # 深度估计：SILog 方差关注因子
        overlap_mask=True,  # 实例分割：合并对象掩码
        mask_ratio=4,  # 实例分割：掩码下采样比例
        dropout=0.0,  # 分类：随机丢弃率
        copy_paste=0.0,  # 分割/OBB：对象复制粘贴比例
        copy_paste_mode="flip",  # 分割/OBB：复制粘贴策略，支持 flip/mixup
        auto_augment="randaugment",  # 分类：自动增强策略
        erasing=0.4,  # 分类：随机擦除概率
    )
