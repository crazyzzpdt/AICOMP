"""在 RTX 5080 16 GB 上训练 YOLO26l RGB、红外、深度融合检测模型。

保留「docs/Ultralytics训练参数参考.md」的 83 个参数，并显式设置验证阈值。
训练直接使用 datasets/train、datasets/val 与 datasets/data.yaml，标签为可追溯清洗版本。
v5_full 使用受限弱类重采样、原尺寸同步目标裁剪、RGB 独立增强和骨干低学习率。
学习率在前 200 轮完成衰减，第 161 轮起关闭 Mosaic、目标裁剪与辅助模态缺失。
5000 轮仍影响双头损失日程；patience=200 为完整模态收尾留出训练机会。
best.pt 按 mAP50-95 选取，另存 best_map50.pt、best_map5095.pt 与逐类 AP 记录。
配置依据、数据划分局限和本机实测见「docs/训练配置与数据集复核.md」。
在项目目录执行 uv run python main.py 开始训练。

断点续训（将路径改为实际运行目录）：
    将 RESUME_PATH 设置为新配方运行目录的 weights/last.pt，再运行本文件。
    续训恢复检查点中的优化器和训练参数；修改配方时应保持 RESUME_PATH=None。

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
from 训练优化 import OptimizedMultimodalTrainer


# v4 已取得线上 55.6000 分，保留同一基底；从 COCO 重新迁移以恢复球类初始化。
MODEL_PATH: str = r"./orgin_models/yolo26l.pt"
# 12 类清洗版三模态配置，三种图像及独立标签均在 datasets 的既有 1744/256 划分内。
DATA_PATH: str = r"./datasets/data.yaml"
# 训练结果根目录
PROJECT_PATH: str = str(Path(__file__).resolve().parent / "runs" / "detect")
# 本轮更换优化配方，从本地 COCO 权重重新训练；仅恢复同配方运行时填写 last.pt。
RESUME_PATH: str | None = None
# 保留用户设置的训练上限；调整此值时自动同步 Mosaic 的末段轮数。
MAX_EPOCHS: int = 5000
# 弱类重采样增加每轮样本，采用较低拼图概率，并延后关闭以保持场景多样性。
MOSAIC_EPOCHS: int = 160
# 沿用 v4 学习率，训练器将预训练骨干降为 0.2 倍；首层与检测头正常学习。
INITIAL_LR: float = 0.0001
# 原尺寸裁剪与收尾阶段单独命名，保留旧 v5 配方及 v4 正式成绩基线。
RUN_NAME: str = "AIC_RGBIRDepth_yolo26l_1280_v5_full"


# Windows 创建 DataLoader 子进程时会重新导入当前脚本，因此训练代码必须放在入口保护内。
if __name__ == "__main__":

    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前配置需要支持 BF16 的 CUDA 显卡")
    if not Path(RESUME_PATH or MODEL_PATH).is_file() or not Path(DATA_PATH).is_file():
        raise FileNotFoundError("模型权重或三模态数据配置不存在，请检查路径")
    if RESUME_PATH:
        YOLO(RESUME_PATH).train(
            trainer=OptimizedMultimodalTrainer,
            resume=True,
            data=DATA_PATH,
            workers=4,
            cache="disk",
            save_dir=str(Path(RESUME_PATH).resolve().parent.parent),
        )
        raise SystemExit(0)
    # 实例化模型类
    model: YOLO = YOLO(str(MODEL_PATH), task="detect")
    # 训练函数
    model.train(
        trainer=OptimizedMultimodalTrainer,  # 受限弱类采样、同步增强、分层学习率及双指标权重留存
        # 一、模型、数据与训练时长
        model=str(MODEL_PATH),  # 显式记录本次训练使用的本地权重；与上方加载路径保持一致
        data=str(DATA_PATH),  # 12 类检测数据集配置
        pretrained=True,  # 初始训练从 MODEL_PATH 指向的官方预训练权重开始
        cls_remap=True,  # 按名称迁移类别输出参数；自定义训练器补充 sports ball 到 ball 的对应
        epochs=MAX_EPOCHS,  # 最大训练轮数；与下方关闭 Mosaic 的轮数使用同一个上限
        time=None,  # 最大训练小时数；设置后覆盖 epochs 限制
        patience=200,  # 为第 161 轮后的完整模态收尾留出机会；仍按历史最佳保存，不保证末轮最好
        batch=4,  # v3 完成 250 轮验证过的批次；资源配置保持稳定以便比较配方
        imgsz=1280,  # 32 的整数倍；相比 960，提高小目标在输入图中的有效像素数
        fraction=1.0,  # 使用清洗后的全部 1744 张训练图，256 张验证图不参与训练
        single_cls=False,  # 不将所有类别合并为一个类别
        classes=None,  # None 使用全部类别，也可指定类别 ID 列表

        # 二、设备、性能与可复现性
        device=0,  # PyTorch 中的 CUDA 设备编号（本机 RTX 5080）
        workers=4,  # 保留用户配置；自定义加载器训练与验证均使用 4 个进程
        cache="disk",  # 五通道完整原尺寸缓存约 18 GiB；保留已完成训练的磁盘缓存方案
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
        save_period=50,  # 两种 AP 最佳权重逐轮留存；每 50 轮额外保存完整检查点以控制磁盘增长
        project=PROJECT_PATH,  # 训练结果根目录
        name=RUN_NAME,  # v5 配方单独保存，保留前三次训练产物与线上基线
        exist_ok=False,  # 同名目录已存在时自动递增运行目录名
        save_dir=None,  # 指定确切输出目录会覆盖 project/name，且不自动递增
        resume=False,  # 首次训练；续训在上方 RESUME_PATH 填写 last.pt，由独立分支恢复

        # 四、优化器、学习率与预热
        optimizer="AdamW",  # 针对 1744 张的小数据预训练微调试验；避免沿用旧 MuSGD 的高学习率配方
        lr0=INITIAL_LR,  # 首层/颈部/检测头 0.0001，预训练骨干 0.00002，减轻迁移特征遗忘
        lrf=0.01,  # 最终学习率比例，最终学习率为 lr0 * lrf
        momentum=0.9,  # AdamW 的 beta1，与原动量数值一致
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=5.0,  # 预热期间逐步调整学习率和梯度累积，适应新的 12 类检测头
        warmup_momentum=0.8,  # 预热阶段的初始动量
        warmup_bias_lr=0.0,  # 与本版本 auto 行为一致，避免预热初期偏置学习率过高
        cos_lr=True,  # 自定义训练器在前 200 轮余弦衰减，此后保持 lr0*lrf，独立于总轮数上限
        nbs=16,  # batch=4 时预热后约累积 4 批，有效批次约 16，保持 v3 的更新频率

        # 五、检测损失与可选知识蒸馏
        box=7.5,  # 边界框损失权重
        cls=0.5,  # 分类损失权重
        cls_pw=0.0,  # v5 用受限采样提高弱类曝光；关闭同时放大该类负样本 BCE 的频次加权
        dfl=1.5,  # 框距离回归损失权重，YOLO26 无 DFL 检测头使用 L1 损失
        distill_model=None,  # 不使用教师模型；可填本地教师权重的路径字符串
        dis=6.0,  # 开启知识蒸馏后的蒸馏损失权重

        # 六、训练期间验证
        val=True,  # 开启验证
        split="val",  # 使用独立验证集，不把官方测试集用于选权重
        conf=0.001,  # 保留低分候选用于计算完整 PR 曲线；不是展示图片时的置信度阈值
        iou=0.7,  # NMS 去重阈值，与 mAP50-95 的评测 IoU 阈值区间不是同一含义
        nms=True,  # 明确使用一对多头；v5 验证器的单标签 NMS 与现有 predict.py 对齐
        max_det=100,  # 对齐比赛每图最多 100 个预测框；当前官方训练标注最大为 66 个
        plots=True,  # 保存训练曲线、验证指标和预测示例图

        # 七、检测数据增强
        rect=False,  # 矩形批次最小填充；False 使用正方形输入
        multi_scale=0.0,  # 多尺度尺寸变化比例；0 关闭，例如 0.25 表示 0.75~1.25 倍
        close_mosaic=max(MAX_EPOCHS - MOSAIC_EPOCHS, 0),  # 第 161 轮同时关闭拼图、目标裁剪与模态缺失，轻度 RGB 增强继续
        hsv_h=0.0,  # 框架整图 HSV 关闭；训练优化.py 仅对 RGB 三通道施加轻度 HSV
        hsv_s=0.0,  # 同上
        hsv_v=0.0,  # 同上
        degrees=0.0,  # 随机旋转角度范围 ±degrees
        translate=0.05,  # 减少边缘小目标被平移裁掉，同时保留弱类上下文裁剪
        scale=0.2,  # 缩放约 0.8~1.2；目标裁剪另行提供放大，降低小球被缩至无效尺寸的概率
        shear=0.0,  # 随机剪切角度
        perspective=0.0,  # 透视变换幅度，通常为 0~0.001
        flipud=0.0,  # 上下翻转概率
        fliplr=0.5,  # 左右翻转概率
        bgr=0.0,  # RGB/BGR 通道顺序翻转概率
        mosaic=0.25,  # 降低拼图比例以保护小目标，前 160 轮保留四图场景组合
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
