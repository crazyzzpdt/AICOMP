"""使用 orgin_models 中的官方预训练权重训练目标检测模型。

已逐项写入「比赛资料/Ultralytics训练参数参考.md」列出的 83 个训练与增强参数。
当前安装的 Ultralytics 还提供 split、save_json、conf、iou、dnn 等验证细节参数；
它们不在该参考清单内，因此沿用框架默认值。
在项目目录执行 uv run python main.py 开始训练。

断点续训（将路径改为实际运行目录）：
    yolo detect train resume=True model="runs/detect/AI COMP/weights/last.pt"

推理：
    yolo predict model="runs/detect/AI COMP/weights/best.pt" source="数据集/测试集/AIC2026_PHASE_1_1000/visible" imgsz=940 device=0 save=True
"""
# 三方库
from ultralytics import YOLO



# 切换模型只需修改文件名：yolo26n.pt / yolo26s.pt / yolo26m.pt / yolo26l.pt / yolo26x.pt。
MODEL_PATH: str =  r"./orgin_models/yolo26m.pt"
# 12 类检测数据集配置
DATA_PATH: str = r"./datasets/data.yaml"
# 训练结果根目录
PROJECT_PATH: str = r"./runs/detect"


# Windows 创建 DataLoader 子进程时会重新导入当前脚本，因此训练代码必须放在入口保护内。
if __name__ == "__main__":
    # 实例化模型类
    model: YOLO = YOLO(str(MODEL_PATH), task="detect")
    model.train(
        # 一、模型、数据与训练时长
        model=str(MODEL_PATH),  # 显式记录本次训练使用的本地权重；与上方加载路径保持一致
        data=str(DATA_PATH),  # 12 类检测数据集配置
        pretrained=True,  # 初始训练从 MODEL_PATH 指向的官方预训练权重开始
        cls_remap=True,  # 按类别名称匹配并迁移预训练分类头中的对应参数
        epochs=10000,  # 最大训练轮数，保留原配置
        time=None,  # 最大训练小时数；设置后覆盖 epochs 限制
        patience=100,  # 验证指标连续 100 轮未改善时提前停止
        batch=32,  # 批次大小；-1 自动分配，0.70 表示按 70% 显存自动估算
        imgsz=940,  # 输入尺寸，框架会按模型步幅向上对齐
        fraction=1.0,  # 使用全部数据；也可指定比例、数量或各拆分的列表
        single_cls=False,  # 不将所有类别合并为一个类别
        classes=None,  # None 使用全部类别，也可指定类别 ID 列表

        # 二、设备、性能与可复现性
        device=0,  # PyTorch 中的 CUDA 设备编号（本机 RTX 5080）
        workers=0,  # 数据加载进程数，保留原 Windows 配置
        cache=False,  # False 不缓存；True/"ram" 缓存到内存；"disk" 缓存到磁盘
        amp=True,  # 混合精度；还支持 "fp16"、"bf16"、False/"fp32"
        quantize=None,  # None 关闭量化感知训练；8/"int8" 开启 QAT
        seed=0,  # 随机种子
        deterministic=True,  # 使用确定性算法，尽量保证可复现性
        profile=False,  # 是否在训练时分析 ONNX/TensorRT 推理速度
        freeze=None,  # 不冻结；可设置前 N 层或层索引/模块名称列表
        compile=False,  # 关闭 torch.compile；可设 True 或支持的编译模式字符串
        channels_last=None,  # 自动选择内存格式；Windows 默认不开启
        verbose=True,  # 输出详细训练信息

        # 三、结果保存与恢复
        save=True,  # 保存训练检查点和最终权重
        save_period=100,  # 每 100 轮额外保存检查点；-1 关闭周期性额外保存
        project=PROJECT_PATH,  # 训练结果根目录
        name="AI COMP",  # 本次运行名称
        exist_ok=False,  # 同名目录已存在时自动递增运行目录名
        save_dir=None,  # 指定确切输出目录会覆盖 project/name，且不自动递增
        resume=False,  # 初始训练不续训；恢复训练请使用文件顶部的 last.pt 命令

        # 四、优化器、学习率与预热
        optimizer="auto",  # 可选 SGD、MuSGD、Adam、Adamax、AdamW、NAdam、RAdam、RMSProp
        # auto 会按训练迭代次数选择优化器，并自动调整 lr0、momentum、warmup_bias_lr。
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率比例，最终学习率为 lr0 * lrf
        momentum=0.937,  # SGD 动量或 Adam beta1
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3.0,  # 预热轮数
        warmup_momentum=0.8,  # 预热阶段的初始动量
        warmup_bias_lr=0.1,  # 预热阶段偏置学习率，auto 优化器下自动设为 0.0
        cos_lr=False,  # 是否使用余弦学习率调度
        nbs=64,  # 用于归一化损失的名义批次大小

        # 五、检测损失与可选知识蒸馏
        box=7.5,  # 边界框损失权重
        cls=0.5,  # 分类损失权重
        cls_pw=0.0,  # 类别频率倒数加权的幂；0 关闭，1 完整加权
        dfl=1.5,  # 框距离回归损失权重，YOLO26 无 DFL 检测头使用 L1 损失
        distill_model=None,  # 不使用教师模型；可填本地教师权重的路径字符串
        dis=6.0,  # 开启知识蒸馏后的蒸馏损失权重

        # 六、训练期间验证
        val=True,  # 开启验证
        nms=None,  # 验证、早停和权重选择使用带 NMS 的检测头；False 时优先使用可用的无 NMS 头
        max_det=300,  # 每张图像最大检测数；默认值会根据密集标注自动上调
        plots=True,  # 保存训练曲线、验证指标和预测示例图

        # 七、检测数据增强
        rect=False,  # 矩形批次最小填充；False 使用正方形输入
        multi_scale=0.0,  # 多尺度尺寸变化比例；0 关闭，例如 0.25 表示 0.75~1.25 倍
        close_mosaic=10,  # 最后 10 轮关闭 Mosaic；0 表示不在末期关闭
        hsv_h=0.015,  # 色调变化幅度
        hsv_s=0.7,  # 饱和度变化幅度
        hsv_v=0.4,  # 亮度变化幅度
        degrees=0.0,  # 随机旋转角度范围 ±degrees
        translate=0.1,  # 水平/垂直平移比例范围
        scale=0.5,  # 随机缩放幅度，或显式设置 (min, max) 元组
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
