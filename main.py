"""在本机训练 D-FINE-L RGB、红外、深度五通道融合检测模型。

从本地 Objects365 E25 公开预训练权重迁移，不续训 v4/v5，不增加外部训练数据。
直接读取 datasets 的清洗标签与 v8 场景审阅划分；结果保存在独立的 v8 目录。
1280 输入、降低学习率、温和同步增强，正式训练内早停并记录 PNG/JPG 分域指标。
参数为待用户正式训练验证的候选配方，不承诺超过 60 分。

在项目根目录执行训练：
    uv run python main.py
预测新模型并生成赛事提交包：
    uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_dfine_l_1280_v8/weights/best.pth --imgsz 1280 --batch 18 --output predict_v8
断点恢复：
    将 RESUME_PATH 改为同配方中断运行的 weights/last.pth，再运行本文件。
    恢复写入新的运行目录，原结果不覆盖；完成的检查点拒绝当作中断任务恢复。

本轮依据与限制见 docs/v8训练方案与数据清洗.md；不自动启动训练或测试。
"""

# 内置库
import os
from pathlib import Path

# 必须在导入模型组件前设置，训练与预测不联网或上传遥测。
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# 自己的模块
from D细化训练 import TrainingConfig, train


# 官方 D-FINE-L Objects365 E25 权重已预先下载；不读取 v5 项目检查点作为基底。
MODEL_PATH: str = "./orgin_models/dfine_l_obj365_e25.pth"
# v8 清洗数据统一落在 datasets；划分与标签须匹配下方审计记录。
DATA_PATH: str = "./datasets/data.yaml"
# 重新下载来源的落位审计；保留既有划分，训练入口只校验不自动移动数据。
DATA_AUDIT: str = "./runs/dataset_cleaning/official_refresh_20260918_214843/manifest.json"
# 新架构单独存放，已有同名结果自动使用时间戳目录。
PROJECT_PATH: str = "./runs/detect"
# 沿用 v6 的 60 轮余弦周期；连续 10 轮无有效提升时提前结束，不强行跑满。
MAX_EPOCHS: int = 60
# 沿用 v6 第 41 轮关闭尺度扰动；早停可能先于收尾发生。
POLISH_EPOCH: int = 40
# 主学习率和骨干学习率均相对 v6 减半，避免与新增强增强同时调整。
INITIAL_LR: float = 0.00005
# 区分新框架与历史 YOLO 配方，便于对照得分。
RUN_NAME: str = "AIC_RGBIRDepth_dfine_l_1280_v8"
# 仅填写同配方中断任务的 last.pth；正常新训保持 None。
RESUME_PATH: str | None = None


# Windows DataLoader 子进程会重新导入入口，正式训练必须放在入口保护内。
if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    train(TrainingConfig(
        # 一、模型、数据与训练时长
        model=MODEL_PATH,  # 官方 366 输出槽的 Objects365 模型，按比赛类别迁移
        data=DATA_PATH,  # 只读 datasets，不改原始图像或清洗标签
        data_audit=DATA_AUDIT,  # 核验已落位清单与全部标签指纹，拒绝无记录的数据改动
        project=PROJECT_PATH,  # 训练结果根目录
        name=RUN_NAME,  # 新运行不覆盖历史产物
        epochs=MAX_EPOCHS,  # 固定短周期，不再以 5000 轮上限控制收尾
        resume=RESUME_PATH,  # 恢复时校验配方、源码版本和数据签名

        # 二、设备、加载与资源：由用户调整，不自动试跑探测显存
        imgsz=1280,  # 回到线上优于 v7 的 v6 尺寸，取消未经证实有益的 1536 放大
        batch=4,  # v6 args.yaml 的实际物理批次；资源参数仍由用户调整
        effective_batch=16,  # 物理 batch=4 时累积 4 批，保持 v6 有效批次
        val_batch=1,  # FP32 轮末验证与正式预测同精度，减少峰值显存
        workers=4,  # 每进程预取 1 批、关闭锁页，不将完整数据集缓存到 RAM
        device=0,  # 本机 RTX 5080；训练使用 BF16 前向与 FP32 匹配/损失
        seed=0,  # 固定种子；GPU 算子仍可能存在非确定性

        # 三、优化器、学习率与收敛
        lr0=INITIAL_LR,  # AdamW 检测头/颈部/首层学习率
        backbone_lr=0.000005,  # 相对 v6 减半；新增模态首层仍使用主学习率
        lrf=0.01,  # 沿用 60 轮余弦周期；早停不提前压缩学习率曲线
        lr_schedule="optimizer_step",  # 按实际优化步衰减；每轮重采样数量变化时学习率不跳变
        warmup_epochs=3,  # 新类别头和新增输入通道短预热，不照搬 30 轮
        weight_decay=0.0001,  # 恢复 v6 实际配置，避免与学习率同时叠加更强权重衰减
        clip_grad=0.1,  # 沿用 DETR 类模型常见梯度裁剪，非有限梯度立即报告
        ema_decay=0.999,  # 单一训练轨迹的 EMA 用于验证与预测，不融合多个模型输出
        ema_warmup=100,  # 按真实优化步预热 EMA，兼容梯度累积后的更新频率

        # 四、同步增强：回到 v6 温和配方，关闭 v7 新增的裁剪与低清模拟
        polish_epoch=POLISH_EPOCH,  # 第 41 轮起关闭尺度扰动；不因早停而补跑收尾
        scale_min=0.9,  # 沿用 v6 的 0.9–1.0 尺度，不裁掉边缘目标
        crop_prob=0.0,  # 关闭完整目标裁剪，避免与降学习率同时引入尺度分布变化
        crop_min=0.75,  # 裁剪关闭时不使用，保留兼容配置
        lowres_prob=0.0,  # 关闭人工低清模拟；真实 JPG 域由官方图像覆盖
        lowres_min=0.6,  # 低清模拟关闭时不使用
        fliplr=0.5,  # 三模态与标签严格同步水平翻转
        hsv_h=0.01,  # 仅增强 RGB，红外和深度不做颜色变换
        hsv_s=0.15,  # 温和饱和度变化
        hsv_v=0.15,  # 温和亮度变化

        # 五、采样：每张训练原图每轮一次，不重复弱类或降低其他类别采样次数
        boat_repeat=1.0,  # v7 的船类未改善，本轮取消定向重复
        garbage_repeat=1.0,  # 单类改善未转化为线上总体收益，恢复普通采样
        repeat_extra_fraction=0.0,  # 不增加额外样本
        repeat_group_limit=4,  # 重采样关闭时不使用

        # 六、D-FINE 原生损失：沿用官方权重，不与重采样叠加 YOLO cls_pw
        loss_vfl=1.0,  # 质量感知分类损失，不把 YOLO cls=0.5 或 cls_pw=0.25 当作等价设置
        loss_bbox=5.0,  # L1 框回归权重，不是 YOLO box=7.5 的直接对应数值
        loss_giou=2.0,  # GIoU 定位约束，与 L1 联合优化，暂不盲目放大垃圾桶框损失
        loss_fgl=0.15,  # 官方细粒度定位损失，与 YOLO DFL 不同
        loss_ddf=1.5,  # 官方模型内部定位自蒸馏项，不额外加载老师模型或集成模型

        # 七、随训练验证与权重留存
        conf=0.001,  # 与预测一致，保留低分候选计算 AP，不作为图片展示阈值
        max_det=100,  # 赛事每图最多 100 框；原生 D-FINE 查询排序，不使用 YOLO NMS
        save_period=5,  # 每 5 轮留完整检查点，减少错过中段泛化较好权重的风险
        patience=10,  # mAP50-95 连续 10 轮无有效提升时早停，停止状态随断点保存
        min_delta=0.0005,  # 累计超过 0.05 个百分点才重置耐心；任何真实新高仍保存 best
        domain_metrics=True,  # 复用轮末预测记录 PNG/JPG 总体及逐类指标，不额外模型前向
    ))
