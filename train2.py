"""D-FINE-X v28：官方RGB骨干、局部质量门控与多模态细节融合。

运行：uv run python train2.py
读取datasets官方原标签副本，不修改图像和标签；历史权重仅保留预测，不恢复旧训练。
"""

# 内置库
import os
import sys
from pathlib import Path

# 必须在导入训练依赖前关闭下载与遥测。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
PROJECT_ROOT: Path = Path(__file__).resolve().parent
# 项目src优先；官方src路径由共用构建器扩展，避免遮蔽项目包。
sys.path.insert(0, str(PROJECT_ROOT))

# 自己的模块
from src.dfine.training import TrainingConfig, train
from src.dfine.reliability import RELIABILITY_ARCHITECTURE
from src.modalities import SensorAugment
from src.augmentation import SmallObjectCrop


# Windows子进程只导入配置和类，不重复启动训练。
if __name__ == "__main__":
    train(TrainingConfig(
        # 一、模型、数据与训练时长
        model="orgin_models/dfine_x_obj365.pth",  # 本地官方366输出槽Objects365基底
        variant="x",  # 官方B5骨干、384维编码器与X解码器，由检查点传给预测
        architecture=RELIABILITY_ARCHITECTURE,  # RGB独立骨干＋IR/Depth辅助分支，P3–P5残差融合
        data="datasets/data.yaml",  # 使用1900/100官方原始标签副本，不做标签清洗
        project="runs/detect",  # 与历史运行并存
        name="AIC_RGBIRDepth_dfine_x_1536_v28_reliability",  # 用户指定v28；原v29草案未开训，不覆盖v27
        epochs=60,  # 固定短日程，结合轮末AP95早停
        resume=None,  # 不接收旧优化器或旧预处理断点
        data_audit="runs/dataset_cleaning/official_labels_split_1900_100_v27/manifest.json",

        # 二、设备、精度与加载：不自动探测资源
        imgsz=1536,  # 与YOLO候选使用相同输入尺寸，等比缩放后填充
        batch=1,  # X模型全FP32从单张起步，未实测本机峰值显存
        effective_batch=16,  # 16批累积，尾组按实际样本数归一化
        val_batch=1,  # 验证与预测均FP32，无额外独立复评
        workers=2,  # 每进程预取1批，关闭锁页和完整数据缓存
        device=0,  # 本地CUDA设备；适配器关闭AMP与TF32
        seed=0,  # 固定种子，不宣称跨设备逐位一致

        # 三、优化器和收敛
        lr0=0.00005,  # 官方编码器和解码器学习率，保留D-FINE v27设置
        backbone_lr=0.00001,  # 完整RGB骨干含首层均受保护，不再扩展RGB首层
        auxiliary_lr=0.0001,  # 新增辅助分支、质量门控和细节投影单独学习
        lrf=0.01,  # 余弦末端比例
        warmup_epochs=5,  # 新类别与新增模态短预热
        weight_decay=0.0001,  # 偏置及一维参数不衰减
        clip_grad=0.1,  # 非有限梯度报错
        ema_decay=0.999,  # 同一训练轨迹EMA，不做多模型集成
        ema_warmup=100,  # 按真实优化步计数

        # 四、同步几何与传感器增强
        polish_epoch=40,  # 第41轮关闭尺度扰动；本轮不裁剪，温和传感器扰动保留
        scale_min=0.9,  # 只缩小后填充，不裁掉目标
        fliplr=0.5,  # 三模态与标签同步翻转
        hsv_h=0.0,  # 不叠加HSV，保留基础传感器增强
        hsv_s=0.0,  # 不改变RGB饱和度
        hsv_v=0.0,  # 不改变RGB亮度
        sensors=SensorAugment(
            ir_gamma_probability=0.0,  # 关闭额外Gamma，与YOLO候选一致
            ir_local_probability=0.0,  # 关闭额外局部对比度
            rgb_exposure_probability=0.0,  # 关闭额外曝光扰动
        ),  # 保留IR传感器噪声与深度扰动；验证预测不随机增强
        small_crop=SmallObjectCrop(
            probability=0.0,  # 关闭裁剪，先验证v28结构，不叠加YOLO v29的裁剪变量
            min_fraction=0.6,  # 以下窗口设置仅留作配方记录，概率为0时不使用
            max_fraction=0.8,  # 未启用的裁剪窗口上限
            max_object_size=64.0,  # 未启用的小目标阈值
            min_visibility=0.8,  # 未启用的截断保护
            attempts=6,  # 未启用的窗口尝试上限
        ),  # 训练、验证与预测均读取完整三模态原图，不改官方文件

        # 五、D-FINE损失与权重留存
        loss_vfl=1.0,  # 官方分类权重，不搬YOLO的cls
        loss_bbox=5.0,  # 原生L1框损失
        loss_giou=2.0,  # 原生GIoU
        loss_fgl=0.15,  # 原生细粒度定位
        loss_ddf=1.5,  # 原生分布蒸馏，不是外部老师模型
        conf=0.001,  # AP低分候选，非展示阈值
        max_det=100,  # 赛事每图上限，查询排序不使用YOLO NMS
        patience=50,  # 轮末AP95无有效提升则停止
        min_delta=0.0,  # 真实AP95新高即刷新耐心，保留best
        save_period=5,  # FP32检查点，不转成FP16
        domain_metrics=True,  # 复用同次验证计算PNG/JPG分域指标
    ))
