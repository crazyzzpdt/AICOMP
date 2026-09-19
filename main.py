"""训练 YOLO26l RGB、红外、深度门控融合模型，保留1920×1080原图细节。

官方RGB预训练主干与两个轻量辅助分支共同优化，固定内容窗口1920×1080，
网络画布补齐为1920×1088，训练验证及预测共用预处理。数据沿用清洗版1709/291。

开始正式训练：
    uv run python main.py
训练完成后生成赛事提交包：
    uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_yolo26l_1920x1080_v9_fusion/weights/best.pt --imgsz 1920 --height 1080 --output predict_v9
中断恢复：
    将RESUME_PATH设为本配方尚未完成的weights/last.pt，再运行本文件。
    恢复另建目录；配方、源码、数据审计必须一致。

本入口只在用户运行时训练，不自动探测显存；说明见docs/v9融合训练实施.md。
"""

# 内置库
import os
from functools import partial
from pathlib import Path

# 必须在导入Ultralytics前设置，训练不下载数据、安装依赖或上传遥测。
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# 三方库
from ultralytics import YOLO

# 自己的模块
from aic.training import FusionDetectionTrainer, FusionRecipe


# RGB主干和可迁移检测层来自官方COCO预训练，新增分支单独学习。
MODEL_PATH: str = "./orgin_models/yolo26l.pt"
# 沿用已审计的官方新版标签与1709/291划分，不重新生成图像副本。
DATA_PATH: str = "./datasets/data.yaml"
# 开训核对图像清单、标签指纹和清洗来源，不自动改标签。
DATA_AUDIT: str = "./runs/dataset_cleaning/official_refresh_20260918_214843/manifest.json"
# 新结构与新尺寸单独命名，不覆盖原D-FINE v8的成绩与文件。
PROJECT_PATH: str = "./runs/detect"
RUN_NAME: str = "AIC_RGBIRDepth_yolo26l_1920x1080_v9_fusion"
# 顺序为高、宽；原1920×1080不缩小，统一预处理仅补齐步长。
IMAGE_HW: tuple[int, int] = (1920, 1080)
# 120轮余弦收敛，第81轮完整画面收尾；不以5000轮间接控制双头损失。
MAX_EPOCHS: int = 120
POLISH_EPOCHS: int = 40
# 首训保持None，仅恢复相同配方中断状态，已完成模型拒绝恢复。
RESUME_PATH: str | None = None


# Windows DataLoader子进程会重新导入脚本，正式训练必须放在入口保护内。
if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)

    # 融合扩展配置：原生YOLO只接受单个imgsz，矩形高宽由数据组件落实。
    trainer = partial(FusionDetectionTrainer, recipe=FusionRecipe(
        image_height=IMAGE_HW[1],  # 内容高1080，张量仅补8像素，不拉伸目标
        image_width=IMAGE_HW[0],  # 内容宽1920，与原生imgsz长边设置一致
        backbone_lr=0.00002,  # RGB主干慢学，保护预训练特征
        auxiliary_lr=0.0002,  # 新IR/Depth编码器和门控使用较高学习率
        val_batch=1,  # 轮末FP32验证，减少峰值显存
        min_delta=0.0002,  # 0.02个百分点重置早停耐心，真实AP新高仍留存
        data_audit=DATA_AUDIT,  # 使用最近一次已落位审计
    ))

    # 实例化模型类；新训练从官方RGB基底迁移，不续训旧D-FINE或v4。
    model = YOLO(RESUME_PATH or MODEL_PATH)

    # 训练函数：保留原生YOLO进度条、损失和轮末验证输出。
    model.train(
        # 一、模型、数据与训练时长
        trainer=trainer,  # 门控融合、固定矩形数据及与预测一致的验证器
        data=DATA_PATH,  # 只读取datasets，不修改官方源文件
        project=PROJECT_PATH,  # 正式训练产物根目录
        name=RUN_NAME,  # 重名自动另建目录，预测时填写实际路径
        epochs=MAX_EPOCHS,  # 最多120轮，双头比例固定0.8/0.2
        resume=RESUME_PATH or False,  # 仅恢复同配方中断检查点
        pretrained=True,  # 迁移RGB和可对应的检测层，新增分支随机初始化
        cls_remap=True,  # sports ball→ball=7，开训核对两头参数
        exist_ok=False,  # 不覆盖历史运行

        # 二、设备、加载与资源：由用户调整，不自动试跑探测显存
        imgsz=IMAGE_HW[1],  # 元数据长边1920，实际每批固定5×1088×1920
        batch=2,  # 新分支及全尺寸未实测显存，16GB单卡起点；可手动改4
        nbs=16,  # 稳态有效批次16，batch=2时累积8批，预热沿用原生策略
        workers=4,  # 每进程预取1批，关闭锁页，降低CPU内存峰值
        device=0,  # 本机RTX 5080，不调整其他进程资源
        amp="bf16",  # 训练BF16；轮末验证与预测统一FP32
        cache=False,  # 不新增大体积融合NPY缓存
        rect=False,  # 固定矩形由自定义数据集实现，不启用原生宽高比分批
        multi_scale=0.0,  # 固定内容窗口，避免动态变成正方形输入
        compile=False,  # 不增加编译预热与显存探测
        channels_last=False,  # 保持NCHW布局

        # 三、优化器、学习率与收敛
        optimizer="AdamW",  # 沿用v4优化器，不用auto切换
        lr0=0.0001,  # 颈部/检测头；主干和辅助分支见FusionRecipe
        lrf=0.01,  # 120轮余弦末端为各组初始学习率的1%
        cos_lr=True,  # 各组保持学习率比例
        momentum=0.9,  # AdamW第一动量系数
        weight_decay=0.0005,  # 沿用v4量级，偏置和归一化参数不衰减
        warmup_epochs=5.0,  # 新模态分支温和预热，不冻结整个主干30轮
        warmup_momentum=0.8,  # 保留框架兼容配置，AdamW不使用SGD动量组
        warmup_bias_lr=0.0,  # 偏置不以高学习率跳启
        freeze=None,  # RGB低学习率微调，各模态分支均可学习

        # 四、同步增强与完整画面收尾
        mosaic=0.25,  # 同步矩形四图拼接，不构建3840平方中间画布
        close_mosaic=POLISH_EPOCHS,  # 第81轮关闭拼图、尺度与位移，至少收尾20轮
        scale=0.2,  # 主训练阶段温和同步尺度扰动
        translate=0.05,  # 三模态及所有类别框同步平移
        fliplr=0.5,  # 训练和收尾均保留同步水平翻转
        flipud=0.0,  # 城市场景不做上下翻转
        hsv_h=0.01,  # HSV仅作用于RGB
        hsv_s=0.15,  # 温和RGB饱和度变化
        hsv_v=0.15,  # 温和RGB亮度变化
        degrees=0.0,  # 不增加旋转
        shear=0.0,  # 不增加剪切形变
        perspective=0.0,  # 不增加透视形变
        bgr=0.0,  # 保持RGB3+IR1+Depth1通道顺序
        mixup=0.0,  # 不叠加样本混合
        cutmix=0.0,  # 不叠加额外裁剪粘贴
        copy_paste=0.0,  # 本轮不复制目标或定向重采样

        # 五、类别与损失
        box=7.5,  # 沿用YOLO定位损失量级，不移植D-FINE权重
        cls=0.5,  # 原生分类损失，不人为抬高某类预测分数
        cls_pw=0.0,  # 不叠加类别加权，避免稀有类所有负项放大
        dfl=1.5,  # 兼容配置，本机YOLO26使用DFL-free定位路径
        single_cls=False,  # 保留赛事12类
        classes=None,  # 不过滤ball等弱类
        fraction=1.0,  # 使用固定划分的全部训练图

        # 六、随训练验证与权重留存
        val=True,  # 正式轮末验证用于AP95择优与早停
        conf=0.001,  # 与提交一致，保留低分候选计算AP
        iou=0.7,  # 验证与预测共享单标签NMS
        nms=True,  # 正式输出使用一对多分支，不融合两头预测
        max_det=100,  # 赛事单图最多100框
        patience=20,  # 收尾阶段计数，第100轮结束起才允许早停
        save=True,  # 保留best、best_map50及last
        save_period=20,  # 减少周期检查点，仍逐轮保存last
        plots=True,  # 保留原生曲线、混淆矩阵和样本可视化

        # 七、复现与日志
        seed=0,  # 固定采样和增强种子
        deterministic=True,  # 不保证不同硬件逐位一致
        verbose=True,  # 显示迁移、训练损失与验证指标

        # 八、其他任务参数：本轮不使用
        distill_model=None,  # 不引入第二个老师模型或投票集成
        augmentations=None,  # 增强由矩形组件统一执行
        auto_augment=None,  # 检测不使用分类自动增强
        erasing=0.0,  # 不做分类随机擦除
        dropout=0.0,  # 不额外引入dropout
        save_json=False,  # 不输出COCO提交格式
        save_txt=False,  # 正式赛事TXT由predict.py生成
        profile=False,  # 不做额外性能测试
    )
