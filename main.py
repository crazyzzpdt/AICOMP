"""以v14配方执行v18分模态首层学习率实验，按总体AP95判断收益。

RGB、红外、深度从首层共同学习；训练、验证和预测统一使用原生1280方形几何。
保留v14的200轮日程及第101轮关闭Mosaic；关闭v17受限采样，清洗版1709/291不变。

开始正式训练：
    uv run python main.py
训练完成且选定v18后生成复赛预测TXT包：
    uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_yolo26l_1280_v18_split_stem/weights/best.pt --source "数据集/复赛测试集" --imgsz 1280 --height 1280 --output predict_v18_round2
中断恢复：
    将RESUME_PATH设为本配方尚未完成的weights/last.pt，再运行本文件。
    恢复另建目录；配方、源码、数据审计必须一致。

用户已用v4提交复赛，分数未知；复赛测试集不用于训练、伪标签或调参。
v14同协议基线AP95为0.3917876103；没有提升时仍保留v14，不把本地AP换算成赛事分数。
先观察前20轮的AP95与分支诊断；不会因首轮未达0.4而停止，不保证达到0.60。
本入口不自动探测显存；方案见docs/superpowers/plans/2026-09-21-v18-split-stem.md。
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


# 与v14相同的官方RGB基底重新迁移，不续训v15或旧划分v4。
MODEL_PATH: str = "./orgin_models/yolo26l.pt"
# 沿用已审计的官方新版标签与1709/291划分，不重新生成图像副本。
DATA_PATH: str = "./datasets/data.yaml"
# 开训核对图像清单、标签指纹和清洗来源，不自动改标签。
DATA_AUDIT: str = "./runs/dataset_cleaning/official_refresh_20260918_214843/manifest.json"
# 由入口位置解析绝对输出目录，避免框架拼接全局runs_dir造成路径重复。
PROJECT_PATH: str = str(Path(__file__).resolve().parent / "runs" / "detect")
RUN_NAME: str = "AIC_RGBIRDepth_yolo26l_1280_v18_split_stem"
# 保留v4的1280方形训练增强；当前固定方形验证不等于v4旧矩形验证。
IMAGE_HW: tuple[int, int] = (1280, 1280)
# 恢复v14日程，预算由阶段筛选与patience控制，不搬回v4名义5000轮。
MAX_EPOCHS: int = 200
# 沿用v14历史最佳AP95门槛，不因首层新方案放宽预算，也不保证后期门槛可达。
SCREENING_THRESHOLDS: tuple[tuple[int, float], ...] = (
    (10, 0.32),  # 节省明显弱配方预算，不要求首轮达到0.4
    (20, 0.34),  # 同划分v14已超过此值，不作为最终上限
    (50, 0.36),  # 主增强阶段仍未达到基础水平则结束
    (75, 0.38),  # 后续门槛只是预算目标，不是预期学习曲线
    (100, 0.40),  # 关Mosaic前评估累计最佳结果
    (125, 0.45),  # 不达预算目标则保存结果后停止
    (150, 0.50),  # 使用本地AP95，不换算赛事分数
    (175, 0.55),  # 保留v14预算约束，禁止自动重试
    (200, 0.60),  # 最终目标不是达到保证
)
# 首训保持None，仅恢复相同配方中断状态，已完成模型拒绝恢复。
RESUME_PATH: str | None = None


# Windows DataLoader子进程会重新导入脚本，正式训练必须放在入口保护内。
if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)

    # 使用已有训练器，保留数据审计与恢复保护，仅拆分首层优化参数。
    trainer = partial(FusionDetectionTrainer, recipe=FusionRecipe(
        architecture="early_v10",  # 恢复v4首层融合，不把模态置零伪装多模态
        split_stem=True,  # RGB/IR/Depth卷积独立参数组，相加后共用原首层BN与激活
        training_stage="main",  # 官方基底重新训练，不走v15已有五通道微调路径
        min_stop_epochs=20,  # 保留v14最低耐心停止轮数；阶段筛选独立生效
        initial_weights_sha256=None,  # 无微调父权重，训练器仍记录实际初始化来源
        image_height=IMAGE_HW[1],  # 原生训练画布高度1280
        image_width=IMAGE_HW[0],  # 与下方imgsz一致
        backbone_lr=0.0001,  # 历史门控专用；当前骨干由early_backbone_lr指定
        auxiliary_lr=0.0005,  # v18新增IR/Depth首层真实AdamW学习率，5倍主学习率为待验证候选
        early_backbone_lr=0.00002,  # 沿用v14有本地改善证据的主学习率0.2倍
        val_batch=2,  # FP32验证与v14独立复评采用同样批次大小
        min_delta=0.0,  # 沿用v14，真实AP95新高即可重置耐心
        polish_scale=None,  # 第101轮只关Mosaic，保留v14的scale=0.3
        polish_translate=None,  # 关闭拼图后仍保留translate=0.1，与v14相同
        screening_thresholds=SCREENING_THRESHOLDS,  # 检查历史最佳AP95，结果写screening_history.csv
        geometry="native_square",  # 复用v4原生LetterBox、Mosaic和透视增强链
        data_audit=DATA_AUDIT,  # 使用最近一次已落位审计
        repeat_threshold=0.0,  # v17未改善总体AP95，关闭受限重复，回到v14基本采样
    ))

    # 从官方模型迁移RGB；IR/Depth新增卷积零初始化后参与学习。
    model = YOLO(RESUME_PATH or MODEL_PATH)

    # 训练函数：保留原生YOLO进度条、损失和轮末验证输出。
    model.train(
        # 一、模型、数据与训练时长
        trainer=trainer,  # 五通道原生增强、审计校验、分层学习率及单标签验证器
        model=RESUME_PATH or MODEL_PATH,  # 与上方YOLO实例使用相同基底或恢复权重
        mode="train",  # 显式记录运行模式，model.train也会固定此值
        data=DATA_PATH,  # 只读取datasets，不修改官方源文件
        cfg=None,  # 不使用额外覆盖配置文件，所有参数在此处显式给出
        project=PROJECT_PATH,  # 正式训练产物根目录
        name=RUN_NAME,  # 重名自动另建目录，预测时填写实际路径
        epochs=MAX_EPOCHS,  # 最多200轮，沿用v14日程和固定双头0.8/0.2
        time=None,  # 不按小时强制截断，训练由epochs和早停控制
        resume=RESUME_PATH or False,  # 仅恢复同配方中断检查点
        pretrained=True,  # 官方COCO预训练迁移，不加载历史赛事权重续训
        cls_remap=True,  # 恢复官方类别语义迁移，包括sports ball到ball
        exist_ok=False,  # 不覆盖历史运行
        save_dir=None,  # 由project/name生成独立目录，不硬编码覆盖路径

        # 二、设备、加载与资源：由用户调整，不自动试跑探测显存
        imgsz=IMAGE_HW[0],  # 恢复v4的1280方形训练尺度
        batch=4,  # v4已在本机验证该分辨率与物理批次可运行
        nbs=16,  # 稳态有效批次16，batch=4时累积4批，预热沿用原生策略
        workers=4,  # 每进程预取1批，关闭锁页，降低CPU内存峰值
        device=0,  # 本机RTX 5080，不调整其他进程资源
        amp="bf16",  # 训练BF16；轮末验证与预测统一FP32
        cache=False,  # 不新增大体积融合NPY缓存
        rect=False,  # 原生方形画布，不使用按批次宽高比分组
        multi_scale=0.0,  # 不额外改变整批尺寸，保留scale几何增强
        compile=False,  # 不增加编译预热与显存探测
        channels_last=False,  # 保持NCHW布局
        fraction=1.0,  # 使用完整审计划分，不作为子集调参开关

        # 三、优化器、学习率与收敛
        optimizer="AdamW",  # 沿用v4优化器，不用auto切换
        lr0=0.0001,  # RGB首层/颈部/双头；骨干2e-5，IR/Depth首层5e-4
        lrf=0.01,  # 恢复v14的200轮余弦末端为初始学习率1%
        cos_lr=True,  # 不继续v16的60轮快速衰减组合
        momentum=0.9,  # AdamW第一动量系数
        weight_decay=0.0005,  # 沿用v4量级，偏置和归一化参数不衰减
        warmup_epochs=5.0,  # 官方基底重迁移，沿用v14的5轮预热
        warmup_momentum=0.8,  # 保留框架兼容配置，AdamW不使用SGD动量组
        warmup_bias_lr=0.0,  # 偏置不以高学习率跳启
        freeze=None,  # 五通道首层与预训练主干全部可训练

        # 四、同步增强与关闭拼图阶段
        mosaic=0.5,  # 沿用v4/v14，不与新首层学习率同时更换增强配方
        close_mosaic=100,  # 最后100轮关闭，即第101轮起不拼图，与v4/v14时间点一致
        scale=0.3,  # 沿用v14，不在拆分首层时同时调整增强幅度
        translate=0.1,  # 沿用v14平移幅度，关闭拼图后保持不变
        fliplr=0.5,  # 训练和收尾均保留同步水平翻转
        flipud=0.0,  # 城市场景不做上下翻转
        hsv_h=0.0,  # 恢复v4，不在五通道原生增强链中改变颜色
        hsv_s=0.0,  # 不改变RGB饱和度
        hsv_v=0.0,  # 不改变RGB亮度
        degrees=0.0,  # 不增加旋转
        shear=0.0,  # 不增加剪切形变
        perspective=0.0,  # 不增加透视形变
        bgr=0.0,  # 保持RGB3+IR1+Depth1通道顺序
        mixup=0.0,  # 不叠加样本混合
        cutmix=0.0,  # 不叠加额外裁剪粘贴
        copy_paste=0.0,  # 不复制粘贴目标，本轮也关闭受限采样
        copy_paste_mode="flip",  # 保留框架默认值；detect任务不启用copy-paste

        # 五、类别与损失
        box=7.5,  # 沿用YOLO定位损失量级，不移植D-FINE权重
        cls=0.5,  # 原生分类损失，不人为抬高某类预测分数
        cls_pw=0.0,  # 沿用v14，不额外叠加类别频率加权
        dfl=1.5,  # YOLO26无传统DFL，此值实际加权归一化框距离的L1损失
        single_cls=False,  # 保留赛事12类
        classes=None,  # 不过滤ball等弱类
        agnostic_nms=False,  # 不跨类别合并框，避免相邻类别互相抑制
        angle=1.0,  # OBB专用参数；普通detect任务不生效
        pose=12.0,  # 姿态专用损失权重；普通detect任务不生效
        kobj=1.0,  # 姿态专用目标性权重；普通detect任务不生效
        rle=1.0,  # 姿态专用残差似然权重；普通detect任务不生效
        dlog=1.0,  # 深度专用SILog权重；普通detect任务不生效
        dgrad=0.5,  # 深度专用梯度权重；普通detect任务不生效
        dlam=1.0,  # 深度专用尺度因子；普通detect任务不生效
        mask_ratio=4,  # 分割任务掩码下采样值；普通detect任务不生效
        overlap_mask=True,  # 分割任务掩码重叠策略；普通detect任务不生效

        # 六、随训练验证与权重留存
        val=True,  # 正式轮末验证用于AP95择优与早停
        conf=0.001,  # 与提交一致，保留低分候选计算AP
        iou=0.7,  # 验证与预测共享单标签NMS
        nms=True,  # 正式输出使用一对多分支，不融合两头预测
        max_det=100,  # 赛事单图最多100框
        patience=40,  # 沿用v14耐心，无新高不因200轮上限强行跑满
        save=True,  # 保留best、best_map50及last
        save_period=5,  # 保留中段检查点与首层诊断，仍逐轮保存last
        plots=True,  # 保留原生曲线、混淆矩阵和样本可视化
        save_conf=False,  # 不额外写原生预测TXT，赛事文件由predict.py生成
        save_crop=False,  # 不保存裁剪目标，避免产生冗余磁盘占用
        save_frames=False,  # 非视频任务，不保存帧图像
        line_width=None,  # 使用框架默认绘图线宽

        # 七、复现与日志
        seed=0,  # 固定采样和增强种子
        deterministic=True,  # 不保证不同硬件逐位一致
        verbose=True,  # 显示迁移、训练损失与验证指标

        # 八、其他任务参数：本轮不使用
        distill_model=None,  # 不引入第二个老师模型或投票集成
        dis=6.0,  # 蒸馏损失默认权重；distill_model=None时不参与训练
        augmentations=None,  # 沿用v13原生增强，本轮不叠加额外Albumentations配方
        auto_augment=None,  # 检测不使用分类自动增强
        erasing=0.0,  # 不做分类随机擦除
        dropout=0.0,  # 不额外引入dropout
        augment=False,  # 训练阶段不启用验证时增强推理
        embed=None,  # 不导出中间层嵌入
        show=False,  # 不弹出窗口
        show_labels=True,  # 保留验证图标签显示
        show_conf=True,  # 保留验证图置信度显示
        show_boxes=True,  # 保留验证图框显示
        save_json=False,  # 不输出COCO提交格式
        save_txt=False,  # 正式赛事TXT由predict.py生成
        dnn=False,  # 不使用OpenCV DNN推理后端
        format="torchscript",  # 仅导出任务使用；训练阶段不执行导出
        keras=False,  # 不使用TensorFlow/Keras导出
        optimize=False,  # 不执行移动端导出优化
        dynamic=False,  # 不导出动态输入图
        simplify=True,  # 导出时默认简化图；训练阶段不执行导出
        opset=None,  # 不指定导出ONNX算子集
        workspace=None,  # 不为TensorRT导出预留工作区
        retina_masks=False,  # 非分割任务
        quantize=None,  # 不进行量化感知训练
        task="detect",  # 固定为目标检测任务
        source=None,  # 训练不从推理源读取图像
        split="val",  # 轮末验证使用data.yaml中的val划分
        tracker="tracktrack.yaml",  # 仅跟踪任务使用，训练阶段不生效
        vid_stride=1,  # 非视频任务，保留逐帧默认值
        stream_buffer=False,  # 非视频流任务，不缓存流帧
        visualize=False,  # 不进行特征可视化
        profile=False,  # 不做额外性能测试
    )
