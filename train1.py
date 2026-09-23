"""执行五通道早期融合 YOLO 训练，联合使用 RGB、红外和深度。

v25在连续FP32基础上加入红外Gamma/局部对比度与RGB曝光扰动，保留1280及1709/291。
五通道早期融合直接使用RGB、红外和深度，训练与正式预测保持同一输入协议。
保留200轮学习率日程，允许五通道模型完整收敛，不用20轮预算提前截断。

由用户启动训练：
    uv run python train1.py
本轮不生成赛事提交包，predict1.py负责YOLO预测。
新输入协议不恢复历史训练；旧权重仅由预测入口保留兼容。

复赛已知v6得49.815；复赛测试集不用于训练、伪标签或修改标签。
v14同协议基线AP95为0.3917876103；没有提升时仍保留v14，不把本地AP换算成赛事分数。
比较前20轮的AP95、AP75及逐类定位，不用首轮0.4门槛判断最终上限。
本入口不自动探测显存；当前方案见docs/浮点三模态实施方案.md。
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
from src.yolo.aic.training import FusionDetectionTrainer, FusionRecipe
from src.modalities import SensorAugment, configure_fp32


# 使用官方RGB预训练权重迁移到五通道首层，不续训历史赛事权重。
MODEL_PATH: str = "./orgin_models/yolo26l.pt"
# 沿用已审计的官方新版标签与1709/291划分，不重新生成图像副本。
DATA_PATH: str = "./datasets/data.yaml"
# 开训核对图像清单、标签指纹和清洗来源，不自动改标签。
DATA_AUDIT: str = "./runs/dataset_cleaning/official_refresh_20260918_214843/manifest.json"
# 由入口位置解析绝对输出目录，避免框架拼接全局runs_dir造成路径重复。
PROJECT_PATH: str = str(Path(__file__).resolve().parent / "runs" / "detect")
RUN_NAME: str = "AIC_RGBIRDepth_yolo26l_1280_v25_illumination"
# 保留v4的1280方形训练增强；当前固定方形验证不等于v4旧矩形验证。
IMAGE_HW: tuple[int, int] = (1280, 1280)
# 保留v14学习率日程，不能把此值改成20来代替预算停止。
MAX_EPOCHS: int = 200
# 五通道模型需覆盖v4约100轮的收敛区间，最多运行200轮。
BUDGET_EPOCHS: int = 200
# 新浮点协议只从官方基底开始，不接入旧训练状态。
RESUME_PATH: str | None = None


# Windows DataLoader子进程会重新导入脚本，正式训练必须放在入口保护内。
if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    configure_fp32()

    # 保留训练器与数据审计；新五通道配方独立，不能恢复旧断点。
    trainer = partial(FusionDetectionTrainer, recipe=FusionRecipe(
        architecture="early_v10",  # RGB3+IR1+Depth1五通道共享骨干，作为正式三模态候选
        continuous_depth=True,  # 原始16位深度直接转FP32，不经过8位取整
        sensors=SensorAugment(
            ir_gamma_probability=0.25,  # 指数0.8–1.2，保留原图训练比例
            ir_local_probability=0.25,  # 浮点局部对比度；增益和改变量受限
            rgb_exposure_probability=0.25,  # 模拟亮度变化，不声称替代真实夜间样本
        ),  # 保留深度有效区扰动；这些随机增强不用于验证/推理
        split_stem=False,  # 不引入v18拆分首层或v19分支
        budget_epochs=BUDGET_EPOCHS,  # 允许完整200轮日程，避免把第20轮误判为最终上限
        training_stage="main",  # 官方基底重新训练，不走v15已有五通道微调路径
        min_stop_epochs=20,  # 保留v14最低耐心停止轮数；阶段筛选独立生效
        initial_weights_sha256=None,  # 无微调父权重，训练器仍记录实际初始化来源
        image_height=IMAGE_HW[1],  # 原生训练画布高度1280
        image_width=IMAGE_HW[0],  # 与下方imgsz一致
        backbone_lr=0.0001,  # 历史v9专用，本轮骨干由early_backbone_lr指定
        auxiliary_lr=0.0001,  # 配方兼容字段，本轮没有辅助分支参数
        early_backbone_lr=0.00001,  # v21骨干2e-5减半，保持骨干与检测头0.2倍比例
        val_batch=1,  # 与v14正式训练轨迹一致，不拿batch2复评替代逐轮对照
        min_delta=0.0,  # 沿用v14，真实AP95新高即可重置耐心
        polish_scale=None,  # 第101轮只关Mosaic，保留v14的scale=0.3
        polish_translate=None,  # 关闭拼图后仍保留translate=0.1，与v14相同
        screening_thresholds=(),  # 本轮仅预算停止，记录budget_history.csv，不按AP硬门槛中断
        geometry="native_square",  # 三模态五通道同步缩放、增强与114补边
        data_audit=DATA_AUDIT,  # 使用最近一次已落位审计
        repeat_threshold=0.0,  # v17未改善总体AP95，关闭受限重复，回到v14基本采样
    ))

    # 官方RGB权重与类别语义迁移；不加载历史赛事权重或新增辅助编码器。
    model = YOLO(RESUME_PATH or MODEL_PATH)

    # 训练函数：保留原生YOLO进度条、损失和轮末验证输出。
    model.train(
        # 一、模型、数据与训练时长
        trainer=trainer,  # 保留审计与原生几何，网络接收五通道，验证协议不变
        model=RESUME_PATH or MODEL_PATH,  # 与上方YOLO实例使用相同基底或恢复权重
        mode="train",  # 显式记录运行模式，model.train也会固定此值
        data=DATA_PATH,  # 只读取datasets，不修改官方源文件
        cfg=None,  # 不使用额外覆盖配置文件，所有参数在此处显式给出
        project=PROJECT_PATH,  # 正式训练产物根目录
        name=RUN_NAME,  # 重名自动另建目录，预测时填写实际路径
        epochs=MAX_EPOCHS,  # 200轮余弦日程，无20轮预算截断
        time=None,  # 不按小时强制截断，训练由epochs和早停控制
        resume=RESUME_PATH or False,  # 仅恢复同配方中断检查点
        pretrained=True,  # 官方COCO预训练迁移，不加载历史赛事权重续训
        cls_remap=True,  # 恢复官方类别语义迁移，包括sports ball到ball
        exist_ok=False,  # 不覆盖历史运行
        save_dir=None,  # 由project/name生成独立目录，不硬编码覆盖路径

        # 二、设备、加载与资源：由用户调整，不自动试跑探测显存
        imgsz=IMAGE_HW[0],  # 保留1280，避免同时更改分辨率影响结构比较
        batch=2,  # 全FP32峰值显存尚未实测；用户可调整，不自动探测
        nbs=16,  # 物理批次2时稳态累积8批；BN批次变化仍影响与v24的比较
        workers=4,  # 每进程预取1批，关闭锁页，降低CPU内存峰值
        device=0,  # 本机RTX 5080，不调整其他进程资源
        amp=False,  # 前向/损失/验证FP32；入口同时关闭TF32
        cache=False,  # 不新增大体积融合NPY缓存
        rect=False,  # 原生方形画布，不使用按批次宽高比分组
        multi_scale=0.0,  # 不额外改变整批尺寸，保留scale几何增强
        compile=False,  # 不增加编译预热与显存探测
        channels_last=False,  # 保持NCHW布局
        fraction=1.0,  # 使用完整审计划分，不作为子集调参开关

        # 三、优化器、学习率与收敛
        optimizer="AdamW",  # 沿用v4优化器，不用auto切换
        lr0=0.00005,  # v21首层/颈部/双头1e-4减半，检验早期更新幅度是否过大
        lrf=0.01,  # 恢复v14的200轮余弦末端为初始学习率1%
        cos_lr=True,  # 不继续v16的60轮快速衰减组合
        momentum=0.9,  # AdamW第一动量系数
        weight_decay=0.0005,  # 沿用v4量级，偏置和归一化参数不衰减
        warmup_epochs=5.0,  # 官方基底重迁移，沿用v14的5轮预热
        warmup_momentum=0.8,  # 保留框架兼容配置，AdamW不使用SGD动量组
        warmup_bias_lr=0.0,  # 偏置不以高学习率跳启
        freeze=None,  # 全部五通道模型参数可训练，不额外冻结骨干

        # 四、同步增强与关闭拼图阶段
        mosaic=0.5,  # 浮点画布：RGB填114、IR/Depth填0，三模态同步几何
        close_mosaic=100,  # 最后100轮关闭：200轮日程第101轮起收尾，与v21计划一致
        scale=0.3,  # 沿用v14同步几何幅度
        translate=0.1,  # 沿用v14平移幅度，关闭拼图后保持不变
        fliplr=0.5,  # 训练和收尾均保留同步水平翻转
        flipud=0.0,  # 城市场景不做上下翻转
        hsv_h=0.0,  # 恢复v4，不在五通道原生增强链中改变颜色
        hsv_s=0.0,  # 不改变RGB饱和度
        hsv_v=0.0,  # 不改变RGB亮度
        degrees=0.0,  # 不增加旋转
        shear=0.0,  # 不增加剪切形变
        perspective=0.0,  # 不增加透视形变
        bgr=0.0,  # 保持RGB、IR、Depth五通道顺序，不进行颜色通道反转
        mixup=0.0,  # 不叠加样本混合
        cutmix=0.0,  # 不叠加额外裁剪粘贴
        copy_paste=0.0,  # 不复制粘贴目标，本轮也关闭受限采样
        copy_paste_mode="flip",  # 保留框架默认值；detect任务不启用copy-paste

        # 五、类别与损失
        box=7.5,  # 沿用YOLO定位损失量级，不移植D-FINE权重
        cls=0.5,  # 原生分类损失，不人为抬高某类预测分数
        cls_pw=0.0,  # 回到v21的无额外类别加权；v22/23没有证明加权改善总体AP
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
        patience=30,  # 连续30轮无AP95新高停止，避免再等待100轮退化阶段
        save=True,  # 保留best、best_map50及last
        save_period=5,  # 每5轮留存，最佳/末轮权重和逐类指标照常保存
        plots=True,  # 保留原生曲线、混淆矩阵和样本可视化
        save_conf=False,  # 不额外写原生预测TXT，赛事文件由predict1.py生成
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
        save_txt=False,  # 正式赛事TXT由predict1.py生成
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
