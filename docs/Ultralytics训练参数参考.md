# Ultralytics 训练参数参考

2026-09-18 适用说明：本文为Ultralytics默认参数与历史YOLO项目覆盖记录，不是当前main.py配置。D-FINE v8使用独立TrainingConfig、官方原生损失和optimizer_step日程，不能把YOLO的box/cls/dfl或Mosaic参数当成等价设置。当前值见 [v8方案](v8训练方案与数据清洗.md)，通用知识见 [知识索引](项目知识索引.md)。下方v4/v5“当前”均限于历史阶段。

v5_full 当前覆盖：lr0=0.0001（骨干 0.2 倍）、cls_pw=0、mosaic=0.25、scale=0.2、translate=0.05、close_mosaic=4840、patience=200、nms=True、save_period=50；弱类采样、原尺寸同步裁剪与 RGB 独立增强属于训练优化.py 的项目实现，不是框架新增参数。第 161 轮同时关闭 Mosaic、目标裁剪、辅助模态缺失。完整配置见 [v5方案](v5训练方案与球类诊断.md)，下方 v4 说明为历史记录。

项目说明：下表保留抓取时的框架默认值，不是当前训练配置。v4 的实际值见 main.py 与 [训练配置与数据集复核](训练配置与数据集复核.md)：YOLO26l、五通道、1280、batch=4、workers=4、AdamW、lr0=0.0001、nbs=16、MAX_EPOCHS=5000、patience=100；已使用清洗数据完成训练。

项目自定义训练器在前 200 轮衰减学习率，不能按框架默认的总轮数推算；close_mosaic 是最后 N 轮关闭，入口由 MAX_EPOCHS-MOSAIC_EPOCHS 自动换算（默认 4900），第 101 轮关闭。sports ball → ball 是项目的预训练名称适配，并非新增 Ultralytics 参数。auto_augment、erasing 等分类参数对当前 detect 不生效。

来源（2026-09-15 抓取）：

- 训练模式参数：<https://docs.ultralytics.com/zh/modes/train/>
- 检测任务参数：<https://docs.ultralytics.com/zh/tasks/detect/>

## 一、训练设置（Train Settings）

来自 `model.train()` 全部参数。

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `model` | `str` | `None` | 指定用于训练的模型文件。接受指向 `.pt` 预训练模型或 `.yaml` 配置文件的路径。对于定义模型结构或初始化权重至关重要。 |
| `data` | `str` | `None` | 数据集 YAML 的路径（例如 `coco8.yaml`），其中包含训练和验证数据的路径、类别名称和类别数量。分类任务则使用数据集目录或内置数据集名称（例如 `imagenet10`）。 |
| `epochs` | `int` | `100` | 训练周期总数。每个周期表示完整遍历整个数据集一次。调整此值会影响训练时长和模型性能。 |
| `time` | `float` | `None` | 最大训练时间（小时）。设置后，它会覆盖 `epochs` 参数，使训练在达到指定时长后自动停止。适用于受时间限制的训练场景。 |
| `patience` | `int` | `100` | 在验证指标没有改善的情况下，等待多少个周期后提前停止训练。通过在性能趋于稳定时停止训练，有助于防止过拟合。 |
| `batch` | `int` 或 `float` | `16` | 批次大小有三种模式：设置为整数（例如 `batch=16`）、自动模式（使用 60% 的 GPU 内存，`batch=-1`），或带指定利用率比例的自动模式（`batch=0.70`）。 |
| `imgsz` | `int` | `640` | 训练目标图像尺寸。图像会被调整为边长等于指定值的正方形（如果为 `rect=False`）；对于 YOLO 模型会保持宽高比，但 RT-DETR 不会。会影响模型准确率和计算复杂度。 |
| `save` | `bool` | `True` | 启用训练检查点和最终模型权重的保存。适用于恢复训练或部署模型。 |
| `save_period` | `int` | `-1` | 模型检查点的保存频率，以周期数指定。值为 -1 时禁用此功能。适用于在长时间训练期间保存中间模型。 |
| `cache` | `bool` | `False` | 启用将数据集图像缓存到内存（`True`/`ram`）、磁盘（`disk`），或禁用缓存（`False`）。通过减少磁盘 I/O 提高训练速度，但会增加内存使用量。 |
| `device` | `int` 或 `str` 或 `list` | `None` | 指定训练使用的计算设备：单个 GPU（`device=0`）、多个 GPU（`device=[0,1]`）、CPU（`device=cpu`）、Apple silicon 上的 MPS（`device=mps`）、Huawei Ascend NPU（`device=npu:0` 或 `device=npu:0,1`），或自动选择空闲 GPU（`device=-1`）或多个空闲 GPU（`device=[-1,-1]`）。 |
| `workers` | `int` | `8` | 数据加载的工作线程数（Multi-GPU 训练时为每个 `RANK`）。会影响数据预处理和输入模型的速度，在多 GPU 配置中尤其有用。 |
| `project` | `str` | `None` | 保存训练输出的项目目录名称。便于有序存储不同实验。 |
| `name` | `str` | `None` | 训练运行名称。用于在项目文件夹中创建子目录，以存储训练日志和输出。 |
| `exist_ok` | `bool` | `False` | 如果为 True，则允许覆盖现有的项目/名称目录。便于迭代实验，无需手动清除之前的输出。 |
| `save_dir` | `str` | `None` | 指定保存运行输出的确切目录，覆盖 `project`/`name` 组合。路径将按原样使用，不会自动递增，因此连续运行会重复使用同一目录。 |
| `pretrained` | `bool` 或 `str` | `True` | 决定是否从预训练权重开始训练。可以是布尔值，也可以是要加载的权重文件路径。`pretrained=False` 会使用随机初始化的权重进行训练，同时保留模型架构。 |
| `cls_remap` | `bool` | `True` | 跨数据集微调时，会将预训练分类头中与新模型类别名称匹配的行复制到新模型中，因此重叠类别会保留已学习的偏置；当分类头宽度不变时，还会保留其权重。无论类别数量不同，还是数量相同但类别顺序不同，均适用。 |
| `optimizer` | `str` | `'auto'` | 训练所使用的优化器。选项包括 `SGD`、`MuSGD`、`Adam`、`Adamax`、`AdamW`、`NAdam`、`RAdam`、`RMSProp`，或 `auto`，根据训练迭代次数选择 `AdamW` 或 `MuSGD`。会影响收敛速度和稳定性。 |
| `seed` | `int` | `0` | 设置训练的随机种子，确保使用相同配置运行时结果具有可复现性。 |
| `deterministic` | `bool` | `True` | 强制使用确定性算法，确保结果可复现，但由于限制使用非确定性算法，可能会影响性能和速度。 |
| `verbose` | `bool` | `True` | 启用训练期间的详细输出，在控制台中显示进度条、每周期指标和其他训练信息。 |
| `single_cls` | `bool` | `False` | 训练期间将多类别数据集中的所有类别视为单一类别。适用于二分类任务，或关注对象是否存在而非其类别的场景。 |
| `classes` | `list[int]` | `None` | 指定要训练的类别 ID 列表。适用于过滤掉某些类别，仅关注指定类别。 |
| `rect` | `bool` | `False` | 启用最小填充策略——批次中的图像只进行最小程度的填充以达到统一尺寸，最长边等于 `imgsz`。可以提高效率和速度，但可能影响模型准确率。 |
| `multi_scale` | `float` | `0.0` | 每个批次随机改变 `imgsz`，变化范围为 +/- `multi_scale`（例如从 `0.25` -> `0.75x` 到 `1.25x`），并四舍五入到模型步幅的倍数；`0.0` 会禁用多尺度训练。 |
| `cos_lr` | `bool` | `False` | 使用余弦学习率调度器，在各个周期内按照余弦曲线调整学习率。有助于管理学习率，从而实现更好的收敛。 |
| `close_mosaic` | `int` | `10` | 在最后 N 个周期中禁用马赛克数据增强，以便在训练结束前稳定训练。设置为 0 时禁用此功能。 |
| `resume` | `bool` | `False` | 从最后保存的检查点恢复训练。自动加载模型权重、优化器状态和周期计数，无缝继续训练。 |
| `amp` | `bool` 或 `str` | `True` | 设置训练精度：`True` 或 `"fp16"` 使用 FP16，`"bf16"` 在受支持的 CUDA 设备上使用 BF16，`False` 或 `"fp32"` 使用 FP32。 |
| `quantize` | `int` 或 `str` | `None` | 设为 `8`（或 `"int8"`）以进行 INT8 量化感知训练（QAT），该训练在循环中使用伪量化进行微调，从而使权重能够承受 INT8 导出。 |
| `fraction` | `float`、`int` 或 `list` | `1.0` | 数据集子集，可指定比例/数量或 `[train, val, test]` 列表。`1` 表示完整拆分，超过 `1` 的整数表示图像数量，只有可选的测试项接受 `0`/`0.0` 来表示无数据。包含两个项目的列表会保留完整测试集。 |
| `profile` | `bool` | `False` | 启用训练期间对 ONNX 和 TensorRT 速度进行分析，有助于优化模型部署。 |
| `freeze` | `int` 或 `list` | `None` | 冻结模型的前 N 层，或按索引或模块名称指定的层（`23.cv2`，不含开头的 `model.`），从而减少可训练参数数量。适用于微调或迁移学习。 |
| `lr0` | `float` | `0.01` | 初始学习率（即 `SGD=1E-2`、`Adam=1E-3`）。调整此值对优化过程至关重要，会影响模型权重更新的速度。 |
| `lrf` | `float` | `0.01` | 最终学习率占初始学习率的比例 =（`lr0 * lrf`），与调度器结合使用，以随时间调整学习率。 |
| `momentum` | `float` | `0.937` | SGD 的动量因子或 Adam 优化器的 beta1，影响当前更新中对历史梯度的整合。 |
| `weight_decay` | `float` | `0.0005` | L2 正则化项，通过惩罚较大的权重来防止过拟合。 |
| `warmup_epochs` | `float` | `3.0` | 学习率预热周期数，将学习率从较低值逐步提高到初始学习率，以在训练早期稳定训练。 |
| `warmup_momentum` | `float` | `0.8` | 预热阶段的初始动量，在预热期间逐步调整到设定的动量。 |
| `warmup_bias_lr` | `float` | `0.1` | 预热阶段偏置参数的学习率，有助于稳定最初几个周期的模型训练。在默认 `optimizer='auto'` 下会自动设置为 `0.0`，因此需要显式指定优化器才能使用它。 |
| `distill_model` | `str` | `None` | 用于知识蒸馏的教师模型检查点路径（例如 `yolo26x.pt`）。设置后，学生模型会在冻结教师模型的指导下，通过额外的蒸馏损失进行训练。 |
| `dis` | `float` | `6.0` | 添加到标准检测损失中的蒸馏损失权重。值越大，教师模型特征指导的影响越大。 |
| `box` | `float` | `7.5` | 损失函数中框损失分量的权重，影响模型对准确预测边界框坐标的重视程度。 |
| `cls` | `float` | `0.5` | 总损失函数中的分类损失权重，影响正确类别预测相对于其他分量的重要性。 |
| `cls_pw` | `float` | `0.0` | 使用类别频率倒数处理类别不平衡时的类别加权幂。`0.0` 禁用类别加权，`1.0` 应用完整的频率倒数加权。0 到 1 之间的值表示部分加权。 |
| `dfl` | `float` | `1.5` | 框距离回归项的权重：当检测头使用 `reg_max > 1` 时为分布式焦点损失（DFL），在无 DFL 的 YOLO26（`reg_max: 1`）中为对归一化框距离计算的 L1 损失。 |
| `pose` | `float` | `12.0` | 姿态估计模型中姿态损失的权重，影响模型对准确预测姿态关键点的重视程度。 |
| `kobj` | `float` | `1.0` | 姿态估计模型中关键点目标性损失的权重，用于平衡检测置信度与姿态准确率。 |
| `rle` | `float` | `1.0` | 姿态估计模型中残差对数似然估计损失的权重，影响关键点定位的精度。 |
| `angle` | `float` | `1.0` | obb 模型中角度损失的权重，影响定向边界框角度预测的精度。 |
| `dlog` | `float` | `1.0` | 深度估计模型中尺度不变对数（SILog）损失的权重，这是驱动深度准确率的主要项。 |
| `dgrad` | `float` | `0.5` | 深度估计模型中梯度损失的权重，用于惩罚深度边缘误差并促使表面边界更加清晰。 |
| `dlam` | `float` | `1.0` | 深度估计模型中 SILog 损失的方差关注因子。`1.0` 使损失完全具有尺度不变性，而 `0.0` 会将其简化为普通的对数均方根误差。 |
| `nbs` | `int` | `64` | 用于归一化损失的名义批次大小。 |
| `overlap_mask` | `bool` | `True` | 决定是否将对象掩码合并为单个掩码进行训练，或为每个对象分别保留掩码。发生重叠时，合并过程中较小的掩码会叠加在较大的掩码上方。 |
| `mask_ratio` | `int` | `4` | 分割掩码的下采样比例，影响训练期间使用的掩码分辨率。 |
| `dropout` | `float` | `0.0` | 分类任务中的正则化丢弃率，通过在训练期间随机忽略单元来防止过拟合。 |
| `val` | `bool` | `True` | 启用训练期间的验证，从而可以定期在独立数据集上评估模型性能。 |
| `nms` | `bool`，可选 | `None` | 选择用于轮次验证、检查点选择和早停的推理头。`None` 或 `True` 使用带 NMS 的一对多（one-to-many）；`False` 在可用时使用无 NMS（NMS-free）的头。两个头都保留其训练损失。 |
| `plots` | `bool` | `True` | 生成并保存训练和验证指标图，以及预测示例，为模型性能和学习进展提供可视化洞察。 |
| `compile` | `bool` 或 `str` | `False` | 使用 `backend='inductor'` 启用 PyTorch 2.x 的 `torch.compile` 图编译。接受 `True` → `"default"`、`False` → 禁用，或使用 `"default"`、`"reduce-overhead"`、`"max-autotune-no-cudagraphs"` 等字符串模式。不支持时会发出警告并回退到 eager 模式。 |
| `channels_last` | `bool` | `None` | 在训练期间为卷积使用 channels_last (NHWC) 内存格式。`None` 会在配备 PyTorch 1.11 或更新版本的 CUDA 上自动启用它，但在测得运行较慢的 Windows 系统上除外。`False` 会禁用它，而 `True` 会显式请求它。PyTorch 1.10 及更旧版本、CPU 和 MPS 默认保持 NCHW 格式。 |
| `max_det` | `int` | `300` | 训练验证期间每张图像的最大检测数。对于 detect、segment、pose 和 OBB，默认的 300 只有在训练集/验证集标注对象最大数量超过 300 时才会增加到该数量。其他值保持固定；超出限制会触发警告。 |

## 二、增强设置与超参数（Augmentation Settings）

| 参数 | 类型 | 默认值 | 支持的任务 | 范围 | 描述 |
|---|---|---|---|---|---|
| `hsv_h` | `float` | `0.015` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0.0 - 1.0` | 按色轮的一定比例调整图像色调，引入颜色变化。帮助模型适应不同的光照条件。对于 `classify`，仅当 `auto_augment=None` 时应用。 |
| `hsv_s` | `float` | `0.7` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0.0 - 1.0` | 按一定比例调整图像饱和度，影响颜色的强度。适用于模拟不同的环境条件。对于 `classify`，仅当 `auto_augment=None` 时应用。 |
| `hsv_v` | `float` | `0.4` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0.0 - 1.0` | 按一定比例修改图像的明度（亮度），帮助模型在各种光照条件下保持良好表现。对于 `classify`，仅当 `auto_augment=None` 时应用。 |
| `degrees` | `float` | `0` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | `0.0 - 180` | 在指定的角度范围内随机旋转图像，提升模型识别不同方向对象的能力。 |
| `translate` | `float` | `0.1` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | `0.0 - 1.0` | 按图像尺寸的一定比例水平和垂直平移图像，帮助模型学习检测部分可见的对象。 |
| `scale` | `float \| tuple` | `0.5` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0 - 1`，或显式的 `(min, max)` 元组（不适用于 `classify`） | 按照增益因子缩放图像，模拟距离摄像头远近不同的对象。 |
| `shear` | `float` | `0` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | `-180 - +180` | 按指定角度剪切图像，模拟从不同角度观察对象的效果。 |
| `perspective` | `float` | `0` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | `0.0 - 0.001` | 对图像应用随机透视变换，增强模型理解三维空间中对象的能力。 |
| `flipud` | `float` | `0` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0.0 - 1.0` | 以指定概率将图像上下翻转，在不影响对象特征的情况下增加数据变化。 |
| `fliplr` | `float` | `0.5` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0.0 - 1.0` | 以指定概率将图像左右翻转，适用于学习对称对象并增加数据集多样性。 |
| `bgr` | `float` | `0` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | `0.0 - 1.0` | 以指定概率将图像通道从 RGB 翻转为 BGR，适用于增强模型对错误通道顺序的鲁棒性。 |
| `mosaic` | `float` | `1` | `detect`、`segment`、`semantic`、`pose`、`obb` | `0.0 - 1.0` | 将四张训练图像合并为一张，模拟不同的场景构成和对象交互。对复杂场景理解非常有效。 |
| `mixup` | `float` | `0` | `detect`、`segment`、`semantic`、`pose`、`obb` | `0.0 - 1.0` | 混合两张图像及其标签，创建合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力。 |
| `cutmix` | `float` | `0` | `detect`、`segment`、`pose`、`obb` | `0.0 - 1.0` | 组合两张图像的部分区域，在保留清晰区域的同时创建局部混合效果。通过创建遮挡场景增强模型鲁棒性。 |
| `copy_paste` | `float` | `0` | `segment`、`obb` | `0.0 - 1.0` | 要粘贴的符合条件对象所占的比例；`flip` 会在图像内镜像这些对象，而 `mixup` 还会将该值用作跨图像应用概率。 |
| `copy_paste_mode` | `str` | `flip` | `segment`、`obb` | - | 指定要使用的 `copy-paste` 策略。选项包括 `'flip'` 和 `'mixup'`。 |
| `auto_augment` | `str` | `randaugment` | `classify` | - | 应用预定义的增强策略（`'randaugment'`、`'autoaugment'` 或 `'augmix'`），通过增加视觉多样性来提升模型性能。 |
| `erasing` | `float` | `0.4` | `classify` | `0.0 - 1.0` | 在训练期间随机擦除图像区域，促使模型关注不太明显的特征。 |
| `augmentations` | `list` | `None` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | - | 用于高级数据增强的自定义 Albumentations 变换（仅限 Python API）。接受变换对象列表，以满足特殊的增强需求。 |

## 三、FAQ 常见训练设置简表

| 参数 | 默认值 | 描述 |
|---|---|---|
| `model` | `None` | 用于训练的模型文件的路径。 |
| `data` | `None` | 数据集 YAML 的路径（例如 `coco8.yaml`），或者是用于分类的数据集目录或名称（例如 `imagenet10`）。 |
| `epochs` | `100` | 训练总轮数。 |
| `batch` | `16` | 批量大小，可调整为整数或自动模式。 |
| `imgsz` | `640` | 训练的目标图像大小。 |
| `device` | `None` | 用于训练的计算设备，例如 `cpu`、`0`、`0,1` 或 `mps`。 |
| `save` | `True` | 启用训练检查点和最终模型权重的保存。 |

## 四、YOLO26 预训练 Detect 模型性能对比

| 模型 | 尺寸（像素） | mAPval 50-95 | mAPval 50-95(e2e) | 速度 CPU ONNX（毫秒） | 速度 T4 TensorRT10（毫秒） | 参数量（M） | FLOPs（B） |
|------|------|------|------|------|------|------|------|
| YOLO26n | 640 | 40.9 | 40.1 | **38.9 ± 0.7** | **1.7 ± 0.0** | **2.4** | **5.5** |
| YOLO26s | 640 | 48.6 | 47.8 | 87.2 ± 0.9 | 2.5 ± 0.0 | 9.5 | 20.9 |
| YOLO26m | 640 | 53.1 | 52.5 | 220.0 ± 1.4 | 4.7 ± 0.1 | 20.4 | 68.4 |
| YOLO26l | 640 | 55.0 | 54.4 | 286.2 ± 2.0 | 6.2 ± 0.2 | 24.8 | 86.8 |
| YOLO26x | 640 | **57.5** | **56.9** | 525.8 ± 4.0 | 11.8 ± 0.2 | 55.7 | 194.4 |

## 五、检测结果属性（result.boxes）

| 属性 | 类型 | 形状 | 描述 |
|------|------|------|------|
| `result.boxes` | `Boxes` | `(N)` | 检测框。 |
| `result.boxes.data` | `torch.float32` | `(N,6/7)` | 原始 `[x1,y1,x2,y2,conf,cls]`，以及可选的跟踪 ID。 |
| `result.boxes.xyxy` | `torch.float32` | `(N,4)` | `xyxy` 像素框。 |
| `result.boxes.conf` | `torch.float32` | `(N,)` | 置信度分数。 |
| `result.boxes.cls` | `torch.float32` | `(N,)` | 类别 ID；转换为 `int` 以获取名称。 |

## 六、YOLO26 导出格式表

`nms=None` 默认输出原始结果用于外部 NMS，`nms=False` 可选择无 NMS 检测头。

| 格式 | `format` 参数 | 模型 | 元数据 | 参数 |
|------|------|------|------|------|
| PyTorch | \- | `yolo26n.pt` | ✅ | \- |
| TorchScript | `torchscript` | `yolo26n.torchscript` | ✅ | `imgsz`、`quantize`、`dynamic`、`nms`、`batch`、`device` |
| ONNX | `onnx` | `yolo26n.onnx` | ✅ | `imgsz`、`quantize`、`dynamic`、`simplify`、`opset`、`nms`、`batch`、`data`、`fraction`、`device` |
| OpenVINO | `openvino` | `yolo26n_openvino_model/` | ✅ | `imgsz`、`quantize`、`dynamic`、`nms`、`batch`、`data`、`fraction`、`device` |
| TensorRT | `engine` | `yolo26n.engine` | ✅ | `imgsz`、`quantize`、`dynamic`、`simplify`、`opset`、`workspace`、`nms`、`batch`、`data`、`fraction`、`device` |
| CoreML | `coreml` | `yolo26n.mlpackage` | ✅ | `imgsz`、`dynamic`、`quantize`、`nms`、`batch`、`device` |
| TF SavedModel | `saved_model` | `yolo26n_saved_model/` | ✅ | `imgsz`、`keras`、`quantize`、`opset`、`nms`、`batch`、`data`、`fraction`、`device` |
| TF GraphDef | `pb` | `yolo26n.pb` | ❌ | `imgsz`、`opset`、`batch`、`device` |
| TF Edge TPU | `edgetpu` | `yolo26n_edgetpu.tflite` | ✅ | `imgsz`、`quantize`、`opset`、`data`、`fraction`、`device` |
| PaddlePaddle | `paddle` | `yolo26n_paddle_model/` | ✅ | `imgsz`、`batch`、`device` |
| MNN | `mnn` | `yolo26n.mnn` | ✅ | `imgsz`、`batch`、`dynamic`、`quantize`、`simplify`、`opset`、`nms`、`device` |
| NCNN | `ncnn` | `yolo26n_ncnn_model/` | ✅ | `imgsz`、`quantize`、`batch`、`device` |
| IMX500 | `imx` | `yolo26n_imx_model/` | ✅ | `imgsz`、`quantize`、`data`、`fraction`、`nms`、`device` |
| RKNN | `rknn` | `yolo26n_rknn_model/` | ✅ | `imgsz`、`batch`、`name`、`quantize`、`simplify`、`opset`、`data`、`fraction`、`device` |
| ExecuTorch | `executorch` | `yolo26n_executorch_model/` | ✅ | `imgsz`、`batch`、`device` |
| Axelera | `axelera` | `yolo26n_axelera_model/` | ✅ | `imgsz`、`batch`、`quantize`、`data`、`fraction`、`device` |
| DEEPX | `deepx` | `yolo26n_deepx_model/` | ✅ | `imgsz`、`quantize`、`simplify`、`opset`、`data`、`optimize`、`device` |
| Qualcomm QNN | `qnn` | `yolo26n_qnn.onnx` | ✅ | `imgsz`、`batch`、`name`、`quantize`、`simplify`、`opset`、`data`、`fraction`、`device` |
| LiteRT | `litert` | `yolo26n.tflite` | ✅ | `imgsz`、`quantize`、`batch`、`data`、`fraction`、`device` |
| Hailo | `hailo` | `yolo26n_hailo_model/` | ✅ | `imgsz`、`name`、`quantize`、`data`、`fraction`、`simplify`、`conf`、`iou` |
| Huawei Ascend | `ascend` | `yolo26n_ascend_model/` | ✅ | `imgsz`、`batch`、`name`、`quantize`、`opset`、`simplify`、`nms` |
| Apple Core AI | `coreai` | `yolo26n.aimodel` | ✅ | `imgsz`、`batch`、`quantize` |
