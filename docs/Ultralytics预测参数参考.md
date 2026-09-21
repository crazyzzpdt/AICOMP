# Ultralytics 预测参数参考

2026-09-21 更新：fusion预测与验证统一五通道补边及真实内容裁框，协议fp32_single_label_content_clip_v2；已通过PNG/JPG输入逐像素一致及两图预测ZIP检查。新协议本地AP须与同协议基线比较，不等同线上分数。复赛数据数量须以实际资料为准，不套用初赛1000组；详见[v14执行记录](v14一致性修复与执行记录.md)。

2026-09-20 修正：不能只凭imgsz相同就声称训练/预测一致。v12/v13的native_square验证五通道补114，旧预测误用了fixed_rect的辅助通道补0；现按权重recipe.geometry选择，native_square共享ceil缩放并使用原生114补边。prediction.json记录实际geometry、padding_values和预处理版本；固定画布rect=False。默认重用v13 best输出predict_v13_preprocess_fix，不改conf0.001/iou0.7/max_det100。旧v4、固定矩形及D-FINE保留各自历史路径。未跑预测，收益待用户提交；详见[v13复盘](v13复盘与v14阶段筛选.md)。下方历史默认值不覆盖本段。

2026-09-18 项目适配：当前入口区分YOLO .pt与D-FINE .pth。YOLO默认yolo_profile=v4，逐张矩形填充、原生单标签NMS；current保留批量自定义候选。D-FINE使用五通道正方形等比填充和原生查询排序，iou/multi_label不是它的调参开关，必须匹配训练imgsz。visual_conf与输出压缩只影响展示和写盘，不改变检测输入或TXT候选；conf降低未必改变每图前100框。用法见 [预测文档](预测与赛事提交.md)。

来源（2026-09-16 整理）：

- [使用 Ultralytics YOLO 进行模型预测](https://docs.ultralytics.com/zh/modes/predict)
- 根据用户提供的网页保存文件整理，保留官方参数、输入约束和结果接口原文；删除介绍、应用展示与重复示例。

## 推理源

如下面的表格所示，YOLO26 可以处理不同类型的输入源进行推理。这些源包括静态图像、视频流和各种数据格式。表格还说明每种源是否可以通过参数 `stream=True` ✅ 使用流式模式。流式模式适合处理视频或实时流，因为它会生成结果生成器，而不是将所有帧加载到内存中。

**提示**

使用 `stream=True` 处理长视频或大型数据集，以高效管理内存。当使用 `stream=False` 时，所有帧或数据点的结果都会存储在内存中，大型输入很快就会占满内存并导致内存不足错误。相比之下，`stream=True` 使用生成器，只将当前帧或数据点的结果保存在内存中，从而显著减少内存占用并避免内存不足问题。

| 来源 | 示例 | 类型 | 备注 |
|---|---|---|---|
| 图像 | `'image.jpg'` | `str` 或 `Path` | 单个图像文件。 |
| URL | `'https://ultralytics.com/images/bus.jpg'` | `str` | 图像 URL。 |
| 屏幕截图 | `'screen'` | `str` | 截取屏幕截图。 |
| PIL | `Image.open('image.jpg')` | `PIL.Image` | 包含 RGB 通道的 HWC 格式。 |
| [OpenCV](https://www.ultralytics.com/glossary/opencv) | `cv2.imread('image.jpg')` | `np.ndarray` | 包含 BGR 通道的 HWC 格式 `uint8 (0-255)`。 |
| NumPy | `np.zeros((640,1280,3))` | `np.ndarray` | 包含 BGR 通道的 HWC 格式 `uint8 (0-255)`。 |
| torch | `torch.zeros(16,3,320,640)` | `torch.Tensor` | 包含 RGB 通道的 BCHW 格式 `float32 (0.0-1.0)`。 |
| CSV | `'sources.csv'` | `str` 或 `Path` | 包含图像、视频或目录路径的 CSV 文件。 |
| 视频 ✅ | `'video.mp4'` | `str` 或 `Path` | MP4、AVI 等格式的视频文件。 |
| 目录 ✅ | `'path/'` | `str` 或 `Path` | 包含图像或视频的目录路径。 |
| glob ✅ | `'path/*.jpg'` | `str` | 用于匹配多个文件的 Glob 模式。使用 `*` 字符作为通配符。 |
| YouTube ✅ | `'https://youtu.be/LNwODJXcvt4'` | `str` | YouTube 视频 URL。 |
| 流 ✅ | `'rtsp://example.com/media.mp4'` | `str` | 用于 RTSP、RTMP、TCP 等流式协议的 URL，或 IP 地址。 |
| 多流 ✅ | `'list.streams'` | `str` 或 `Path` | `*.streams` 文本文件，每行包含一个流 URL，即 8 个流将以批次大小 8 运行。 |
| 摄像头 ✅ | `0` | `int` | 要执行推理的已连接摄像头设备索引。 |

下面是使用每种源类型的代码示例：

## 推理参数

`model.predict()` 接受多个参数，这些参数可以在推理时传入以覆盖默认值：

### 固定形状与最小矩形（`rect`）

默认情况下，predict 使用 **`rect=True`**，在可能的情况下启用**最小矩形**填充。图像会缩放以适应 `imgsz`，并且只填充到最近的步幅倍数，因此最终张量可能**小于** `imgsz`。只有当批次中的**所有图像形状相同**且后端支持时，才会使用最小矩形填充（PyTorch `.pt`，或动态 ONNX / Triton）。否则，图像会填充到**完整的** `imgsz` 目标尺寸。

使用 **`rect=False`** 始终填充到完整的 `imgsz` 目标尺寸。当你需要固定输入尺寸以匹配导出模型（ONNX、TensorRT 等）时，建议使用此选项。

**整数与元组 `imgsz`**

- **整数** `imgsz=640` 在步幅取整后会变为正方形目标 `(640, 640)`。
- **元组** `imgsz=(384, 672)` 会设置矩形目标尺寸。使用 `rect=True` 和 `auto=True` 时，实际张量可能小于该目标尺寸。

**训练与 predict/export**

训练只接受单个整数 `imgsz`（`[h, w]` 列表会被强制转换为最大值）。Predict 和 export 接受整数或 `(height, width)` 元组。

**示例**

**Python**

```python
from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("https://ultralytics.com/images/bus.jpg", save=True, imgsz=320, conf=0.25)
```

**CLI**

```bash
# Run inference on 'bus.jpg'
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'
```

推理参数：

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `source` | `str` 或 `int` 或 `None` | `None` | 指定推理的数据源。可以是图像路径、视频文件、目录、URL 或实时流的设备 ID。如果省略，系统会记录警告，并将模型回退到内置演示资源（`ultralytics/assets`，或 OBB 的演示 URL）。支持多种格式和数据源，可灵活应用于[不同类型的输入](https://docs.ultralytics.com/modes/predict#inference-sources)。 |
| `conf` | `float` | `0.25` | 设置检测的最低置信度阈值。置信度低于此阈值的检测对象将被忽略。调整此值有助于减少误报。 |
| `iou` | `float` | `0.7` | [交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) 阈值，用于非极大值抑制 (NMS)。较低的值会通过消除重叠框来减少检测结果，有助于减少重复检测。 |
| `imgsz` | `int` 或 `tuple` | `640` | Letterbox 目标尺寸。整数值表示正方形 `N×N`；元组表示 `(height, width)`。使用 `rect=True` 时，由于最小矩形填充，实际张量可能小于此目标尺寸。使用 `rect=False` 可指定固定尺寸。参见[固定形状与最小矩形](https://docs.ultralytics.com/modes/predict#fixed-shape-vs-minimum-rectangle-rect)。 |
| `rect` | `bool` | `True` | 如果为 `True`，则在可能时使用最小矩形填充（相同形状的批次和受支持的后端）。如果为 `False`，则始终填充到完整的 `imgsz`。参见[固定形状与最小矩形](https://docs.ultralytics.com/modes/predict#fixed-shape-vs-minimum-rectangle-rect)。 |
| `quantize` | `int` 或 `str` | `None` | 推理精度：`16`/`"fp16"` 与 `32`/`"fp32"`/未设置为 PyTorch 和 TorchScript 模型选择 FP16 或 FP32 计算；其他格式则在其构件与运行时所选的精度下进行计算。在 `16` 下，OpenVINO 仍在客户端将输入进行 FP16 舍入并将其展宽回 FP32，但这不会改变运行时实际的计算精度。INT8/PTQ 量化在[导出](https://docs.ultralytics.com/modes/export#quantization-options)期间进行配置，然后通过加载导出的模型来使用。该参数替代了已弃用的 `half` 标志。 |
| `device` | `str` | `None` | 指定推理设备（例如 `cpu`、`cuda:0`、`0`、`npu` 或 `npu:0`）。允许你在 CPU、特定 GPU、华为 Ascend NPU 或其他计算设备之间进行选择，以执行模型推理。 |
| `dnn` | `bool` | `False` | 如果为 `True`，则在 ONNX 模型推理中使用 [OpenCV](https://www.ultralytics.com/glossary/opencv) DNN 模块，而不是 ONNX Runtime。 |
| `data` | `str` | `None` | 数据集 YAML 的路径（例如 `coco8.yaml`），仅读取其中的 `names`，且仅当加载的模型不包含自身的类别名称时使用：例如第三方导出模型，或与其附带元数据分离的 Ultralytics 导出模型。否则，此类模型会报告 `class0`、`class1` 等。 |
| `batch` | `int` | `1` | 指定推理的批次大小（仅当源是[目录、视频文件或 `.txt` 文件](https://docs.ultralytics.com/modes/predict#inference-sources)时有效）。较大的批次大小可以提供更高的吞吐量，从而缩短推理所需的总时间。 |
| `max_det` | `int` | `300` | 每张图像允许的最大检测数量。限制模型在单次推理中可以检测的对象总数，防止密集场景产生过多输出。 |
| `vid_stride` | `int` | `1` | 视频输入的帧步长。允许跳过视频帧以加快处理速度，但会降低时间分辨率。值为 1 时处理每一帧，值更高时跳过部分帧。 |
| `stream_buffer` | `bool` | `False` | 决定是否为视频流排队接收到的帧。如果为 `False`，则会丢弃旧帧以容纳新帧（针对实时应用进行了优化）。如果为 `True`，则会将新帧加入缓冲区，确保不跳过任何帧，但当推理 FPS 低于流 FPS 时会产生延迟。 |
| `visualize` | `bool` | `False` | 在每个预测结果旁保存类别激活热力图，显示哪些像素提高了预测类别的分数。遵循 `conf` 和 `classes`，因此 `classes=[0]` 仅映射该类别。仅适用于 Ultralytics PyTorch 模型。 |
| `augment` | `bool` | `False` | 为预测启用测试时增强 (TTA)，可能提升检测鲁棒性，但会降低推理速度。仅适用于 Ultralytics PyTorch 模型。 |
| `agnostic_nms` | `bool` | `False` | 启用类无关的 NMS，压制不同类别中得分较低的重叠框，而不仅仅是同一类别内。在类别重叠常见的多类检测场景中很有用。使用无 NMS 推理（在 YOLO26 或 YOLOv10 上为 `nms=False`）时，这仅防止同一检测结果出现多个类别标签（IoU=1.0 重复），而不会在不同框之间执行基于 IoU 阈值的压制。 |
| `classes` | `list[int]` | `None` | 将预测结果过滤为一组类别 ID。仅返回属于指定类别的检测结果。适用于在多类别检测任务中聚焦相关对象。 |
| `retina_masks` | `bool` | `False` | 返回高分辨率分割掩码。启用后，返回的掩码（`masks.data`）将匹配原始图像尺寸。禁用后，掩码将使用推理期间的图像尺寸。 |
| `embed` | `list[int]` | `None` | 指定用于提取特征向量或[嵌入](https://www.ultralytics.com/glossary/embeddings)的层。使用 `model.embed(source)` 获取倒数第二层嵌入，或使用 `model.predict(source, embed=[layer])` 选择特定层。适用于聚类或相似性搜索等下游任务。仅适用于 Ultralytics PyTorch 模型。 |
| `project` | `str` | `None` | 项目目录的名称；启用 `save` 后，预测输出将保存到该目录。 |
| `name` | `str` | `None` | 预测运行的名称。用于在项目文件夹中创建子目录；启用 `save` 后，预测输出将保存到该子目录。 |
| `stream` | `bool` | `False` | 通过返回 Results 对象生成器，而不是一次性将所有帧加载到内存中，为长视频或大量图像启用节省内存的处理方式。 |
| `verbose` | `bool` | `True` | 控制是否在终端中显示详细的推理日志，为预测过程提供实时反馈。 |
| `compile` | `bool` 或 `str` | `False` | 使用 `backend='inductor'` 启用 PyTorch 2.x 的 `torch.compile` 图编译。接受 `True` → `"default"`、`False` → 禁用，或使用 `"default"`、`"reduce-overhead"`、`"max-autotune-no-cudagraphs"` 等字符串模式。不支持时会发出警告并回退到 eager 模式。 |
| `channels_last` | `bool` | `None` | 对原生 PyTorch 推理使用 channels_last (NHWC) 内存格式。`None` 会在启用 oneDNN 的 Linux 和 Windows x86 CPU 上、PyTorch 1.13 或更高版本中自动启用该格式，`False` 会禁用该格式，`True` 会在受支持的 x86 CPU 或 CUDA 设备上请求该格式。ARM64、MPS、较旧的 PyTorch 版本、不支持 oneDNN 的 CPU，以及 TensorRT 和 ONNX 等导出格式均保持不变。 |
| `nms` | `bool`，可选 | `None` | 默认运行带 NMS 的一对多推理（`None` 或 `True`）。设置 `False` 以在可用时使用无 NMS 的一对一（one-to-one）头。详见[端到端检测指南](https://docs.ultralytics.com/guides/end2end-detection)。 |

可视化参数：

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `show` | `bool` | `False` | 如果为 `True`，则会在窗口中显示带标注的图像或视频。适用于开发或测试期间获取即时视觉反馈。 |
| `save` | `bool` | `False or True` | 启用后将带注释的图像或视频保存到文件。适用于文档记录、进一步分析或分享结果。使用 CLI 时默认为 True，使用 Python 时默认为 False。 |
| `save_frames` | `bool` | `False` | 处理视频时，将单独的帧保存为图像。适用于提取特定帧或进行详细的逐帧分析。 |
| `save_txt` | `bool` | `False` | 将检测结果按照 `[class] [x_center] [y_center] [width] [height] [confidence]` 格式保存到文本文件中。适用于与其他分析工具集成。 |
| `save_conf` | `bool` | `False` | 在保存的文本文件中包含置信度分数。增加可用于后处理和分析的详细信息。 |
| `save_crop` | `bool` | `False` | 保存检测对象的裁剪图像。适用于数据集增强、分析，或为特定对象创建专用数据集。 |
| `show_labels` | `bool` | `True` | 在可视化输出中显示每个检测结果的标签，帮助你立即了解检测到的对象。 |
| `show_conf` | `bool` | `True` | 在每个检测结果旁显示其置信度分数，让你了解模型对每个检测结果的确定程度。 |
| `show_boxes` | `bool` | `True` | 在检测到的对象周围绘制边界框。对于在图像或视频帧中直观识别和定位对象至关重要。 |
| `line_width` | `int or None` | `None` | 指定边界框的线宽。如果为 `None`，则会根据图像尺寸自动调整线宽。可自定义视觉效果以提高清晰度。 |

## 图像和视频格式

YOLO26 支持各种图像和视频格式，具体格式见 [ultralytics/data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py)。有效后缀和示例预测命令请参阅下表。

### 图像

下表包含有效的 Ultralytics 图像格式。

**注意**

HEIC/HEIF 格式需要 `pi-heif`，首次使用时会自动安装。AVIF 由 Pillow 原生支持。

| 图像后缀 | 示例预测命令 | 参考 |
|---|---|---|
| `.avif` | `yolo predict source=image.avif` | [AV1 图像文件格式](https://en.wikipedia.org/wiki/AVIF) |
| `.bmp` | `yolo predict source=image.bmp` | [Microsoft BMP 文件格式](https://en.wikipedia.org/wiki/BMP_file_format) |
| `.dng` | `yolo predict source=image.dng` | [Adobe DNG](https://en.wikipedia.org/wiki/Digital_Negative) |
| `.heic` | `yolo predict source=image.heic` | [高效图像文件格式](https://en.wikipedia.org/wiki/HEIF) |
| `.heif` | `yolo predict source=image.heif` | [高效图像文件格式](https://en.wikipedia.org/wiki/HEIF) |
| `.jp2` | `yolo predict source=image.jp2` | [JPEG 2000](https://en.wikipedia.org/wiki/JPEG_2000) |
| `.jpeg` | `yolo predict source=image.jpeg` | [JPEG](https://en.wikipedia.org/wiki/JPEG) |
| `.jpg` | `yolo predict source=image.jpg` | [JPEG](https://en.wikipedia.org/wiki/JPEG) |
| `.mpo` | `yolo predict source=image.mpo` | [多图像对象](https://fileinfo.com/extension/mpo) |
| `.png` | `yolo predict source=image.png` | [便携式网络图形](https://en.wikipedia.org/wiki/PNG) |
| `.tif` | `yolo predict source=image.tif` | [标签图像文件格式](https://en.wikipedia.org/wiki/TIFF) |
| `.tiff` | `yolo predict source=image.tiff` | [标签图像文件格式](https://en.wikipedia.org/wiki/TIFF) |
| `.webp` | `yolo predict source=image.webp` | [WebP](https://en.wikipedia.org/wiki/WebP) |

### 视频

下表包含有效的 Ultralytics 视频格式。

| 视频后缀 | 示例预测命令 | 参考 |
|---|---|---|
| `.asf` | `yolo predict source=video.asf` | [高级系统格式](https://en.wikipedia.org/wiki/Advanced_Systems_Format) |
| `.avi` | `yolo predict source=video.avi` | [音视频交错格式](https://en.wikipedia.org/wiki/Audio_Video_Interleave) |
| `.gif` | `yolo predict source=video.gif` | [图形交换格式](https://en.wikipedia.org/wiki/GIF) |
| `.m4v` | `yolo predict source=video.m4v` | [MPEG-4 第 14 部分](https://en.wikipedia.org/wiki/M4V) |
| `.mkv` | `yolo predict source=video.mkv` | [Matroska](https://en.wikipedia.org/wiki/Matroska) |
| `.mov` | `yolo predict source=video.mov` | [QuickTime 文件格式](https://en.wikipedia.org/wiki/QuickTime_File_Format) |
| `.mp4` | `yolo predict source=video.mp4` | [MPEG-4 第 14 部分 - Wikipedia](https://en.wikipedia.org/wiki/MPEG-4_Part_14) |
| `.mpeg` | `yolo predict source=video.mpeg` | [MPEG-1 第 2 部分](https://en.wikipedia.org/wiki/MPEG-1) |
| `.mpg` | `yolo predict source=video.mpg` | [MPEG-1 第 2 部分](https://en.wikipedia.org/wiki/MPEG-1) |
| `.ts` | `yolo predict source=video.ts` | [MPEG 传输流](https://en.wikipedia.org/wiki/MPEG_transport_stream) |
| `.wmv` | `yolo predict source=video.wmv` | [Windows Media 视频](https://en.wikipedia.org/wiki/Windows_Media_Video) |
| `.webm` | `yolo predict source=video.webm` | [WebM 项目](https://en.wikipedia.org/wiki/WebM) |

## 处理 Results

### 列表与流式生成器

**使用 `stream=False` 返回列表**

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # pretrained YOLO26n model

# Run batched inference on a list of images
results = model(["image1.jpg", "image2.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
```

**使用 `stream=True` 返回生成器**

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # pretrained YOLO26n model

# Run batched inference on a list of images
results = model(["image1.jpg", "image2.jpg"], stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
```

`Results` 对象具有以下属性：

| 属性 | 类型 | 描述 |
|---|---|---|
| `orig_img` | `np.ndarray` | 作为 NumPy 数组的原始图像。 |
| `orig_shape` | `tuple` | 采用 (height, width) 格式的原始图像形状。 |
| `boxes` | `Boxes, optional` | 包含检测边界框的 Boxes 对象。 |
| `masks` | `Masks, optional` | 包含检测掩码的 Masks 对象。 |
| `probs` | `Probs, optional` | 包含分类任务中每个类别概率的 Probs 对象。 |
| `keypoints` | `Keypoints, optional` | 包含每个对象检测到的关键点的 Keypoints 对象。 |
| `obb` | `OBB, optional` | 包含定向边界框的 OBB 对象。 |
| `semantic_mask` | `SemanticMask, optional` | 包含密集逐像素类别映射的 SemanticMask 对象。 |
| `speed` | `dict` | 一个字典，记录预处理、推理和后处理速度，单位为每张图像的毫秒数。 |
| `names` | `dict` | 将类别索引映射到类别名称的字典。 |
| `path` | `str` | 图像文件的路径。 |
| `save_dir` | `str, optional` | 用于保存结果的目录。 |

### 按任务划分的 Results

每次预测每张图像或每帧返回一个 `Results` 对象。上面的通用字段始终可用，而特定于任务的预测数据存储在下面的字段中。YOLO坐标和置信度张量是 `torch.float32`；概率张量是 `torch.float32`，除非使用半精度，此时为 `torch.float16`。在 `result.numpy()` 之后，张量会变成具有相应 NumPy 数据类型的 NumPy 数组。实例掩码是 `torch.uint8` 二进制张量，而语义掩码使用适合类别的最小实际整数数据类型：`torch.uint8`、`torch.int16` 或 `torch.int32`，具体取决于类别数量。

**Detect**

| 属性 | 类型 | 形状 | 描述 |
|---|---|---|---|
| `result.boxes` | `Boxes` | `(N)` | 检测框。 |
| `result.boxes.data` | `torch.float32` | `(N,6/7)` | 原始 `[x1,y1,x2,y2,conf,cls]`，以及可选的跟踪 ID。 |
| `result.boxes.xyxy` | `torch.float32` | `(N,4)` | `xyxy` 像素框。 |
| `result.boxes.conf` | `torch.float32` | `(N,)` | 置信度分数。 |
| `result.boxes.cls` | `torch.float32` | `(N,)` | 类别 ID；转换为 `int` 以获取名称。 |

**Segment**

| 属性 | 类型 | 形状 | 描述 |
|---|---|---|---|
| `result.boxes` | `Boxes` | `(N)` | 实例框/类别/置信度。 |
| `result.masks` | `Masks` | `(N)` | 实例掩码。 |
| `result.masks.data` | `torch.uint8` | `(N,H,W)` | 二进制掩码，值为 `0` 或 `1`。 |
| `result.masks.xy` | `np.float32` | `list[(P,2)]` | 像素多边形。 |
| `result.masks.xyn` | `np.float32` | `list[(P,2)]` | 归一化多边形。 |

**语义分割**

| 属性 | 类型 | 形状 | 描述 |
|---|---|---|---|
| `result.semantic_mask` | `SemanticMask` | `(H,W)` | 密集类别映射。 |
| `result.semantic_mask.data` | `torch.uint8`<br>`torch.int16`<br>`torch.int32` | `(H,W)` | 逐像素类别 ID，数据类型由类别数量决定。 |
| `result.masks` | - | - | 没有实例掩码。 |
| `result.boxes` | - | - | 没有实例框/置信度。 |

**分类**

| 属性 | 类型 | 形状 | 描述 |
|---|---|---|---|
| `result.probs` | `Probs` | `(C,)` | 类别概率。 |
| `result.probs.data` | `torch.float32` | `(C,)` | 每个类别的概率。 |
| `result.probs.top1` | `int` | `()` | 最高类别 ID。 |
| `result.probs.top1conf` | `torch.float32` | `()` | 最高置信度。 |
| `result.probs.top5` | `list[int]` | `(<=5)` | 排名前 5 的类别 ID。 |

**姿态**

| 属性 | 类型 | 形状 | 描述 |
|---|---|---|---|
| `result.boxes` | `Boxes` | `(N)` | 实例框。 |
| `result.keypoints` | `Keypoints` | `(N)` | 关键点。 |
| `result.keypoints.data` | `torch.float32` | `(N,K,2/3)` | `x,y`，以及可选的可见性/置信度。 |
| `result.keypoints.xy` | `torch.float32` | `(N,K,2)` | 像素关键点。 |
| `result.keypoints.xyn` | `torch.float32` | `(N,K,2)` | 归一化关键点。 |

**OBB**

| 属性 | 类型 | 形状 | 描述 |
|---|---|---|---|
| `result.obb` | `OBB` | `(N)` | 定向框。 |
| `result.obb.data` | `torch.float32` | `(N,7/8)` | 带置信度/类别的原始旋转框。 |
| `result.obb.xywhr` | `torch.float32` | `(N,5)` | `xywhr` 旋转框。 |
| `result.obb.xyxyxyxy` | `torch.float32` | `(N,4,2)` | 四个角点。 |
| `result.obb.conf` | `torch.float32` | `(N,)` | 置信度分数。 |

`Results` 对象具有以下方法：

| 方法 | 返回类型 | 描述 |
|---|---|---|
| `update()` | `None` | 使用框、掩码、概率、obb、关键点或语义掩码等新数据更新 Results 对象。 |
| `cpu()` | `Results` | 返回 Results 对象的副本，并将所有张量移至 CPU 内存。 |
| `numpy()` | `Results` | 返回一个副本，其中所有张量都已转换为 NumPy 数组。 |
| `cuda()` | `Results` | 返回一个副本，其中所有张量都已移动到 GPU 内存。 |
| `to()` | `Results` | 返回一个副本，其中张量已移动到指定设备并转换为指定 dtype。 |
| `new()` | `Results` | 创建一个新的 Results 对象，其中包含相同的图像、路径、名称和速度属性。 |
| `plot()` | `np.ndarray` | 在输入 BGR 图像上绘制检测结果，并返回标注后的图像。 |
| `show()` | `None` | 显示包含标注推理结果的图像。 |
| `save()` | `str` | 将标注后的推理结果图像保存到文件，并返回文件名。 |
| `verbose()` | `str` | 为每个任务返回一条日志字符串，详细说明检测和分类结果。 |
| `save_txt()` | `str` | 将检测结果保存到文本文件，并返回保存文件的路径。 |
| `save_crop()` | `None` | 将裁剪后的检测图像保存到指定目录。 |
| `summary()` | `List[Dict[str, Any]]` | 将推理结果转换为摘要字典，可选择进行归一化。 |
| `to_df()` | `DataFrame` | 将检测结果转换为 Polars DataFrame。 |
| `to_csv()` | `str` | 将检测结果转换为 CSV 格式。 |
| `to_json()` | `str` | 将检测结果转换为 JSON 格式。 |

有关更多详情，请参阅 [`Results` 类文档](https://docs.ultralytics.com/reference/engine/results)。

### 框数

`Boxes` 对象可用于索引、操作边界框，并将其转换为不同格式。

下面的表格列出了 `Boxes` 类的方法和属性，包括其名称、类型和描述：

| 名称 | 类型 | 描述 |
|---|---|---|
| `cpu()` | 方法 | 将对象移动到 CPU 内存。 |
| `numpy()` | 方法 | 将对象转换为 NumPy 数组。 |
| `cuda()` | 方法 | 将对象移动到 CUDA 内存。 |
| `to()` | 方法 | 将对象移动到指定设备。 |
| `xyxy` | 属性（`torch.Tensor`） | 返回 xyxy 格式的边界框。 |
| `conf` | 属性（`torch.Tensor`） | 返回边界框的置信度值。 |
| `cls` | 属性（`torch.Tensor`） | 返回边界框的类别值。 |
| `id` | 属性（`torch.Tensor`） | 返回边界框的跟踪 ID（如果可用）。 |
| `xywh` | 属性（`torch.Tensor`） | 返回 xywh 格式的边界框。 |
| `xyxyn` | 属性（`torch.Tensor`） | 返回按原始图像尺寸归一化的 xyxy 格式边界框。 |
| `xywhn` | 属性（`torch.Tensor`） | 返回按原始图像尺寸归一化的 xywh 格式边界框。 |

有关更多详情，请参阅 [`Boxes` 类文档](https://docs.ultralytics.com/reference/engine/results#ultralytics.engine.results.Boxes)。

### 掩码

`Masks` 对象可用于索引、操作掩码，并将其转换为分段。

下面的表格列出了 `Masks` 类的方法和属性，包括其名称、类型和描述：

| 名称 | 类型 | 描述 |
|---|---|---|
| `data` | 属性（`torch.Tensor`） | 形状为 `(N,H,W)`、值为 `0` 或 `1` 的 `torch.uint8` 二值掩码张量。 |
| `cpu()` | 方法 | 返回位于 CPU 内存中的掩码张量。 |
| `numpy()` | 方法 | 以 NumPy 数组形式返回掩码张量。 |
| `cuda()` | 方法 | 返回位于 GPU 内存中的掩码张量。 |
| `to()` | 方法 | 返回位于指定设备上并具有指定 dtype 的掩码张量。 |
| `xyn` | 属性（`list[np.ndarray]`） | 归一化掩码多边形列表。 |
| `xy` | 属性（`list[np.ndarray]`） | 以像素坐标表示的掩码多边形列表。 |

有关更多详情，请参阅 [`Masks` 类文档](https://docs.ultralytics.com/reference/engine/results#ultralytics.engine.results.Masks)。

### SemanticMask

`SemanticMask` 存储语义分割结果的一张稠密类别图。与 `Masks` 不同，它不包含每个对象对应的一张二值掩码，也不提供多边形辅助方法。

| 名称 | 类型 | 描述 |
|---|---|---|
| `data` | 属性（`torch.Tensor`） | 形状为 `(H,W)` 的类别 ID 图。dtype 为 `torch.uint8`、`torch.int16` 或 `torch.int32`，具体取决于类别数量。 |
| `shape` | 属性（`tuple`） | 类别图的形状，通常与 `result.orig_shape` 匹配。 |
| `cpu()` | 方法 | 返回位于 CPU 内存中的语义掩码张量。 |
| `numpy()` | 方法 | 以 NumPy 数组形式返回语义掩码张量。 |
| `cuda()` | 方法 | 返回位于 GPU 内存中的语义掩码张量。 |
| `to()` | 方法 | 返回位于指定设备上并具有指定 dtype 的语义掩码张量。 |

### 关键点

`Keypoints` 对象可用于索引、操作坐标并进行归一化。

下面的表格列出了 `Keypoints` 类的方法和属性，包括其名称、类型和描述：

| 名称 | 类型 | 描述 |
|---|---|---|
| `cpu()` | 方法 | 返回位于 CPU 内存中的关键点张量。 |
| `numpy()` | 方法 | 以 NumPy 数组形式返回关键点张量。 |
| `cuda()` | 方法 | 返回位于 GPU 内存中的关键点张量。 |
| `to()` | 方法 | 返回位于指定设备上并具有指定 dtype 的关键点张量。 |
| `xyn` | 属性（`torch.Tensor`） | 以张量表示的归一化关键点列表。 |
| `xy` | 属性（`torch.Tensor`） | 以张量表示的像素坐标关键点列表。 |
| `conf` | 属性（`torch.Tensor`） | 如果可用，则返回关键点的置信度值，否则返回 None。 |

有关更多详情，请参阅 [`Keypoints` 类文档](https://docs.ultralytics.com/reference/engine/results#ultralytics.engine.results.Keypoints)。

### Probs

`Probs` 对象可用于获取 `top1` 和 `top5` 分类索引及分数。

下面的表格总结了 `Probs` 类的方法和属性：

| 名称 | 类型 | 描述 |
|---|---|---|
| `cpu()` | 方法 | 返回位于 CPU 内存中的 probs 张量副本。 |
| `numpy()` | 方法 | 以 NumPy 数组形式返回 probs 张量副本。 |
| `cuda()` | 方法 | 返回位于 GPU 内存中的 probs 张量副本。 |
| `to()` | 方法 | 返回位于指定设备上并具有指定 dtype 的 probs 张量副本。 |
| `top1` | 属性（`int`） | 排名第 1 的类别索引。 |
| `top5` | 属性（`list[int]`） | 排名前 5 的类别索引。 |
| `top1conf` | 属性（`torch.Tensor`） | 排名第 1 的类别置信度。 |
| `top5conf` | 属性（`torch.Tensor`） | 排名前 5 的类别置信度。 |

有关更多详情，请参阅 [`Probs` 类文档](https://docs.ultralytics.com/reference/engine/results#ultralytics.engine.results.Probs)。

### OBB

`OBB` 对象可用于索引、操作定向边界框，并将其转换为不同格式。

下面的表格列出了 `OBB` 类的方法和属性，包括其名称、类型和描述：

| 名称 | 类型 | 描述 |
|---|---|---|
| `cpu()` | 方法 | 将对象移动到 CPU 内存。 |
| `numpy()` | 方法 | 将对象转换为 NumPy 数组。 |
| `cuda()` | 方法 | 将对象移动到 CUDA 内存。 |
| `to()` | 方法 | 将对象移动到指定设备。 |
| `conf` | 属性（`torch.Tensor`） | 返回边界框的置信度值。 |
| `cls` | 属性（`torch.Tensor`） | 返回边界框的类别值。 |
| `id` | 属性（`torch.Tensor`） | 返回边界框的跟踪 ID（如果可用）。 |
| `xyxy` | 属性（`torch.Tensor`） | 返回 xyxy 格式的水平边界框。 |
| `xywhr` | 属性（`torch.Tensor`） | 返回 xywhr 格式的旋转边界框。 |
| `xyxyxyxy` | 属性（`torch.Tensor`） | 返回 xyxyxyxy 格式的旋转边界框。 |
| `xyxyxyxyn` | 属性（`torch.Tensor`） | 返回按图像尺寸归一化的 xyxyxyxy 格式旋转边界框。 |

有关更多详情，请参阅 [`OBB` 类文档](https://docs.ultralytics.com/reference/engine/results#ultralytics.engine.results.OBB)。

## 绘制结果

`Results` 对象中的 `plot()` 方法可通过将检测到的对象（例如边界框、掩码、关键点和概率）叠加到原始图像上，实现预测结果的可视化。此方法以 NumPy 数组形式返回标注后的图像，便于显示或保存。

### `plot()` 方法参数

`plot()` 方法支持多个参数，用于自定义输出：

| 参数 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| `conf` | `bool` | 包含检测置信度分数。 | `True` |
| `line_width` | `float` | 边界框的线宽。如果为 `None`，则会随图像尺寸缩放。 | `None` |
| `font_size` | `float` | 文本字体大小。如果为 `None`，则会随图像尺寸缩放。 | `None` |
| `font` | `str` | 文本标注的字体名称。 | `'Arial.ttf'` |
| `pil` | `bool` | 将图像作为 PIL Image 对象返回。 | `False` |
| `img` | `np.ndarray \| torch.Tensor` | 备用图像。张量必须是连续的 HWC BGR uint8。 | `None` |
| `kpt_radius` | `int` | 绘制关键点的半径。 | `5` |
| `kpt_line` | `bool` | 使用线条连接关键点。 | `True` |
| `labels` | `bool` | 在标注中包含类别标签。 | `True` |
| `boxes` | `bool` | 在图像上叠加边界框。 | `True` |
| `masks` | `bool` | 在图像上叠加掩码。 | `True` |
| `probs` | `bool` | 包含分类概率。 | `True` |
| `show` | `bool` | 使用默认图像查看器直接显示标注后的图像。 | `False` |
| `save` | `bool` | 将标注后的图像保存到 `filename` 指定的文件。 | `False` |
| `filename` | `str` | 当 `save` 为 `True` 时，用于保存标注后图像的文件路径和名称。 | `None` |
| `color_mode` | `str` | 指定颜色模式，例如“instance”或“class”。 | `'class'` |
| `txt_color` | `tuple[int, int, int]` | 边界框和图像分类标签的 BGR 文本颜色。 | `(255, 255, 255)` |

## 线程安全推理

在不同线程中并行运行多个 YOLO 模型时，确保推理过程的线程安全至关重要。线程安全的推理可确保每个线程的预测彼此隔离且互不干扰，从而避免竞态条件，并确保输出一致且可靠。

在多线程应用中使用 YOLO 模型时，应为每个线程实例化独立的模型对象，或使用线程本地存储来防止冲突：

**线程安全推理**

在每个线程中实例化一个模型，以实现线程安全的推理：

```python
from threading import Thread

from ultralytics import YOLO

def thread_safe_predict(model, image_path):
    """Performs thread-safe prediction on an image using a locally instantiated YOLO model."""
    model = YOLO(model)
    results = model.predict(image_path)
    # Process results

# Starting threads that each have their own model instance
Thread(target=thread_safe_predict, args=("yolo26n.pt", "image1.jpg")).start()
Thread(target=thread_safe_predict, args=("yolo26n.pt", "image2.jpg")).start()
```
