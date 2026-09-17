"""使用本地五通道检测权重预测官方测试集，并生成赛事提交 ZIP。

输入为同名 visible、infrared、depth 图，复用训练时的 RGB、红外、深度预处理。
输出包含 images、labels、比赛提交内容三个子目录；已有输出目录一律拒绝复用。
正式 ZIP 仅在所有图片预测完成、标签逐项核验通过后生成，不上传任何赛事数据。

在项目目录执行：
    uv run python predict.py
更换权重或输出位置：
    uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_yolo26l_1280_v3/weights/best.pt --output predict_v3
可选保留单模型的次高类别候选（可能增加误检，不保证提分）：
    uv run python predict.py --weights <本地权重路径> --multi-label --output predict_multilabel
查看全部参数：
    uv run python predict.py --help
"""

# 内置库
import argparse
import hashlib
import json
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# 必须在导入 Ultralytics 前设置，推理不联网检查或自动安装依赖。
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

# 三方库
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops

# 自己的模块
from 三模态训练 import fuse_modalities
from 准备三模态数据集 import CLASS_NAMES, IMAGE_SUFFIXES


# 相对命令行路径统一以脚本所在目录为基准，兼容 IDE 从其他位置启动。
PROJECT_ROOT: Path = Path(__file__).resolve().parent
# 选择推理的模型
MODEL_PATH: Path = PROJECT_ROOT / "runs/detect/AIC_RGBIRDepth_yolo26l_1280_v5_full/weights/best.pt"
# 官方初赛测试集；三种模态必须位于该目录的同名子文件夹。
SOURCE_PATH: Path = PROJECT_ROOT / "数据集/测试集/AIC2026_PHASE_1_1000"
# 预测结果与训练 runs 隔离，重复运行须指定新的输出目录。
OUTPUT_PATH: Path = PROJECT_ROOT / "predict"
# 初赛官方提供 1000 组；显式核对数量，防止误交子集。
EXPECTED_COUNT: int = 1000
# 对齐训练验证分辨率；尚未用验证集证明更大推理尺寸能提升精度。
IMAGE_SIZE: int = 1280
# 与训练验证一致，保留低分候选；不是图片展示阈值。
CONF_THRESHOLD: float = 0.001
# 对齐本轮权重选择时的 NMS 阈值，不是比赛计算 AP 的 IoU 阈值。
IOU_THRESHOLD: float = 0.7
# 仅影响带框图片，避免低置信度候选遮挡原始场景。
VISUAL_CONF: float = 0.25
# 赛事规定单图最多 100 个框，超限按置信度截断。
MAX_DETECTIONS: int = 100
# 保持已有提交行为；需要保留同一位置的多个类别候选时显式使用 --multi-label。
MULTI_LABEL: bool = False


# 一、输入配对与模型检查
def collect_samples(source: Path) -> list[tuple[Path, Path, Path]]:
    """按完整文件名配对三模态，拒绝缺图与提交 TXT 同名冲突。

    Args:
        source: 包含 visible、infrared、depth 的官方测试集根目录。

    Returns:
        按文件名排序的可见光、红外、深度路径三元组。

    Raises:
        FileNotFoundError: 模态目录缺失。
        ValueError: 没有图像、模态配对不完整或不同图像共用同一词干。
    """
    folders: list[Path] = [source / modality for modality in ("visible", "infrared", "depth")]
    files: list[dict[str, Path]] = []
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到模态目录：{folder}")
        files.append({p.name: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES})
    names: set[str] = set(files[0])
    if not names:
        raise ValueError(f"可见光目录没有 PNG/JPG/JPEG 图像：{folders[0]}")
    for folder, modality_files in zip(folders[1:], files[1:]):
        if names != set(modality_files):
            missing = sorted(names - set(modality_files))[:5]
            extra = sorted(set(modality_files) - names)[:5]
            raise ValueError(f"三模态配对不完整：{folder.name}，缺失 {missing}，多余 {extra}")
    if len({Path(name).stem.casefold() for name in names}) != len(names):
        raise ValueError("不同图片会生成同名 TXT，请检查大小写或不同扩展名的词干冲突")
    return [(files[0][name], files[1][name], files[2][name]) for name in sorted(names)]


def validate_model(model: YOLO) -> None:
    """只接受本项目的五通道、12 类本地检测权重。

    Args:
        model: 已从本地检查点加载的 YOLO 模型。

    Raises:
        ValueError: 任务、首层通道数或类别编号不符。
    """
    first_conv = next((layer for layer in model.model.modules() if isinstance(layer, torch.nn.Conv2d)), None)
    if model.task != "detect" or first_conv is None or first_conv.in_channels != 5:
        raise ValueError("必须使用训练后的五通道检测权重，不能使用 orgin_models 中的 RGB 预训练权重")
    if model.names != dict(enumerate(CLASS_NAMES)):
        raise ValueError(f"模型类别编号与比赛 12 类不一致：{model.names}")
    # 框架图片加载器从 YAML 读取通道数，缺省的 3 会把五通道数组裁掉后两通道。
    model.model.yaml["channels"] = 5


# 二、五通道预测与原图坐标恢复
class MultimodalDetectionPredictor(DetectionPredictor):
    """保留五通道输入，仅在构建结果图片时将前三通道恢复为 BGR。"""

    def __init__(self, *args: object, multi_label: bool = False, **kwargs: object) -> None:
        """将候选筛选开关保存在预测器中，不传入框架不支持的配置参数。"""
        super().__init__(*args, **kwargs)
        self.multi_label: bool = multi_label

    def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        """检查 RGB、红外、深度顺序的 uint8 输入，再统一缩放和归一化。"""
        for image in images:
            if image.ndim != 3 or image.shape[2] != 5 or image.dtype != np.uint8:
                raise ValueError("预测输入必须是 RGB、红外、深度顺序的 uint8 五通道图像")
        # 当前框架仅翻转三通道 BGR；五通道保持 fuse_modalities 的 RGBIRDepth 顺序。
        return super().preprocess(images)

    def postprocess(self, preds: torch.Tensor, img: torch.Tensor, orig_imgs: list[np.ndarray], **kwargs: object) -> list[Results]:
        """可选保留同框多类别候选，再按类别 NMS；仍仅使用一个模型。

        Note:
            默认沿用框架单标签路径。多标签只保留模型已给出的分数，不人为抬高
            ball 置信度；低分候选可能争用每图 100 框的名额，效果需实际提交确认。
        """
        if not self.multi_label:
            return super().postprocess(preds, img, orig_imgs, **kwargs)
        if not isinstance(orig_imgs, list) or getattr(self.model, "end2end", False):
            raise ValueError("多标签候选要求五通道 NumPy 图像和 nms=True 的一对多检测头")
        rows = nms.non_max_suppression(
            preds, self.args.conf, self.args.iou, classes=self.args.classes,
            agnostic=self.args.agnostic_nms, multi_label=True, max_det=self.args.max_det,
            nc=len(self.model.names), end2end=False, rotated=False,
            max_time_img=2.0,  # 多类别低分候选较多，适度放宽后处理时间预算
        )
        return self.construct_results(rows, img, orig_imgs)

    def construct_result(self, pred: torch.Tensor, img: torch.Tensor, orig_img: np.ndarray, img_path: str) -> Results:
        """去掉 LetterBox 缩放和填充，返回原图坐标及可绘制的三通道图片。"""
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        visible: np.ndarray = np.ascontiguousarray(orig_img[:, :, :3][:, :, ::-1])
        return Results(visible, path=img_path, names=self.model.names, boxes=pred[:, :6])


def predict_image(model: YOLO, fused: np.ndarray, output: Path, device: str, imgsz: int, conf: float,
                  iou: float, multi_label: bool = False) -> Results:
    """逐张预测融合图，不积累整套测试集或让框架自行保存额外目录。

    Args:
        model: 已校验的五通道模型，循环中重复使用同一个实例。
        fused: fuse_modalities 生成的五通道原尺寸图像。
        output: 已独占创建的输出目录。
        device: 推理设备编号或 cpu。
        imgsz: LetterBox 目标尺寸。
        conf: 提交候选的最低置信度。
        iou: NMS 去重阈值。
        multi_label: 是否保留同一候选框超过阈值的多个类别，不进行多模型融合。

    Returns:
        包含原图坐标预测框与 BGR 可见光图的检测结果。
    """
    if model.predictor is not None:
        if not isinstance(model.predictor, MultimodalDetectionPredictor):
            raise ValueError("当前模型已绑定其他预测器，请重新加载五通道权重")
        model.predictor.multi_label = multi_label
    results: list[Results] = model.predict(
        predictor=partial(MultimodalDetectionPredictor, multi_label=multi_label),  # 开关由自定义预测器接收
        # 一、输入、检测与精度
        source=fused,  # 已同步融合的 RGB 3 + 红外 1 + 深度 1，不传单张 RGB 路径
        imgsz=imgsz,  # 默认 1280，与训练验证分辨率一致
        rect=True,  # 单图最小矩形填充，保持比例，减少无效像素
        conf=conf,  # 默认 0.001，为 AP 评测保留低分候选
        iou=iou,  # 默认 0.7，与训练验证的 NMS 一致
        nms=True,  # 明确使用带 NMS 的一对多头，与 v5 训练验证一致
        max_det=MAX_DETECTIONS,  # 官方单图最多 100 个框
        agnostic_nms=False,  # 不跨类别互相抑制，保留人与车等重叠目标
        classes=None,  # 预测全部 12 类
        augment=False,  # 不引入未经验证的 TTA 或多模型集成
        # 二、设备与性能
        device=device,  # 默认本机 CUDA:0；测试可显式指定 cpu
        quantize=32,  # FP32 推理基线，不使用 INT8 量化或未验证的精度压缩
        batch=1,  # 外层逐张融合，峰值内存不随测试集数量增长
        stream=False,  # 每次只传一张图，结果列表不会包含整个测试集
        compile=False,  # 避免 Windows 上首次编译开销和额外依赖
        channels_last=False,  # 沿用训练的 NCHW 布局
        verbose=False,  # 外层输出样本进度，不重复输出每张图的框架日志
        # 三、输出由本脚本统一管理
        project=str(output.parent),  # 不向训练 runs 目录写入预测产物
        name=output.name,  # 与用户指定的输出根目录一致
        save_dir=str(output),  # 固定位置；防覆盖由 prepare_output 独立保证
        exist_ok=True,  # 框架不另建 predict2；本脚本已经独占创建输出目录
        save=False,  # 手工保存前三通道的可见光带框图片
        save_txt=False,  # 手工排他写入，确保无目标也有空 TXT，避免框架追加写
        save_conf=True,  # 六列 TXT 必须含置信度，由 save_prediction 实际写出
        save_crop=False,  # 不创建额外 crops 文件夹
        show=False,  # 批量预测不弹出窗口
        show_labels=True,  # 图片中保留类别名
        show_conf=True,  # 图片中保留置信度
        show_boxes=True,  # 图片中绘制预测框
        line_width=None,  # 绘图随原图尺寸自动调整线宽
        # 四、其他输入和任务参数：当前静态图检测不使用
        data=None,  # 使用检查点名称，并在预测前核对完整类别顺序
        dnn=False,  # 本地 PyTorch 权重，不使用 ONNX OpenCV 后端
        vid_stride=1,  # 不使用视频抽帧
        stream_buffer=False,  # 不使用实时视频队列
        save_frames=False,  # 静态图无视频帧输出
        visualize=False,  # 不计算激活热图
        embed=None,  # 不输出特征向量
        retina_masks=False,  # 检测任务无分割掩码
    )
    return results[0]


# 三、标签、图片与提交打包
def prepare_output(output: Path) -> None:
    """独占创建输出根目录；即使已有目录为空也不复用。

    Raises:
        FileExistsError: 输出位置已经存在，需通过 --output 指定新位置。
    """
    if output.exists():
        raise FileExistsError(f"输出目录已存在，不会覆盖：{output}；请用 --output 指定新目录")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("images", "labels", "比赛提交内容"):
        (output / name).mkdir()


def validate_rows(rows: np.ndarray) -> None:
    """检查六列提交格式、类别、置信度及框的归一化边界。

    Raises:
        ValueError: 存在非法预测行、超限或未按置信度降序排列。
    """
    if rows.ndim != 2 or rows.shape[1] != 6 or len(rows) > MAX_DETECTIONS or not np.isfinite(rows).all():
        raise ValueError("预测标签必须为不超过 100 行的有限数值六列表格")
    if not len(rows):
        return
    classes, coords, scores = rows[:, 0], rows[:, 1:5], rows[:, 5]
    if np.any(classes != np.floor(classes)) or np.any((classes < 0) | (classes >= len(CLASS_NAMES))):
        raise ValueError("预测类别必须是 0 至 11 的整数")
    if np.any((coords < 0) | (coords > 1)) or np.any(coords[:, 2:] <= 0):
        raise ValueError("预测坐标必须归一化到 [0,1]，框宽高必须大于 0")
    if np.any(coords[:, :2] - coords[:, 2:] / 2 < -1e-7) or np.any(coords[:, :2] + coords[:, 2:] / 2 > 1 + 1e-7):
        raise ValueError("预测框边界超出原图")
    if np.any((scores < 0) | (scores > 1)) or np.any(np.diff(scores) > 0):
        raise ValueError("置信度必须位于 [0,1] 并按降序排列")


def prediction_rows(result: Results) -> np.ndarray:
    """以原图宽高归一化、过滤退化框，并按置信度保留前 100 项。

    Args:
        result: 已恢复到原图坐标的检测结果。

    Returns:
        class_id、cx、cy、w、h、confidence 六列数组，空预测形状为 (0, 6)。
    """
    if result.boxes is None:
        raise ValueError("检测结果缺少 boxes，不能作为有效的空检测处理")
    boxes = result.boxes.data.detach().cpu().numpy().astype(np.float64, copy=True)
    if boxes.shape[1] != 6 or not np.isfinite(boxes).all():
        raise ValueError("检测输出包含非有限数值或非六列检测框")
    height, width = result.orig_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    boxes = boxes[np.argsort(-boxes[:, 4], kind="stable")[:MAX_DETECTIONS]]
    xywh: np.ndarray = np.column_stack((
        (boxes[:, 0] + boxes[:, 2]) / (2 * width), (boxes[:, 1] + boxes[:, 3]) / (2 * height),
        (boxes[:, 2] - boxes[:, 0]) / width, (boxes[:, 3] - boxes[:, 1]) / height,
    ))
    rows: np.ndarray = np.column_stack((boxes[:, 5], xywh, boxes[:, 4]))
    validate_rows(rows)
    return rows


def save_prediction(result: Results, image_name: str, output: Path, visual_conf: float) -> None:
    """写入同名带框图片和六列标签；展示阈值不影响提交候选。

    Args:
        result: 已恢复原图坐标的检测结果。
        image_name: 配对清单中的原始图片文件名。
        output: 本次独占的输出根目录。
        visual_conf: 仅用于筛选展示框的置信度。

    Raises:
        FileExistsError: 同名结果已经存在。
        OSError: 图片无法编码或文件无法写入。
    """
    image_path, label_path = output / "images" / image_name, output / "labels" / f"{Path(image_name).stem}.txt"
    if image_path.exists() or label_path.exists():
        raise FileExistsError(f"预测结果已存在，拒绝覆盖：{image_name}")
    rows = prediction_rows(result)
    content: str = "".join(f"{int(row[0])} " + " ".join(f"{value:.10f}" for value in row[1:]) + "\n" for row in rows)
    # 绘图也使用实际提交的框，只有显示阈值不同；原始 Results 不做阈值修改。
    visible_rows = rows[rows[:, 5] >= visual_conf]
    height, width = result.orig_shape
    xyxy: np.ndarray = np.column_stack((
        (visible_rows[:, 1] - visible_rows[:, 3] / 2) * width, (visible_rows[:, 2] - visible_rows[:, 4] / 2) * height,
        (visible_rows[:, 1] + visible_rows[:, 3] / 2) * width, (visible_rows[:, 2] + visible_rows[:, 4] / 2) * height,
    ))
    display = Results(result.orig_img, path=image_name, names=result.names, boxes=np.column_stack((xyxy, visible_rows[:, 5], visible_rows[:, 0])))
    plotted: np.ndarray = display.plot(conf=True, labels=True, boxes=True, line_width=None, pil=False)
    success, encoded = cv2.imencode(image_path.suffix, plotted)
    if not success:
        raise OSError(f"预测图片编码失败：{image_path}")
    # 排他创建避免追加写和静默覆盖；空内容也会创建空 TXT。
    with label_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    with image_path.open("xb") as handle:
        handle.write(encoded.tobytes())


def build_submission(output: Path, image_names: list[str]) -> Path:
    """重新核验全部输出后生成仅含根目录 TXT 的 ZIP。

    Args:
        output: 预测结果根目录。
        image_names: 本次完整输入清单中的原始图片文件名。

    Returns:
        完成完整性校验的正式 submission.zip 路径。

    Note:
        打包中断仅留下 submission.zip.partial，不会留下冒充完整提交的 ZIP。
        Windows rename 不覆盖已有文件，符合本项目的防覆盖要求。
    """
    archive: Path = output / "比赛提交内容" / "submission.zip"
    if archive.exists():
        raise FileExistsError(f"提交包已存在，拒绝覆盖：{archive}")
    expected: set[str] = {f"{Path(name).stem}.txt" for name in image_names}
    if not expected or len(expected) != len(image_names):
        raise ValueError("提交清单为空或存在同名标签")
    if {p.name for p in (output / "labels").iterdir()} != expected:
        raise ValueError("标签文件数量或名称与待预测图片不一致，不能生成提交包")
    if {p.name for p in (output / "images").iterdir()} != set(image_names):
        raise ValueError("结果图片数量或名称与待预测图片不一致，不能生成提交包")
    # 将核验过的文本直接打包，不在核验后重新读取可能被外部修改的文件。
    contents: dict[str, str] = {}
    for name in sorted(expected):
        content = (output / "labels" / name).read_text(encoding="utf-8")
        lines = [line.split() for line in content.splitlines()]
        if any(len(line) != 6 or not line[0].isdigit() for line in lines):
            raise ValueError(f"标签不是整数类别开头的六列格式：{name}")
        validate_rows(np.array(lines, dtype=np.float64).reshape(-1, 6))
        contents[name] = content
    temporary: Path = archive.with_suffix(".zip.partial")
    with ZipFile(temporary, "x", compression=ZIP_DEFLATED) as zipped:
        for name, content in contents.items():
            zipped.writestr(name, content.encode("utf-8"))
    with ZipFile(temporary) as zipped:
        if zipped.testzip() is not None or set(zipped.namelist()) != expected:
            raise ValueError("提交 ZIP 完整性校验失败，保留临时文件供检查")
    temporary.rename(archive)
    return archive


# 四、命令行入口
def main(argv: list[str] | None = None) -> None:
    """检查本地路径和权重，完整预测测试集后输出可提交压缩包。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=MODEL_PATH, help="本地五通道 best.pt，不自动下载")
    parser.add_argument("--source", type=Path, default=SOURCE_PATH, help="含 visible/infrared/depth 的测试集根目录")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="必须尚不存在的输出根目录")
    parser.add_argument("--device", default="0", help="推理设备，默认 0，可指定 cpu")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="推理尺寸，默认 1280，必须为 32 的正整数倍")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="提交候选阈值，默认 0.001")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="NMS 阈值，默认 0.7")
    parser.add_argument("--multi-label", action=argparse.BooleanOptionalAction, default=MULTI_LABEL,
                        help="保留同框多类别候选；默认关闭，可能改善召回也可能增加误检")
    parser.add_argument("--visual-conf", type=float, default=VISUAL_CONF, help="仅图片显示阈值，默认 0.25")
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT, help="预期测试图数量，初赛为 1000")
    args = parser.parse_args(argv)
    if args.imgsz <= 0 or args.imgsz % 32 or args.expected_count <= 0:
        parser.error("imgsz 必须为 32 的正整数倍，expected-count 必须为正整数")
    if not all(0 <= value <= 1 for value in (args.conf, args.iou, args.visual_conf)):
        parser.error("conf、iou、visual-conf 必须位于 [0,1]")
    weights, source, output = ((PROJECT_ROOT / path).resolve() for path in (args.weights, args.source, args.output))
    if output.exists():
        raise FileExistsError(f"输出目录已存在，不会覆盖：{output}；请用 --output 指定新目录")
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise FileNotFoundError(f"请指定已有的本地 .pt 检测权重：{weights}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("预测输出与官方源数据目录不能互相包含")
    samples = collect_samples(source)
    if len(samples) != args.expected_count:
        raise ValueError(f"测试集应有 {args.expected_count} 组，实际找到 {len(samples)} 组；请检查 --source")
    with weights.open("rb") as handle:
        weight_hash: str = hashlib.file_digest(handle, "sha256").hexdigest()
    model: YOLO = YOLO(str(weights), task="detect")
    validate_model(model)
    prepare_output(output)
    print(f"预测权重：{weights}\n三模态配对完成，共 {len(samples)} 组；结果写入 {output}")
    for index, (visible, infrared, depth) in enumerate(samples, start=1):
        fused = fuse_modalities(visible, infrared, depth)
        result = predict_image(model, fused, output, args.device, args.imgsz, args.conf, args.iou, args.multi_label)
        save_prediction(result, visible.name, output, args.visual_conf)
        if index == 1 or index % 25 == 0 or index == len(samples):
            print(f"预测完成 {index}/{len(samples)}：{visible.name}")
    metadata: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(), "weights": weights.name,
        "run": weights.parent.parent.name, "weights_sha256": weight_hash, "source": source.name,
        "images": len(samples), "classes": list(CLASS_NAMES), "ultralytics": ultralytics.__version__,
        "torch": torch.__version__, "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
        "visual_conf": args.visual_conf, "nms": True, "multi_label": args.multi_label, "max_det": MAX_DETECTIONS,
        "quantize": 32, "rect": True, "device": args.device, "batch": 1, "augment": False,
    }
    with (output / "prediction.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    archive = build_submission(output, [sample[0].name for sample in samples])
    print(f"提交包校验通过，共 {len(samples)} 个 TXT：{archive}\n初赛只提交此 ZIP；images、prediction.json 不放入提交包。")


# 防止被测试或 Windows 子进程导入时意外执行整套预测。
if __name__ == "__main__":
    main()
