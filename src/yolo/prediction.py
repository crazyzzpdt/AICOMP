"""YOLO 三模态预测：检查点协议、五通道输入和检测后处理。"""
from __future__ import annotations

# 共用输出先设置离线环境，再导入模型框架。
from src.prediction_io import (OutputConfig, PredictionSample, parse_arguments, run_prediction)

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops

from src.modalities import FLOAT_PREPROCESS_VERSION, configure_fp32, read_float_modalities, letterbox_float
from src.yolo.aic.data import CLASS_NAMES, QUALITY_PREPROCESS_VERSION, fuse_modalities, fuse_quality_modalities
from src.yolo.aic.model import (EARLY_FUSION_VERSION, EVALUATION_PROTOCOL, FUSION_VERSION, SPLIT_STEM_VERSION, QUALITY_FUSION_VERSION,
    EarlyFusionDetectionModel, FusionDetectionModel, SplitModalStem, QualityFusionDetectionModel,
    canvas_shape, clip_canvas_boxes, letterbox_fused, letterbox_native_fused,
    letterbox_native_quality, restore_boxes, single_label_nms)

SOURCE_PATHS = ("predict1.py", "src/__init__.py", "src/prediction_io.py", "src/modalities.py",
                "src/yolo", "tools/__init__.py", "tools/prepare_dataset.py")


@dataclass(frozen=True)
class PredictionConfig(OutputConfig):
    """列出参考文档全部预测参数；不适用选项在入口说明并拒绝静默忽略。"""

    project: Path = Path("历史产出")
    name: str = "predict_yolo"
    imgsz: int | tuple[int, int] = 1280
    iou: float = 0.7
    rect: bool | None = None
    quantize: int = 32
    dnn: bool = False
    data: Path | None = None
    vid_stride: int = 1
    stream_buffer: bool = False
    visualize: bool = False
    augment: bool = False
    agnostic_nms: bool = False
    classes: list[int] | None = None
    retina_masks: bool = False
    embed: list[int] | None = None
    stream: bool = False
    compile: bool | str = False
    channels_last: bool = False
    nms: bool = True
    show: bool = False
    save_frames: bool = False
    save_crop: bool = False
    multi_label: bool = False
    yolo_profile: str = "v4"


class FusionPredictor:
    """单次加载融合模型，对同版本几何及通道契约的批次执行FP32推理。"""

    def __init__(self, model: FusionDetectionModel | EarlyFusionDetectionModel, device: str, content_hw: tuple[int, int]) -> None:
        early = isinstance(model, EarlyFusionDetectionModel)
        quality = isinstance(model, QualityFusionDetectionModel)
        version = getattr(model, "early_fusion_version" if early else "fusion_version", None)
        split_stem = early and isinstance(model.model[0], SplitModalStem)
        expected = QUALITY_FUSION_VERSION if quality else SPLIT_STEM_VERSION if split_stem else EARLY_FUSION_VERSION if early else FUSION_VERSION
        if version != expected or tuple(getattr(model, "content_hw", ())) != tuple(content_hw):
            raise ValueError(f"权重版本{version}、高宽{getattr(model, 'content_hw', None)}不匹配；请使用对应源码和预测尺寸")
        self.device = torch.device("cpu" if str(device) == "cpu" else f"cuda:{int(device)}")
        configure_fp32()
        self.model = model.to(self.device).float().eval()
        self.model.end2end = False
        self.names = model.names
        self.content_hw = content_hw
        self.version = version
        self.architecture = "quality_v19" if quality else "split_stem_v18" if split_stem else "early_fusion" if early else "gated_fusion"
        signature = getattr(model, "fusion_training", {}).get("signature", {})
        recipe = signature.get("recipe", {})
        self.geometry = recipe.get("geometry", "fixed_rect")
        if self.geometry not in {"native_square", "fixed_rect"}:
            raise ValueError(f"未知的训练预处理几何：{self.geometry}")
        if self.geometry == "native_square" and content_hw[0] != content_hw[1]:
            raise ValueError("原生方形权重要求height与imgsz一致")
        self.quality_preprocessing = signature.get("quality_preprocessing", {}) if quality else {}
        if quality and (getattr(model, "preprocess_version", None) != QUALITY_PREPROCESS_VERSION or
                self.quality_preprocessing.get("version") != QUALITY_PREPROCESS_VERSION or
                self.quality_preprocessing.get("input_channels") != 6 or
                signature.get("version") != QUALITY_FUSION_VERSION or
                recipe.get("architecture") != "quality_v19" or self.geometry != "native_square" or
                signature.get("evaluation_protocol") != EVALUATION_PROTOCOL):
            raise ValueError("v19检查点的六通道支持掩码/几何/验证协议不一致，拒绝套用旧预测路径")
        self.continuous_depth = getattr(model, "preprocess_version", None) == FLOAT_PREPROCESS_VERSION
        if bool(recipe.get("continuous_depth", False)) != self.continuous_depth:
            raise ValueError("浮点输入协议与训练签名不一致")
        if self.continuous_depth and (
                self.geometry != "native_square" or not early or split_stem or
                signature.get("float_preprocessing", {}).get("version") != FLOAT_PREPROCESS_VERSION):
            raise ValueError("连续浮点权重缺少一致的训练预处理签名，不能回退为旧8位输入")
        self.training_sensor_augmentation = recipe.get("sensors")
        self.input_geometry = "float_native" if self.continuous_depth else "quality_native" if quality else self.geometry
        self.padding_values = [114, 114, 114, 0, 0] if self.continuous_depth else [114, 114, 114, 0, 0, 0] if quality else [114] * 5 if self.geometry == "native_square" else [114, 114, 114, 0, 0]

    @torch.inference_mode()
    def predict_batch(self, images: torch.Tensor, targets: list[dict[str, torch.Tensor]],
                      conf: float, max_det: int, iou: float, config: PredictionConfig) -> list[dict[str, torch.Tensor]]:
        """只做一次模型前向，用训练验证相同的NMS与原图坐标还原。"""
        channels = 6 if self.architecture == "quality_v19" else 5
        dtype = torch.float32 if self.continuous_depth else torch.uint8
        if images.dtype != dtype or images.ndim != 4 or tuple(images.shape[1:]) != (channels, *canvas_shape(self.content_hw)):
            raise ValueError(f"融合预测要求{dtype}、{channels}通道及检查点对应画布")
        if not len(images) or len(images) != len(targets):
            raise ValueError("预测批次不能为空，且图像与坐标信息数量必须一致")
        # 不原地归一化CPU调用方的浮点输入，避免复用同一批时再次除255。
        images = images.to(self.device, non_blocking=True).float() / 255
        if config.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=self.device.type, enabled=False):
            predictions = self.model(images)
            if config.classes is None and not config.agnostic_nms:
                rows = single_label_nms(predictions, conf, iou, max_det)
            else:
                rows = nms.non_max_suppression(
                    predictions, conf, iou, nc=12, classes=config.classes,
                    multi_label=False, agnostic=config.agnostic_nms, max_det=max_det,
                    end2end=False, max_time_img=float("inf"),
                )
        outputs: list[dict[str, torch.Tensor]] = []
        for row, target in zip(rows, targets, strict=True):
            width, height = target["orig_size"].tolist()
            geometry = tuple(target["geometry"].tolist())
            boxes, keep = clip_canvas_boxes(row[:, :4], geometry, (height, width))
            boxes = restore_boxes(boxes, geometry, (height, width))
            outputs.append({"boxes": boxes, "scores": row[keep, 4], "labels": row[keep, 5].long()})
        return outputs


def validate_model(model: YOLO) -> None:
    """只接受本项目的三模态、12类本地检测权重，包括v19派生支持通道。

    Args:
        model: 已从本地检查点加载的 YOLO 模型。

    Raises:
        ValueError: 任务、首层通道数或类别编号不符。
    """
    if getattr(model.model, "diagnostic_only", False):
        raise ValueError("v20 RGB诊断权重不是三模态赛事模型，不能生成赛事提交结果")
    first_conv = next((layer for layer in model.model.modules() if isinstance(layer, torch.nn.Conv2d)), None)
    split_stem = (isinstance(model.model, EarlyFusionDetectionModel) and
                  getattr(model.model, "early_fusion_version", None) == SPLIT_STEM_VERSION and
                  isinstance(model.model.model[0], SplitModalStem))
    if model.task != "detect" or first_conv is None or (
            first_conv.in_channels != 5 and not isinstance(model.model, FusionDetectionModel) and not split_stem):
        raise ValueError("必须使用训练后的五通道检测权重，不能使用 orgin_models 中的 RGB 预训练权重")
    if model.names != dict(enumerate(CLASS_NAMES)):
        raise ValueError(f"模型类别编号与比赛 12 类不一致：{model.names}")
    # 框架图片加载器从 YAML 读取通道数，缺省的 3 会把五通道数组裁掉后两通道。
    model.model.yaml["channels"] = 6 if isinstance(model.model, QualityFusionDetectionModel) else 5


# 三、五通道预测与原图坐标恢复
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
            默认沿用框架单标签筛选。多标签只保留模型已给出的分数，不人为抬高
            ball 置信度；低分候选可能争用每图 100 框的名额，效果需实际提交确认。
        """
        if not isinstance(orig_imgs, list) or getattr(self.model, "end2end", False):
            raise ValueError("批量候选要求五通道 NumPy 图像和 nms=True 的一对多检测头")
        rows = nms.non_max_suppression(
            preds, self.args.conf, self.args.iou, classes=self.args.classes,
            agnostic=self.args.agnostic_nms, multi_label=self.multi_label, max_det=self.args.max_det,
            nc=len(self.model.names), end2end=False, rotated=False,
            max_time_img=float("inf"),  # 离线提交必须处理完整批次，不因实时场景的超时预算跳过后续图片
        )
        return self.construct_results(rows, img, orig_imgs)

    def construct_result(self, pred: torch.Tensor, img: torch.Tensor, orig_img: np.ndarray, img_path: str) -> Results:
        """去掉 LetterBox 缩放和填充，返回原图坐标及可绘制的三通道图片。"""
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        visible: np.ndarray = np.ascontiguousarray(orig_img[:, :, :3][:, :, ::-1])
        return Results(visible, path=img_path, names=self.model.names, boxes=pred[:, :6])


class V4MultimodalDetectionPredictor(MultimodalDetectionPredictor):
    """恢复历史提交 86411da 的 YOLO 原生单标签后处理。

    Note:
        五通道检查与原图坐标恢复沿用相同实现；原生后处理由框架执行。
        该路径用于保留历史 YOLO 五通道权重的原生后处理。
    """

    def postprocess(self, preds: torch.Tensor, img: torch.Tensor, orig_imgs: list[np.ndarray], **kwargs: object) -> list[Results]:
        """沿用 v4 时未重写的框架后处理，保留原生单标签 NMS。"""
        return DetectionPredictor.postprocess(self, preds, img, orig_imgs, **kwargs)


def predict_yolo_batch(model: YOLO, images: list[np.ndarray], config: PredictionConfig) -> list[Results]:
    """调用历史五通道预测器，输出统一交由共用保存层处理。"""
    if not images or len({image.shape for image in images}) != 1:
        raise ValueError("旧YOLO批量输入须为相同原图尺寸")
    legacy = config.yolo_profile == "v4"
    if legacy and (len(images) != 1 or config.multi_label):
        raise ValueError("v4配方要求逐张输入及单标签后处理")
    predictor_class = V4MultimodalDetectionPredictor if legacy else MultimodalDetectionPredictor
    if model.predictor is not None:
        if type(model.predictor) is not predictor_class:
            raise ValueError("当前模型已绑定其他预测器，请重新加载权重")
        model.predictor.multi_label = config.multi_label
    results = model.predict(
        predictor=partial(predictor_class, multi_label=config.multi_label),
        source=images[0] if legacy else images,
        imgsz=config.imgsz if config.height == config.imgsz else (config.height, config.imgsz),
        rect=True if config.rect is None else config.rect,
        conf=config.conf, iou=config.iou, nms=config.nms, max_det=config.max_det,
        agnostic_nms=config.agnostic_nms, classes=config.classes, augment=config.augment,
        device=config.device, quantize=config.quantize, batch=len(images), stream=config.stream,
        compile=config.compile, channels_last=config.channels_last, verbose=config.verbose,
        # 框架只前向和后处理，避免与后台图片/TXT保存重复写入。
        project=str(config.output.parent), name=config.output.name, save_dir=str(config.output),
        exist_ok=True, save=False, save_txt=False, save_crop=False, show=False,
        data=config.data, dnn=config.dnn, vid_stride=config.vid_stride,
        stream_buffer=config.stream_buffer, save_frames=config.save_frames,
        visualize=config.visualize, embed=config.embed, retina_masks=config.retina_masks,
    )
    return list(results)


def load_prediction_sample(paths: tuple[Path, Path, Path], backend: str, imgsz: int, height: int = 1080,
                           fusion_geometry: str = "fixed_rect") -> PredictionSample:
    """在读取线程中融合模态，并完成相应后端的同步缩放与填充。"""
    quality = backend == "fusion" and fusion_geometry == "quality_native"
    floating = fusion_geometry == "float_native"
    fused = read_float_modalities(*paths)[0] if floating else fuse_quality_modalities(*paths) if quality else fuse_modalities(*paths)
    channels = 6 if quality else 5
    expected_dtype = np.float32 if floating else np.uint8
    if fused.dtype != expected_dtype or fused.ndim != 3 or fused.shape[2] != channels:
        raise ValueError(f"预测输入不符合{expected_dtype}/{channels}通道：{paths[0].name}")
    if backend == "yolo":
        return PredictionSample(paths[0], fused, None, {})
    if floating:
        canvas, geometry = letterbox_float(fused, imgsz, native=fusion_geometry == "float_native")
    else:
        canvas, geometry = (letterbox_native_quality(fused, imgsz) if quality else
                            letterbox_native_fused(fused, imgsz) if fusion_geometry == "native_square"
                            else letterbox_fused(fused, (height, imgsz)))
    original_height, width = fused.shape[:2]
    # 仅绘图副本转8位，模型输入canvas保持原协议精度。
    visible = np.ascontiguousarray(fused[:, :, :3][:, :, ::-1]).astype(np.uint8)
    target = {"orig_size": torch.tensor([width, original_height]),
              "geometry": torch.tensor(geometry, dtype=torch.float64 if backend == "fusion" else torch.float32)}
    # 新协议CPU预取也是FP32；旧协议仍保留uint8读取兼容。
    return PredictionSample(paths[0], np.ascontiguousarray(canvas.transpose(2, 0, 1)), visible, target)


def predict_batch(model: YOLO | FusionPredictor, batch: list[PredictionSample],
                  config: PredictionConfig, fusion: bool) -> list[Results]:
    """只在主线程调用 GPU，将结果转回 CPU 后交给后台绘图线程。"""
    if not fusion:
        if config.yolo_profile == "v4":
            return [predict_yolo_batch(model, [sample.image], config)[0].cpu() for sample in batch]
        return [result.cpu() for result in predict_yolo_batch(model, [sample.image for sample in batch], config)]
    images = torch.from_numpy(np.stack([sample.image for sample in batch]))
    if config.pin_memory and model.device.type == "cuda":
        images = images.pin_memory()
    predictions = model.predict_batch(images, [sample.target for sample in batch], config.conf, config.max_det, config.iou, config)
    results: list[Results] = []
    for sample, prediction in zip(batch, predictions, strict=True):
        boxes = torch.cat((prediction["boxes"], prediction["scores"][:, None], prediction["labels"][:, None].float()), dim=1).cpu()
        results.append(Results(sample.visible, path=str(sample.path), names=model.names, boxes=boxes))
    return results


def create_backend(config: PredictionConfig):
    """加载一个YOLO检查点，返回读图、批量前向与本次真实执行信息。"""
    model = YOLO(str(config.weights), task="detect")
    validate_model(model)
    fusion = isinstance(model.model, (FusionDetectionModel, EarlyFusionDetectionModel))
    metadata = {
        "backend": "fusion" if fusion else "yolo", "nms": True, "iou": config.iou,
        "multi_label": config.multi_label, "agnostic_nms": config.agnostic_nms,
        "classes_filter": config.classes, "channels_last": config.channels_last, "stream": config.stream,
        "rect": True if config.rect is None else config.rect, "yolo_profile": config.yolo_profile,
        "effective_model_batch": 1 if not fusion and config.yolo_profile == "v4" else config.batch,
    }
    if fusion:
        if config.rect is True or config.multi_label:
            raise ValueError("融合权重保持训练画布及单标签验证口径，请用rect=False/None、multi_label=False")
        epoch = int(model.ckpt["epoch"]) + 1
        model = FusionPredictor(model.model, config.device, (config.height, config.imgsz))
        if config.channels_last:
            model.model.to(memory_format=torch.channels_last)
        metadata.update({
            "fusion_version": model.version, "architecture": model.architecture,
            "content_hw": list(model.content_hw), "tensor_hw": list(canvas_shape(model.content_hw)),
            "evaluation_protocol": EVALUATION_PROTOCOL, "epoch": epoch, "weights_kind": "ema",
            "yolo_profile": None, "postprocess": "shared_single_label_nms" if
                config.classes is None and not config.agnostic_nms else "configured_single_label_nms",
            "rect": False, "validation_geometry": model.geometry, "padding_values": model.padding_values,
            "preprocess": FLOAT_PREPROCESS_VERSION if model.continuous_depth else
                QUALITY_PREPROCESS_VERSION if model.architecture == "quality_v19" else
                "native_square_ceil_pad114_div255_v1" if model.geometry == "native_square" else
                "fixed_rectangle_rgbirdepth_div255",
            "quality_preprocessing": model.quality_preprocessing,
        })
    else:
        if config.yolo_profile == "v4" and config.multi_label:
            raise ValueError("v4采用单标签；如需旧模型多标签候选，请显式选择yolo_profile=current")
        metadata.update({
            "postprocess": "framework_native_single_label_nms" if config.yolo_profile == "v4" else "custom_nms",
            "legacy_source_commit": "86411da" if config.yolo_profile == "v4" else None,
        })
    floating = bool(getattr(model, "continuous_depth", False))
    metadata.update({
        "input_dtype": "float32" if floating else "uint8", "continuous_modalities": floating,
        "training_sensor_augmentation": getattr(model, "training_sensor_augmentation", None),
        "pin_memory": config.pin_memory and fusion and str(config.device) != "cpu",
    })
    loader = partial(load_prediction_sample, backend="fusion" if fusion else "yolo",
                     imgsz=config.imgsz, height=config.height,
                     fusion_geometry=getattr(model, "input_geometry", "fixed_rect"))
    forward = partial(predict_batch, model, config=config, fusion=fusion)
    return loader, forward, metadata


def predict(config: PredictionConfig, argv: list[str] | None = None) -> None:
    """执行纯YOLO预测；所有参数先校验，再读取模型和创建输出。"""
    if argv is not None:
        config = parse_arguments(config, argv)
    if config.weights.suffix.lower() != ".pt":
        raise ValueError("predict1.py仅接收本项目YOLO .pt权重")
    if isinstance(config.imgsz, (tuple, list)):
        if len(config.imgsz) != 2:
            raise ValueError("imgsz元组须按(height, width)填写")
        config = replace(config, height=config.imgsz[0], imgsz=config.imgsz[1])
    if config.output is None:
        if not config.name or Path(config.name).name != config.name or config.name in {".", ".."}:
            raise ValueError("name须为单个目录名；完整输出路径使用--output")
        config = replace(config, output=config.project / config.name)
    if not 0 <= config.iou <= 1 or config.yolo_profile not in {"v4", "current"}:
        raise ValueError("iou须在[0,1]；yolo_profile须为v4或current")
    if config.classes is not None and (not config.classes or any(type(c) is not int or not 0 <= c < 12 for c in config.classes)):
        raise ValueError("classes须为0–11的非空类别列表或None")
    # 当前交付是本地五通道PT检测，不把RGB专用或其他任务参数静默传给自定义前向。
    required = {
        "quantize": 32, "nms": True, "dnn": False, "data": None,
        "vid_stride": 1, "stream_buffer": False, "save_frames": False,
        "retina_masks": False, "embed": None, "visualize": False,
        "augment": False, "compile": False, "show": False, "save_crop": False,
    }
    for name, expected in required.items():
        if getattr(config, name) != expected:
            raise ValueError(f"当前三模态赛事预测要求{name}={expected!r}；该选项不适用于本执行路径")
    run_prediction(config, create_backend, SOURCE_PATHS, "predict1.py")
