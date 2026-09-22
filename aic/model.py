"""提供YOLO26l质量感知融合、历史五通道结构及共用几何处理。

模型类别保持可导入，保证本地检查点在 Windows 子进程与 predict.py 中恢复。
只有正式训练或预测入口会执行前向，本模块导入不加载权重或启动训练。
"""

from __future__ import annotations

# 内置库
import sys
from copy import deepcopy
from typing import Any

# 三方库
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.data.augment import LetterBox
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import nms
from ultralytics.utils.loss import E2ELoss

# 自己的模块
from .data import QUALITY_PREPROCESS_VERSION, resize_quality_image


# 结构和预处理共同版本化，不把旧五通道首层模型当作新分支模型。
FUSION_VERSION: str = "yolo26l_gated_rect_v9_1"
# 全尺寸v10与上一版1280候选区分，禁止预测时静默混用预处理。
EARLY_FUSION_VERSION: str = "yolo26l_early_rect_v10_2"
# 首层参数路径改变，须与旧五通道整块卷积区分恢复。
SPLIT_STEM_VERSION: str = "yolo26l_split_stem_v18_1"
# v19输入额外携带深度支持比例，不能用旧五通道预处理预测。
QUALITY_FUSION_VERSION: str = "yolo26l_quality_p3_v19_1"
# 单模态仅用于训练方向诊断，与所有赛事三模态检查点明确区分。
RGB_DIAGNOSTIC_VERSION: str = "yolo26l_rgb_diagnostic_v20_1"
# 协议变化不冒充模型提升；新旧权重须在同一划分、同一协议下比较。
EVALUATION_PROTOCOL: str = "fp32_single_label_content_clip_v2"
# 五通道顺序沿用已有数据集；真实内容框宽高与网络补齐画布分别记录。
DEFAULT_CONTENT_HW: tuple[int, int] = (1080, 1920)
# YOLO26l P3、P4、P5 在官方主干的层号，不改动其参数路径。
FUSION_LAYERS: tuple[int, int, int] = (4, 6, 10)


def canvas_shape(content_hw: tuple[int, int]) -> tuple[int, int]:
    """将内容窗口向上补齐为最大步长 32 的整数倍。"""
    if len(content_hw) != 2 or min(content_hw) <= 0:
        raise ValueError("输入尺寸必须为正整数 (高, 宽)")
    return tuple((int(side) + 31) // 32 * 32 for side in content_hw)


def resize_channels(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """逐通道缩放，避开 OpenCV 部分插值路径最多四通道的限制。"""
    return np.stack([cv2.resize(image[:, :, index], size_wh, interpolation=cv2.INTER_LINEAR)
                     for index in range(image.shape[2])], axis=-1)


def resize_native_fused(image: np.ndarray, imgsz: int) -> np.ndarray:
    """复用原生数据集的长边缩放及向上取整，训练和预测不各用一套取整规则。"""
    height, width = image.shape[:2]
    ratio = imgsz / max(height, width)
    if ratio == 1:
        return image
    size = (min(int(np.ceil(width * ratio)), imgsz), min(int(np.ceil(height * ratio)), imgsz))
    return resize_channels(image, size)


def letterbox_native_fused(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, tuple[float, float, int, int]]:
    """按原生验证链缩放并对五通道全部填114，不混用固定矩形的辅助通道零补边。"""
    if image.ndim != 3 or image.shape[2] != 5 or image.dtype != np.uint8:
        raise ValueError("原生融合输入必须是 uint8 RGB3+IR1+Depth1")
    height, width = image.shape[:2]
    resized = resize_native_fused(image, imgsz)
    new_height, new_width = resized.shape[:2]
    canvas = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=False, padding_value=114)(image=resized)
    return canvas, (new_width / width, new_height / height, (imgsz - new_width) // 2, (imgsz - new_height) // 2)


class QualityLetterBox(LetterBox):
    """复用原生标签几何，只有v19图像插值和辅助补边改为质量感知处理。"""

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = resize_quality_image(labels["img"], tuple(params["new_unpad"]))
        height, width = image.shape[:2]
        top, left = params["top"], params["left"]
        canvas = np.zeros((height + top + params["bottom"], width + left + params["right"], 6), dtype=np.uint8)
        canvas[:, :, :3] = 114
        canvas[top:top + height, left:left + width] = image
        labels["img"], labels["resized_shape"] = canvas, params["new_shape"]
        return labels


def resize_native_quality(image: np.ndarray, imgsz: int) -> np.ndarray:
    """v19沿用原生长边与ceil取整，只改变深度插值语义。"""
    height, width = image.shape[:2]
    ratio = imgsz / max(height, width)
    return resize_quality_image(image, (min(int(np.ceil(width * ratio)), imgsz), min(int(np.ceil(height * ratio)), imgsz)))


def letterbox_native_quality(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, tuple[float, float, int, int]]:
    """预测复用v19数据集相同的长边缩放与LetterBox。"""
    height, width = image.shape[:2]
    resized = resize_native_quality(image, imgsz)
    new_height, new_width = resized.shape[:2]
    canvas = QualityLetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=False)(image=resized)
    return canvas, (new_width / width, new_height / height, (imgsz - new_width) // 2, (imgsz - new_height) // 2)


def letterbox_fused(image: np.ndarray, content_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[float, float, int, int]]:
    """等比放入内容窗口后补齐步长，返回精确到取整后尺寸的缩放与偏移。

    Returns:
        uint8 五通道画布和 (x缩放, y缩放, 左填充, 上填充)。
    """
    if image.ndim != 3 or image.shape[2] != 5 or image.dtype != np.uint8:
        raise ValueError("矩形融合输入必须是 uint8 RGB3+IR1+Depth1")
    h, w = image.shape[:2]
    content_h, content_w = content_hw
    out_h, out_w = canvas_shape(content_hw)
    scale = min(content_h / h, content_w / w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image if (new_w, new_h) == (w, h) else resize_channels(image, (new_w, new_h))
    left, top = (out_w - new_w) // 2, (out_h - new_h) // 2
    canvas = np.zeros((out_h, out_w, 5), dtype=np.uint8)
    canvas[:, :, :3] = 114
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, (new_w / w, new_h / h, left, top)


def restore_boxes(boxes: torch.Tensor, geometry: tuple[float, float, int, int],
                  original_hw: tuple[int, int]) -> torch.Tensor:
    """按实际缩放尺寸逆变换回原图，避免缩放取整带来的坐标误差。"""
    boxes = boxes.clone()
    sx, sy, left, top = geometry
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / sx
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / sy
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_hw[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_hw[0])
    return boxes


def clip_canvas_boxes(boxes: torch.Tensor, geometry: tuple[float, float, int, int],
                      original_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """验证和提交都裁回真实内容区，去除补边区域里的退化框。"""
    boxes = boxes.clone()
    sx, sy, left, top = geometry
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(left, left + original_hw[1] * sx)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(top, top + original_hw[0] * sy)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep], keep


def single_label_nms(predictions: Any, conf: float, iou: float, max_det: int) -> list[torch.Tensor]:
    """验证和提交共享同一一对多、单标签 NMS，离线批次不受实时超时截断。"""
    return nms.non_max_suppression(predictions, conf, iou, nc=12, multi_label=False,
                                   agnostic=False, max_det=max_det, end2end=False,
                                   max_time_img=float("inf"))


class AuxiliaryEncoder(nn.Module):
    """用小通道卷积提取单模态三尺度特征，小批次采用 GroupNorm。"""

    def __init__(self) -> None:
        super().__init__()
        channels = (1, 8, 16, 32, 64, 128)
        self.stages = nn.ModuleList(nn.Sequential(
            nn.Conv2d(source, target, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, target), nn.SiLU(),
        ) for source, target in zip(channels[:-1], channels[1:]))

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for index, layer in enumerate(self.stages):
            image = layer(image)
            if index >= 2:
                features.append(image)
        return features


class GatedResidual(nn.Module):
    """根据 RGB 与辅助模态内容学习空间门控，不固定清零任何模态。"""

    def __init__(self, aux_channels: int, rgb_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(aux_channels, rgb_channels, 1, bias=False)
        self.rgb_context = nn.Conv2d(rgb_channels, 8, 1)
        self.aux_context = nn.Conv2d(aux_channels, 8, 1)
        self.gate = nn.Sequential(nn.SiLU(), nn.Conv2d(16, 1, 1), nn.Sigmoid())
        # 非零小残差保护初始 RGB 响应，同时使编码器和门控都能获得梯度。
        nn.init.normal_(self.project.weight, std=0.001)
        nn.init.constant_(self.gate[1].bias, -2.0)
        self.last_gate_mean: torch.Tensor | None = None

    def forward(self, rgb: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat((self.rgb_context(rgb), self.aux_context(auxiliary)), dim=1))
        if self.training:
            self.last_gate_mean = gate.detach().mean()
        return gate * self.project(auxiliary)


class FixedHeadLoss(E2ELoss):
    """固定双头 0.8/0.2，并从同次前向累计两头损失供轮末记录。"""

    def __init__(self, model: DetectionModel) -> None:
        super().__init__(model)
        self.running: dict[str, torch.Tensor] = {}
        self.batches: int = 0

    def update(self) -> None:
        self.updates += 1
        self.o2m, self.o2o = 0.8, 0.2

    def __call__(self, preds: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parsed = self.one2many.parse_output(preds)
        many, one = self.one2many.loss(parsed["one2many"], batch), self.one2one.loss(parsed["one2one"], batch)
        if torch.is_grad_enabled():
            for prefix, items in (("one2many", many[1]), ("one2one", one[1])):
                for key, value in items.items():
                    name = f"{prefix}/{key}"
                    self.running[name] = self.running.get(name, torch.zeros_like(value)) + value.detach()
            self.batches += 1
        return many[0] * 0.8 + one[0] * 0.2, many[1]


class SplitModalStem(nn.Module):
    """拆分首层可训练参数，在共享BN前相加，不增加辅助骨干。"""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        conv = source.conv
        if conv.in_channels != 5 or conv.groups != 1 or conv.bias is not None:
            raise ValueError("分模态首层要求五通道、groups=1且无偏置")
        for name, start, stop in (("rgb", 0, 3), ("infrared", 3, 4), ("depth", 4, 5)):
            # 不额外随机初始化，避免拆分动作推进全局随机状态。
            branch = deepcopy(conv)
            branch.in_channels = stop - start
            branch.weight = nn.Parameter(conv.weight[:, start:stop].detach().clone())
            setattr(self, name, branch)
        self.bn, self.act = source.bn, source.act
        for key in ("i", "f", "type", "np"):
            setattr(self, key, getattr(source, key))
        self.capture_response: bool = False
        self.response_rms: dict[str, torch.Tensor] = {}

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """三路卷积求和后共享归一化，诊断只复用每轮首个训练批次。"""
        if image.ndim != 4 or image.shape[1] != 5:
            raise ValueError("v18首层要求NCHW五通道RGB3+IR1+Depth1输入")
        rgb = self.rgb(image[:, :3])
        infrared = self.infrared(image[:, 3:4])
        depth = self.depth(image[:, 4:5])
        combined = rgb + infrared + depth
        if self.training and self.capture_response and torch.is_grad_enabled():
            self.response_rms = {name: value.detach()[:, :, ::8, ::8].float().square().mean().sqrt()
                                 for name, value in (("rgb", rgb), ("infrared", infrared),
                                                     ("depth", depth), ("sum", combined))}
            self.capture_response = False
        return self.act(self.bn(combined))


class EarlyFusionDetectionModel(DetectionModel):
    """从第一层共同学习三模态，保留官方YOLO主干和双检测头。"""

    def __init__(self, cfg: dict[str, Any], nc: int = 12, verbose: bool = True) -> None:
        super().__init__(deepcopy(cfg), ch=5, nc=nc, verbose=verbose)
        if self.yaml.get("scale") != "l" or len(self.model) != 24:
            raise ValueError("v10配方使用官方YOLO26l，不自动换模型规模")
        self.early_fusion_version = EARLY_FUSION_VERSION
        self.content_hw = DEFAULT_CONTENT_HW
        self.end2end = False
        # 只在初始化时置零新增卷积切片，保护RGB迁移；不清零输入、不冻结辅助通道。
        with torch.no_grad():
            self.model[0].conv.weight[:, 3:].zero_()

    def init_criterion(self) -> FixedHeadLoss:
        """保持一对多头训练份额，避免缩短epochs后改变v4的实际双头配比。"""
        return FixedHeadLoss(self)

    def split_modal_stem(self) -> None:
        """在官方迁移后拆分首层，保留已迁移RGB及新增通道初值。"""
        if isinstance(self.model[0], SplitModalStem):
            raise ValueError("首层已拆分，不能重复转换")
        self.model[0] = SplitModalStem(self.model[0])
        self.early_fusion_version = SPLIT_STEM_VERSION


class RGBDiagnosticModel(EarlyFusionDetectionModel):
    """仅用于RGB对照，不作为三模态赛事提交模型。"""

    def __init__(self, cfg: dict[str, Any], nc: int = 12, verbose: bool = True) -> None:
        # 先沿用v14构建顺序，再裁掉辅助切片，避免改变未迁移检测头的随机初始化顺序。
        super().__init__(cfg, nc, verbose)
        first = self.model[0].conv
        first.weight = nn.Parameter(first.weight.detach()[:, :3].clone())
        first.in_channels = 3
        self.model[0].np = sum(parameter.numel() for parameter in self.model[0].parameters())
        self.yaml["channels"] = 3
        self.early_fusion_version = RGB_DIAGNOSTIC_VERSION
        self.diagnostic_only = True


class FusionDetectionModel(DetectionModel):
    """保持原 YOLO26l RGB 参数路径，在颈部前融合 IR/Depth 特征。"""

    def __init__(self, cfg: dict[str, Any], nc: int = 12, verbose: bool = True) -> None:
        super().__init__(deepcopy(cfg), ch=3, nc=nc, verbose=verbose)
        if self.yaml.get("scale") != "l" or len(self.model) != 24:
            raise ValueError("当前融合结构限定官方 YOLO26l 的 24 个顶层模块")
        self.ir_encoder = AuxiliaryEncoder()
        self.depth_encoder = AuxiliaryEncoder()
        self.ir_fusion = nn.ModuleList(GatedResidual(c, 512) for c in (32, 64, 128))
        self.depth_fusion = nn.ModuleList(GatedResidual(c, 512) for c in (32, 64, 128))
        self.fusion_version = FUSION_VERSION
        self.content_hw = DEFAULT_CONTENT_HW
        self.yaml["channels"] = 5
        self.end2end = False

    def _predict_once(self, x: torch.Tensor, profile: bool = False,
                      embed: list[int] | None = None) -> Any:
        """RGB主干独立前向后替换颈部输入，辅助特征不污染后续RGB主干。"""
        # 父类构建时用三通道小张量推导步长，此时辅助分支尚未建立。
        if not hasattr(self, "ir_encoder"):
            # 本机Ultralytics接口为(x, profile, embed)，不再接受visualize参数。
            return super()._predict_once(x, profile=profile, embed=embed)
        if x.shape[1] != 5 or embed is not None or profile:
            raise ValueError("融合模型要求五通道常规前向，不支持 embed/profile")
        infrared, depth = self.ir_encoder(x[:, 3:4]), self.depth_encoder(x[:, 4:5])
        x = x[:, :3]
        saved: list[Any] = []
        for layer in self.model:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            saved.append(x if layer.i in self.save else None)
            if layer.i == 10:
                for index, slot in enumerate(FUSION_LAYERS):
                    rgb = saved[slot]
                    saved[slot] = rgb + self.ir_fusion[index](rgb, infrared[index]) + self.depth_fusion[index](rgb, depth[index])
                x = saved[10]
        return x

    def init_criterion(self) -> FixedHeadLoss:
        """双头训练与选择哪一个预测头相互独立。"""
        return FixedHeadLoss(self)


class LocalInfraredResidual(nn.Module):
    """在P3的3×3邻域匹配红外特征，不做全图注意力或全图平移。"""

    def __init__(self) -> None:
        super().__init__()
        # RGB/IR都来自同一预训练前五层，用共享投影比较局部结构。
        self.context = nn.Conv2d(512, 8, 1, bias=False)
        self.value = nn.Conv2d(512, 64, 1, bias=False)
        self.project = nn.Conv2d(64, 512, 1, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())
        nn.init.normal_(self.project.weight, std=0.001)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -2.0)
        self.last_gate_mean: torch.Tensor | None = None
        self.last_center_weight: torch.Tensor | None = None

    def forward(self, rgb: torch.Tensor, infrared: torch.Tensor) -> torch.Tensor:
        """软对应只在一个特征格邻域内，边界候选显式屏蔽而非循环回绕。"""
        query, key = self.context(rgb), self.context(infrared)
        height, width = query.shape[-2:]
        query_unit, key_unit = F.normalize(query.float(), dim=1), F.normalize(key.float(), dim=1)
        neighbors = F.unfold(key_unit, kernel_size=3, padding=1).reshape(rgb.shape[0], 8, 9, height, width)
        logits = (query_unit.unsqueeze(2) * neighbors).sum(1)
        # 同位置有固定先验；偏移只能±1个P3格（输入8像素），不宣称完成传感器标定。
        prior = logits.new_zeros((1, 9, 1, 1))
        prior[:, 4] = 2.0
        valid = F.unfold(torch.ones_like(key_unit[:1, :1]), kernel_size=3, padding=1).reshape(1, 9, height, width)
        attention = (logits + prior).masked_fill(valid == 0, -1e4).softmax(dim=1)
        values = self.value(infrared)
        padded = F.pad(values, (1, 1, 1, 1))
        matched = torch.zeros_like(values)
        # 逐偏移累加，避免展开64通道×9邻域的大张量。
        for index in range(9):
            dy, dx = divmod(index, 3)
            matched = matched + padded[:, :, dy:dy + height, dx:dx + width] * attention[:, index:index + 1].to(values.dtype)
        gate = self.gate(torch.cat((query, key), dim=1))
        if self.training:
            self.last_gate_mean = gate.detach().float().mean()
            self.last_center_weight = attention[:, 4].detach().mean()
        return self.project(matched) * gate


class SupportedDepthEncoder(nn.Module):
    """把距离与支持比例共同编码为P3特征，缺测不伪装成近距离。"""

    def __init__(self) -> None:
        super().__init__()
        channels = (2, 32, 48, 64)
        self.stages = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(source, target, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, target), nn.SiLU(),
        ) for source, target in zip(channels[:-1], channels[1:])])

    def forward(self, depth_and_support: torch.Tensor) -> torch.Tensor:
        return self.stages(depth_and_support)


class QualityFusionDetectionModel(FusionDetectionModel):
    """v19保留RGB定位主干，仅在进入颈部的P3加入质量感知三模态残差。"""

    def __init__(self, cfg: dict[str, Any], nc: int = 12, verbose: bool = True) -> None:
        # 不构建历史v9随机三尺度分支；初始化步长仍由父类RGB路径推导。
        DetectionModel.__init__(self, deepcopy(cfg), ch=3, nc=nc, verbose=verbose)
        if self.yaml.get("scale") != "l" or len(self.model) != 24:
            raise ValueError("v19只支持官方YOLO26l的24层结构")
        self.ir_encoder = deepcopy(self.model[:5])
        first = self.ir_encoder[0].conv
        first.in_channels = 1
        first.weight = nn.Parameter(first.weight.detach().sum(dim=1, keepdim=True))
        self.depth_encoder = SupportedDepthEncoder()
        self.ir_fusion = LocalInfraredResidual()
        self.depth_fusion = GatedResidual(64, 512)
        self.fusion_version = QUALITY_FUSION_VERSION
        self.preprocess_version = QUALITY_PREPROCESS_VERSION
        self.content_hw = (1280, 1280)
        self.yaml["channels"] = 6
        self.end2end = False
        self.last_support_mean: torch.Tensor | None = None

    def initialize_auxiliary_from_rgb(self) -> None:
        """官方RGB迁移完成后复制兼容参数，不把随机构建参数当作预训练。

        Note:
            IR第0–4层除首层通道求和外完整继承RGB；Depth只继承首层前32个
            RGB滤波器的通道和，支持通道权重置零可训练，其余深度层随机初始化。
        """
        state = {name: value.detach().clone() for name, value in self.model[:5].state_dict().items()}
        state["0.conv.weight"] = state["0.conv.weight"].sum(dim=1, keepdim=True)
        self.ir_encoder.load_state_dict(state, strict=True)
        with torch.no_grad():
            conv = self.depth_encoder.stages[0][0]
            conv.weight[:, :1].copy_(self.model[0].conv.weight[:32].sum(dim=1, keepdim=True))
            conv.weight[:, 1:].zero_()

    def _predict_once(self, x: torch.Tensor, profile: bool = False,
                      embed: list[int] | None = None) -> Any:
        """仅替换供颈部使用的P3，RGB主干P4/P5不先被辅助模态改写。"""
        if not hasattr(self, "ir_encoder"):
            return DetectionModel._predict_once(self, x, profile=profile, embed=embed)
        if x.ndim != 4 or x.shape[1] != 6 or profile or embed is not None:
            raise ValueError("v19要求RGB3+IR1+Depth1+Support1，不支持旧五通道或embed/profile")
        infrared = self.ir_encoder(x[:, 3:4])
        support = F.adaptive_avg_pool2d(x[:, 5:6].float(), infrared.shape[-2:]).to(infrared.dtype)
        depth = self.depth_encoder(torch.cat((x[:, 4:5] * x[:, 5:6], x[:, 5:6]), dim=1))
        if self.training:
            self.last_support_mean = support.detach().float().mean()
        x = x[:, :3]
        saved: list[Any] = []
        for layer in self.model:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            saved.append(x if layer.i in self.save else None)
            if layer.i == 10:
                rgb = saved[4]
                saved[4] = rgb + self.ir_fusion(rgb, infrared) + support * self.depth_fusion(rgb, depth)
        return x


# 历史v9权重记录中文模块名；在入口加载权重前注册别名，不保留空壳转发文件。
sys.modules.setdefault("三模态融合", sys.modules[__name__])
