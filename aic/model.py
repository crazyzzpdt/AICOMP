"""提供 YOLO26l 三分支门控融合与训练、预测共用的矩形预处理。

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
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import nms
from ultralytics.utils.loss import E2ELoss


# 结构和预处理共同版本化，不把旧五通道首层模型当作新分支模型。
FUSION_VERSION: str = "yolo26l_gated_rect_v9_1"
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

    def _predict_once(self, x: torch.Tensor, profile: bool = False, visualize: bool = False,
                      embed: list[int] | None = None) -> Any:
        """RGB主干独立前向后替换颈部输入，辅助特征不污染后续RGB主干。"""
        # 父类构建时用三通道小张量推导步长，此时辅助分支尚未建立。
        if not hasattr(self, "ir_encoder"):
            return super()._predict_once(x, profile, visualize, embed)
        if x.shape[1] != 5 or embed is not None or profile or visualize:
            raise ValueError("融合模型要求五通道常规前向，不支持 embed/profile/visualize")
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


# 历史v9权重记录中文模块名；在入口加载权重前注册别名，不保留空壳转发文件。
sys.modules.setdefault("三模态融合", sys.modules[__name__])
