"""v28：RGB旁路、局部质量门控及浅层细节融合，训练和预测共用模型。

借鉴局部光照融合、辅助模态选择和非对称分支的思想；不是论文模块的完整复现。
输入仍为连续FP32的RGB3+IR1+Depth1，新增有效性与光照线索在网络内计算。
"""

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ultralytics.nn.tasks import DetectionModel

from .model import DEFAULT_CONTENT_HW, FUSION_LAYERS, FusionDetectionModel


RELIABILITY_FUSION_VERSION = "yolo26_reliability_p345_v28_1"


def conv_norm(source: int, target: int, stride: int = 1) -> nn.Sequential:
    """新增分支使用GroupNorm，不依赖batch1的批统计量。"""
    return nn.Sequential(nn.Conv2d(source, target, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, target), nn.SiLU())


class DetailEncoder(nn.Module):
    """轻量独立分支；把P2四个子像素位置送入P3，减轻直接下采样的信息损失。"""

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList((conv_norm(2, 8, 2), conv_norm(8, 16, 2),
                                     conv_norm(16, 32, 2), conv_norm(32, 64, 2),
                                     conv_norm(64, 128, 2)))
        self.detail = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(64, 32, 1, bias=False),
                                    nn.GroupNorm(8, 32), nn.SiLU())

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        features = []
        detail = None
        for index, stage in enumerate(self.stages):
            image = stage(image)
            if index == 1:
                detail = self.detail(image)
            if index == 2:
                image = image + detail
            if index >= 2:
                features.append(image)
        return features


class LocalQualityFusion(nn.Module):
    """在压缩特征上做局部IR对应与三模态选择，再以门控残差注入RGB。"""

    def __init__(self, rgb_channels: int, auxiliary_channels: int, align: bool) -> None:
        super().__init__()
        width = 24
        self.rgb_context = nn.Conv2d(rgb_channels, width, 1, bias=False)
        self.ir_context = nn.Conv2d(auxiliary_channels, width, 1, bias=False)
        self.depth_context = nn.Conv2d(auxiliary_channels, width, 1, bias=False)
        self.gate = nn.Sequential(conv_norm(width * 3 + 4, width), nn.Conv2d(width, 3, 1))
        self.project = nn.Conv2d(width, rgb_channels, 1, bias=False)
        self.align = align
        # 零残差保留官方RGB起点；第一步先学习投影，后续梯度进入分支和门控。
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
        self.register_buffer("center_prior", torch.tensor([0., 0., 0., 0., 2., 0., 0., 0., 0.]).view(1, 9, 1, 1))
        self.last_gates: torch.Tensor | None = None

    def align_infrared(self, query: torch.Tensor, infrared: torch.Tensor) -> torch.Tensor:
        """仅搜索相邻3×3特征，带中心先验；不做高分辨率全局注意力。"""
        batch, channels, height, width = infrared.shape
        patches = F.unfold(F.pad(infrared, (1, 1, 1, 1), mode="replicate"), 3)
        patches = patches.view(batch, channels, 9, height, width)
        similarity = (F.normalize(query, dim=1).unsqueeze(2) * F.normalize(patches, dim=1)).sum(1)
        weights = (4 * similarity + self.center_prior).softmax(1)
        return (patches * weights.unsqueeze(1)).sum(2)

    def forward(self, rgb: torch.Tensor, infrared: torch.Tensor, depth: torch.Tensor,
                cues: torch.Tensor) -> torch.Tensor:
        query = self.rgb_context(rgb)
        ir = self.ir_context(infrared)
        if self.align:
            ir = self.align_infrared(query, ir)
        dep = self.depth_context(depth)
        cues = F.adaptive_avg_pool2d(cues, rgb.shape[-2:])
        support = cues[:, 3:4].clamp(0, 1)
        dep = dep * (support > 0).to(dep.dtype)
        logits = self.gate(torch.cat((query, ir, dep, cues), dim=1))
        # RGB始终保留直接旁路；深度有效比例只限制辅助贡献，不删除RGB目标。
        weights = logits.softmax(1)
        usable = torch.cat((weights[:, :2], weights[:, 2:3] * support), dim=1)
        usable = usable / usable.sum(1, keepdim=True).clamp_min(1e-6)
        if self.training:
            self.last_gates = usable.detach().mean((0, 2, 3))
        residual = usable[:, 1:2] * ir + usable[:, 2:3] * dep
        return rgb + self.project(residual)


class ReliabilityFusionDetectionModel(FusionDetectionModel):
    """三分支P3/P4/P5融合；官方RGB骨干和检测头参数路径保持可迁移。"""

    def __init__(self, cfg: Any, nc: int = 12, verbose: bool = True) -> None:
        # 绕过历史v9的L规模硬编码；构建期间的步长探测走官方RGB路径。
        DetectionModel.__init__(self, deepcopy(cfg), ch=3, nc=nc, verbose=verbose)
        if self.yaml.get("scale") not in {"l", "x"} or len(self.model) != 24:
            raise ValueError("v28只接受已核对层号的YOLO26l/x官方结构")
        rgb_channels = tuple(self.model[index].cv2.conv.out_channels for index in FUSION_LAYERS)
        p2_channels = self.model[2].cv2.conv.out_channels
        self.ir_encoder = DetailEncoder()
        self.depth_encoder = DetailEncoder()
        self.fusion_blocks = nn.ModuleList(LocalQualityFusion(channels, auxiliary, level < 2)
                                          for level, (channels, auxiliary) in enumerate(zip(rgb_channels, (32, 64, 128))))
        # 先压缩RGB P2再保留四个子位置，避免建立昂贵的额外P2检测头。
        self.rgb_detail = nn.Sequential(nn.Conv2d(p2_channels, 16, 1, bias=False),
                                        nn.GroupNorm(8, 16), nn.SiLU(), nn.PixelUnshuffle(2),
                                        nn.Conv2d(64, rgb_channels[0], 1, bias=False))
        nn.init.zeros_(self.rgb_detail[-1].weight)
        self.fusion_version = RELIABILITY_FUSION_VERSION
        self.content_hw = DEFAULT_CONTENT_HW
        self.yaml["channels"] = 5
        self.end2end = False
        self.checkpoint_auxiliary = True
        self.last_support_mean: torch.Tensor | None = None

    def run_auxiliary(self, module: nn.Module, *inputs: torch.Tensor) -> Any:
        """训练重算轻量分支节约激活显存；预测直接前向，不改变数学配方。"""
        if self.training and torch.is_grad_enabled() and self.checkpoint_auxiliary:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def _predict_once(self, x: torch.Tensor, profile: bool = False, embed: Any = None) -> Any:
        if not hasattr(self, "fusion_blocks"):
            return DetectionModel._predict_once(self, x, profile=profile, embed=embed)
        if profile or embed:
            raise ValueError("v28融合前向不支持profile/embed，请使用正常训练或预测入口")
        if x.ndim != 4 or x.shape[1] != 5 or x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError("v28要求RGB3+IR1+Depth1，画布高宽须被32整除")
        rgb, infrared, depth = x[:, :3], x[:, 3:4], x[:, 4:5]
        support = (depth > 0).to(depth.dtype)
        ir_local = infrared - F.avg_pool2d(infrared, 5, 1, 2, count_include_pad=False)
        ir_features = self.run_auxiliary(self.ir_encoder, torch.cat((infrared, ir_local), dim=1))
        depth_features = self.run_auxiliary(self.depth_encoder, torch.cat((depth, support), dim=1))
        # 光照/局部起伏只是学习线索，不能直接当作昼夜标签或热伪影真值。
        luma = (rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114)
        mean = F.avg_pool2d(luma, 8, 8)
        contrast = (F.avg_pool2d(luma.square(), 8, 8) - mean.square()).clamp_min(0).sqrt()
        cues = torch.cat((mean, contrast, F.avg_pool2d(ir_local.abs(), 8, 8),
                          F.avg_pool2d(support, 8, 8)), dim=1)
        if self.training:
            self.last_support_mean = support.detach().mean()
        saved = []
        detail = None
        x = rgb
        for layer in self.model:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if index == -1 else saved[index] for index in layer.f]
            x = layer(x)
            if layer.i == 2:
                detail = self.run_auxiliary(self.rgb_detail, x)
            saved.append(x if layer.i in self.save else None)
            if layer.i == 10:
                # RGB主干0–10层先独立完成；融合只交给Neck，避免辅助噪声逐级污染主干。
                for level, index in enumerate(FUSION_LAYERS):
                    original = x if index == 10 else saved[index]
                    fused = self.run_auxiliary(self.fusion_blocks[level], original,
                                               ir_features[level], depth_features[level], cues)
                    saved[index] = fused + detail if level == 0 else fused
                x = saved[10]
        return x
