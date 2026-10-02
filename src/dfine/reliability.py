"""D-FINE v28：独立RGB骨干、轻量辅助模态和局部质量残差融合。

训练和预测共用结构，官方骨干/编码器/解码器参数路径保持不变。
借鉴YOLO v28的融合设计，不导入YOLO包，也不修改上游D-FINE源码。
"""

# 三方库
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# 写入检查点，预测不能仅凭文件名决定网络结构。
RELIABILITY_ARCHITECTURE: str = "dfine_reliability_p345_v28_1"
# 仅这些模块允许在官方预训练权重之外重新初始化。
AUXILIARY_PREFIXES: tuple[str, ...] = ("ir_encoder.", "depth_encoder.", "fusion_blocks.", "rgb_detail.")


def conv_norm(source: int, target: int, stride: int = 1) -> nn.Sequential:
    """辅助分支采用GroupNorm，避免单张物理批次的批统计依赖。"""
    return nn.Sequential(nn.Conv2d(source, target, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, target), nn.SiLU())


class DetailEncoder(nn.Module):
    """提取P3/P4/P5辅助特征，并把P2四个子像素位置送入P3。"""

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList((conv_norm(2, 8, 2), conv_norm(8, 16, 2),
                                     conv_norm(16, 32, 2), conv_norm(32, 64, 2),
                                     conv_norm(64, 128, 2)))
        self.detail = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(64, 32, 1, bias=False),
                                    nn.GroupNorm(8, 32), nn.SiLU())

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        detail: torch.Tensor | None = None
        for index, stage in enumerate(self.stages):
            image = stage(image)
            if index == 1:
                detail = self.detail(image)
            if index == 2:
                if detail is None:
                    raise RuntimeError("辅助分支缺少P2细节")
                image = image + detail
            if index >= 2:
                features.append(image)
        return features


class LocalQualityFusion(nn.Module):
    """以局部IR对应和深度支持限制辅助残差，RGB保持直接旁路。"""

    def __init__(self, rgb_channels: int, auxiliary_channels: int, align: bool) -> None:
        super().__init__()
        width = 24
        self.rgb_context = nn.Conv2d(rgb_channels, width, 1, bias=False)
        self.ir_context = nn.Conv2d(auxiliary_channels, width, 1, bias=False)
        self.depth_context = nn.Conv2d(auxiliary_channels, width, 1, bias=False)
        self.gate = nn.Sequential(conv_norm(width * 3 + 4, width), nn.Conv2d(width, 3, 1))
        self.project = nn.Conv2d(width, rgb_channels, 1, bias=False)
        self.align = align
        # 零投影从官方RGB起点学习，第一步更新投影后再向辅助编码器传递梯度。
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
        self.register_buffer("center_prior", torch.tensor([0., 0., 0., 0., 2., 0., 0., 0., 0.]).view(1, 9, 1, 1))

    def align_infrared(self, query: torch.Tensor, infrared: torch.Tensor) -> torch.Tensor:
        """在3×3邻域做带中心先验的软对应，不使用全图注意力。"""
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
        weights = self.gate(torch.cat((query, ir, dep, cues), dim=1)).softmax(1)
        usable = torch.cat((weights[:, :2], weights[:, 2:3] * support), dim=1)
        usable = usable / usable.sum(1, keepdim=True).clamp_min(1e-6)
        residual = usable[:, 1:2] * ir + usable[:, 2:3] * dep
        return rgb + self.project(residual)


class ReliabilityDFine(nn.Module):
    """在官方RGB骨干输出与HybridEncoder之间融合辅助模态。

    Args:
        model: 尚未扩展成五通道的官方D-FINE模型。

    Note:
        backbone、encoder和decoder保持原参数路径，便于严格核对预训练迁移。
        不把融合结果反馈给后续RGB骨干层，避免改变独立RGB特征路径。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.encoder = model.encoder
        self.decoder = model.decoder
        if (list(self.backbone.return_idx) != [1, 2, 3]
                or list(self.backbone._out_strides) != [4, 8, 16, 32]
                or len(self.backbone.stages) != 4
                or self.backbone.stem.stem1.conv.in_channels != 3):
            raise ValueError("D-FINE v28要求官方RGB HGNetv2四阶段、P3/P4/P5输出")
        channels = self.backbone._out_channels
        self.ir_encoder = DetailEncoder()
        self.depth_encoder = DetailEncoder()
        self.fusion_blocks = nn.ModuleList(LocalQualityFusion(rgb, aux, level < 2)
                                          for level, (rgb, aux) in enumerate(zip(channels[1:], (32, 64, 128))))
        self.rgb_detail = nn.Sequential(nn.Conv2d(channels[0], 16, 1, bias=False),
                                        nn.GroupNorm(8, 16), nn.SiLU(), nn.PixelUnshuffle(2),
                                        nn.Conv2d(64, channels[1], 1, bias=False))
        nn.init.zeros_(self.rgb_detail[-1].weight)
        self.architecture = RELIABILITY_ARCHITECTURE

    def run_encoder(self, module: DetailEncoder, image: torch.Tensor) -> list[torch.Tensor]:
        """训练重算轻量分支以减少激活占用，预测直接前向。"""
        if self.training and torch.is_grad_enabled():
            return checkpoint(module, image, use_reentrant=False)
        return module(image)

    def run_fusion(self, module: LocalQualityFusion, rgb: torch.Tensor, ir: torch.Tensor,
                   depth: torch.Tensor, cues: torch.Tensor) -> torch.Tensor:
        """训练时重算局部对应与门控，避免保留全部邻域展开激活。"""
        if self.training and torch.is_grad_enabled():
            return checkpoint(module, rgb, ir, depth, cues, use_reentrant=False)
        return module(rgb, ir, depth, cues)

    def forward(self, images: torch.Tensor,
                targets: list[dict[str, torch.Tensor]] | None = None) -> dict[str, object]:
        """接收已归一化的五通道，保持官方解码器输出和损失接口。"""
        if images.ndim != 4 or images.shape[1] != 5 or images.shape[-2] % 32 or images.shape[-1] % 32:
            raise ValueError("D-FINE v28要求[N,5,H,W]且画布高宽为32的倍数")
        rgb, infrared, depth = images[:, :3], images[:, 3:4], images[:, 4:5]
        support = (depth > 0).to(depth.dtype)
        ir_local = infrared - F.avg_pool2d(infrared, 5, 1, 2, count_include_pad=False)
        ir_features = self.run_encoder(self.ir_encoder, torch.cat((infrared, ir_local), dim=1))
        depth_features = self.run_encoder(self.depth_encoder, torch.cat((depth, support), dim=1))
        luma = rgb[:, :1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114
        mean = F.avg_pool2d(luma, 8, 8)
        contrast = (F.avg_pool2d(luma.square(), 8, 8) - mean.square()).clamp_min(0).sqrt()
        cues = torch.cat((mean, contrast, F.avg_pool2d(ir_local.abs(), 8, 8),
                          F.avg_pool2d(support, 8, 8)), dim=1)
        features: list[torch.Tensor] = []
        detail: torch.Tensor | None = None
        feature = self.backbone.stem(rgb)
        for index, stage in enumerate(self.backbone.stages):
            feature = stage(feature)
            if index == 0:
                if self.training and torch.is_grad_enabled():
                    detail = checkpoint(self.rgb_detail, feature, use_reentrant=False)
                else:
                    detail = self.rgb_detail(feature)
            else:
                features.append(feature)
        if detail is None:
            raise RuntimeError("RGB骨干未产生P2细节")
        fused: list[torch.Tensor] = []
        for level, module in enumerate(self.fusion_blocks):
            result = self.run_fusion(module, features[level], ir_features[level], depth_features[level], cues)
            fused.append(result + detail if level == 0 else result)
        return self.decoder(self.encoder(fused), targets)
