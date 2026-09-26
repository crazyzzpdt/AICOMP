"""提供D-FINE训练/预测共用构建、源码身份和查询解码，不依赖YOLO。"""

from __future__ import annotations

# 内置库
import hashlib
import json
import sys
from pathlib import Path

# 三方库
import torch
from torch import nn
from torchvision.ops import box_convert

# 自己的模块
from src.modalities import CLASS_NAMES

DFINE_COMMIT: str = "956d1709314c2c6a4df6f34de232054578a7449f"
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def check_source() -> Path:
    """验证随仓库分发的源码摘要，不依赖子目录存在.git。"""
    source = PROJECT_ROOT / "src" / "D-FINE"
    manifest = json.loads((source / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("upstream_commit") != DFINE_COMMIT or not manifest.get("files"):
        raise ValueError("D-FINE源码清单缺失或版本不符")
    for name, digest in manifest["files"].items():
        path = source / name
        # Git在Windows可换为CRLF；统一换行后校验内容，不能因平台差异误拒绝相同源码。
        if not path.is_file() or hashlib.sha256(path.read_bytes().removeprefix(b"\xef\xbb\xbf").replace(b"\r\n", b"\n")).hexdigest() != digest:
            raise ValueError(f"D-FINE源码与已记录快照不符：{name}")
    import src as project_src
    package = str(source / "src")
    for loaded_name in ("src.core", "src.nn", "src.zoo", "src.optim"):
        loaded = sys.modules.get(loaded_name)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
            raise ValueError(f"已加载另一份D-FINE模块：{loaded_name}；请在新进程运行")
    if package not in project_src.__path__:
        project_src.__path__.append(package)
    return source


def build_model(imgsz: int, training: bool = False, variant: str = "l") -> tuple[nn.Module, nn.Module | None]:
    """构建不联网的12类五通道D-FINE-L/X，输入端不冻结。

    Args:
        imgsz: 正方形填充尺寸，必须为 32 的倍数。
        training: 是否同时创建官方匹配器与损失函数。

    Returns:
        五通道模型，以及训练时使用的损失函数。
    """
    if variant not in {"l", "x"}:
        raise ValueError("D-FINE规模仅支持l或x，不能按文件名静默猜测")
    source = check_source()
    from src.core import YAMLConfig
    import src.nn  # noqa: F401
    import src.zoo.dfine  # noqa: F401
    import src.optim  # noqa: F401

    config = YAMLConfig(
        str(source / ("configs/dfine/dfine_hgnetv2_x_coco.yml" if variant == "x"
                      else "configs/dfine/include/dfine_hgnetv2.yml")),
        num_classes=len(CLASS_NAMES), remap_mscoco_category=False,
        eval_spatial_size=[imgsz, imgsz], num_top_queries=100,
        HGNetv2={"name": "B5" if variant == "x" else "B4", "pretrained": False,
                 "freeze_at": -1, "freeze_norm": True},
    )
    model = config.model
    original = model.backbone.stem.stem1.conv
    expanded = nn.Conv2d(5, original.out_channels, original.kernel_size, original.stride,
                         original.padding, bias=original.bias is not None)
    model.backbone.stem.stem1.conv = expanded
    return model, config.criterion if training else None


def decode_predictions(outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]], imgsz: int,
                       conf: float, max_det: int) -> list[dict[str, torch.Tensor]]:
    """保留 D-FINE 原生查询类别排序，反算填充和缩放；不使用 NMS 或集成。"""
    probabilities = outputs["pred_logits"].float().sigmoid()
    # 与官方后处理一致，从查询×类别联合分数选择候选；不是 YOLO 的多标签 NMS。
    scores, positions = probabilities.flatten(1).topk(min(300, probabilities.shape[1] * len(CLASS_NAMES)), dim=1)
    categories = positions % len(CLASS_NAMES)
    queries = positions // len(CLASS_NAMES)
    all_boxes = box_convert(outputs["pred_boxes"].float(), "cxcywh", "xyxy") * imgsz
    all_boxes = all_boxes.gather(1, queries.unsqueeze(-1).expand(-1, -1, 4))
    results: list[dict[str, torch.Tensor]] = []
    for boxes, confidence, labels, target in zip(all_boxes, scores, categories, targets):
        sx, sy, left, top = [float(v) for v in target["geometry"]]
        width, height = [int(v) for v in target["orig_size"]]
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / sx).clamp(0, width)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / sy).clamp(0, height)
        valid = (confidence >= conf) & torch.isfinite(boxes).all(1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        results.append({"boxes": boxes[valid][:max_det], "scores": confidence[valid][:max_det], "labels": labels[valid][:max_det]})
    return results
