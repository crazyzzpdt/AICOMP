"""显式评估当前291张验证集，使用训练与预测共用的五通道FP32协议。

运行：uv run python -m tools.validate_local --weights runs/detect/运行名/weights/best.pt
不读取官方测试集、不训练、不覆盖旧结果；当前仅接受native_square早期融合权重。
"""

# 内置库
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

# 必须在导入Ultralytics前设置，赛事数据与模型不上传在线服务。
os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", HF_HUB_OFFLINE="1")

# 三方库
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import init_seeds

# 自己的模块
from aic.data import CLASS_NAMES, file_hash, verify_source_images
from aic.model import EVALUATION_PROTOCOL, EarlyFusionDetectionModel
from aic.training import MultimodalYOLODataset, RectangularValidator, save_validation_artifacts


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
AUDIT_PATH: Path = PROJECT_ROOT / "runs/dataset_cleaning/official_refresh_20260918_214843/manifest.json"


def validate_local(weights: Path, output: Path, batch_size: int, workers: int) -> dict[str, float]:
    """按固定划分和新协议评估已有权重，并保存可比的逐类指标与曲线。"""
    if output.exists():
        raise FileExistsError(f"评估目录已存在，不覆盖：{output}")
    if batch_size < 1 or workers < 0:
        raise ValueError("batch必须为正数，workers不能为负")
    weights = weights.resolve(strict=True)
    init_seeds(0, deterministic=True)
    loaded = YOLO(str(weights), task="detect")
    model = loaded.model
    recipe = getattr(model, "fusion_training", {}).get("signature", {}).get("recipe", {})
    if not isinstance(model, EarlyFusionDetectionModel) or recipe.get("geometry") != "native_square":
        raise ValueError("本工具只比较当前原生方形早期融合权重；v4曾见过新增val样本，不能直接当公平基线")
    if model.names != dict(enumerate(CLASS_NAMES)) or model.content_hw[0] != model.content_hw[1]:
        raise ValueError("类别或输入几何与当前协议不匹配")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("status") != "applied" or len(audit.get("samples", [])) != 2000:
        raise ValueError("当前审计不是已落位的2000组")
    rows = [row for row in audit["samples"] if row["after_split"] == "val"]
    if len(rows) != 291:
        raise ValueError("当前比较协议固定为291张val，不能静默更换划分")
    data_root = PROJECT_ROOT / "datasets"
    verify_source_images(data_root, rows)
    for row in rows:
        label = data_root / "val/labels" / (Path(row["image"]).stem + ".txt")
        if file_hash(label) != row["after_sha256"]:
            raise ValueError(f"验证标签发生未审阅改动：{label.name}")
    config = get_cfg(overrides={"task": "detect", "mode": "val", "data": str(data_root / "data.yaml"),
                                "imgsz": model.content_hw[0], "batch": batch_size, "workers": workers,
                                "device": "0", "quantize": 32, "rect": False, "conf": 0.001, "iou": 0.7,
                                "max_det": 100, "nms": True, "plots": False, "save_json": False, "save_txt": False})
    data = check_det_dataset(config.data)
    dataset = MultimodalYOLODataset(img_path=data["val"], imgsz=config.imgsz, batch_size=batch_size,
                                    augment=False, hyp=config, rect=False, cache=False, stride=32, pad=0.,
                                    prefix="local val: ", task="detect", data=data)
    if {Path(name).name for name in dataset.im_files} != {row["image"] for row in rows}:
        raise ValueError("验证加载器的实际图片与审计清单不一致")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=dataset.collate_fn, prefetch_factor=1 if workers else None, pin_memory=False)
    output.mkdir(parents=True, exist_ok=False)
    validator = RectangularValidator(loader, save_dir=output, args=config)
    validator.device, validator.data, validator.training = torch.device("cuda:0"), data, False
    validator.epoch = int(loaded.ckpt["epoch"]) + 1
    model = model.to(validator.device).float().eval()
    model.end2end = False
    validator.init_metrics(model)
    started = time.perf_counter()
    # 不走AutoBackend的自动卷积融合；与FusionPredictor及轮末EMA前向保持相同模型形式。
    with torch.inference_mode():
        for batch in tqdm(loader, desc="本地291张验证", mininterval=5):
            prepared = validator.preprocess(batch)
            predictions = validator.postprocess(model(prepared["img"]))
            validator.update_metrics(predictions, prepared)
    metrics = validator.get_stats()
    save_validation_artifacts(validator.metrics, validator.confusion_matrix, output, validator.epoch)
    metadata = {"weights": str(weights.relative_to(PROJECT_ROOT)) if weights.is_relative_to(PROJECT_ROOT) else weights.name,
                "weights_sha256": file_hash(weights), "audit_sha256": file_hash(AUDIT_PATH),
                "evaluation_protocol": EVALUATION_PROTOCOL, "images": len(dataset), "epoch": validator.epoch,
                "imgsz": config.imgsz, "batch": batch_size, "workers": workers,
                "conf": config.conf, "iou": config.iou, "max_det": config.max_det,
                "multi_label": False, "precision": "FP32", "seconds": time.perf_counter() - started,
                "source_sha256": {name: file_hash(PROJECT_ROOT / name) for name in
                                  ("tools/validate_local.py", "aic/model.py", "aic/training.py", "aic/data.py")}}
    (output / "evaluation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics}, ensure_ascii=False), flush=True)
    return metrics


def main() -> None:
    """解析显式评估命令，默认创建带时间戳的新结果目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output or Path("runs/local_validation") / f"{args.weights.parent.parent.name}_{datetime.now():%Y%m%d_%H%M%S}"
    validate_local(PROJECT_ROOT / args.weights, PROJECT_ROOT / output, args.batch, args.workers)


if __name__ == "__main__":
    main()
