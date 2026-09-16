"""用提交预测流程统一评估本地五通道权重，并记录可复核的验证集指纹。

直接读取三种源图和标签内容，不读取或写入 Ultralytics 的标签、NPY 缓存。
每项评估独占新目录；已有结果一律不覆盖，也不按旧元数据自动跳过。

在项目目录执行：
    uv run python 评估模型.py --weights runs/detect/运行名/weights/best.pt --imgsz 1280 1536
指定新的结果目录：
    uv run python 评估模型.py --weights 权重一.pt 权重二.pt --output runs/evaluations/统一复评
"""

# 内置库
import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

# 必须在导入 Ultralytics 前设置，评估不联网检查或自动安装依赖。
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

# 三方库
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import DetMetrics

# 自己的模块
from predict import CONF_THRESHOLD, IOU_THRESHOLD, MAX_DETECTIONS, predict_image, validate_model
from 三模态训练 import DEPTH_MAX_MM, fuse_modalities
from 准备三模态数据集 import CLASS_NAMES, IMAGE_SUFFIXES


# 命令行相对路径以仓库根目录为基准，避免 IDE 工作目录改变评估对象。
PROJECT_ROOT: Path = Path(__file__).resolve().parent
# 正式复评只使用现有验证划分，不读取官方测试集。
DATA_PATH: Path = PROJECT_ROOT / "datasets/data.yaml"
# 现有划分固定为 1744/256，默认必须评估完整的 256 张。
EXPECTED_COUNT: int = 256
# 统一结果根目录，每次命令创建带时间戳的独立子目录。
OUTPUT_ROOT: Path = PROJECT_ROOT / "runs/evaluations"


@dataclass(frozen=True)
class ValidationSample:
    """保存一个已校验样本的三种路径和实际读取的五列标签。"""

    visible: Path
    infrared: Path
    depth: Path
    labels: np.ndarray


@dataclass(frozen=True)
class ValidationSnapshot:
    """将评估使用的标签快照与源文件内容指纹绑定。"""

    samples: tuple[ValidationSample, ...]
    fingerprints: dict[str, str]


def file_sha256(path: Path) -> str:
    """流式计算文件内容指纹，不依赖文件长度或修改时间。"""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def records_sha256(records: list[tuple[str, str]]) -> str:
    """将带相对名称的有序记录编码为无拼接歧义的指纹。"""
    encoded = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_labels(content: bytes, name: str) -> np.ndarray:
    """读取已清洗的 YOLO 五列标签，拒绝坏行而不隐式丢弃目标。"""
    rows = [line.split() for line in content.decode("utf-8-sig").splitlines() if line.strip()]
    if any(len(row) != 5 for row in rows):
        raise ValueError(f"验证标签必须为五列：{name}")
    labels = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
    if not np.isfinite(labels).all():
        raise ValueError(f"验证标签包含非有限数值：{name}")
    if len(labels):
        classes, boxes = labels[:, 0], labels[:, 1:]
        if np.any(classes != np.floor(classes)) or np.any((classes < 0) | (classes >= len(CLASS_NAMES))):
            raise ValueError(f"验证标签类别必须是 0 至 11 的整数：{name}")
        if np.any((boxes < 0) | (boxes > 1)) or np.any(boxes[:, 2:] <= 0):
            raise ValueError(f"验证标签坐标越界或框宽高为零：{name}")
    return labels


def inspect_validation(data: Path, expected_count: int = EXPECTED_COUNT) -> ValidationSnapshot:
    """读取完整验证集并计算标签内容、图像内容和 train/val 名单指纹。

    Args:
        data: 含五通道模态路径和 12 类顺序的本地 YAML。
        expected_count: 必须读到的验证样本数；正式运行默认 256。

    Returns:
        标签内容快照及 SHA256 指纹，不使用框架缓存。

    Raises:
        ValueError: 类别、数量、图像划分或标签格式不符合要求。
        FileNotFoundError: 对应模态或标签缺失。
    """
    config_content = data.read_bytes()
    config = yaml.safe_load(config_content)
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if config.get("channels") != 5 or names != dict(enumerate(CLASS_NAMES)):
        raise ValueError("评估数据必须使用五通道和比赛 12 类的固定顺序")
    root = (data.parent / str(config.get("path", "."))).resolve()
    split_files: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        folder = root / str(config[split])
        split_files[split] = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if len({path.stem.casefold() for path in split_files[split]}) != len(split_files[split]):
            raise ValueError(f"{split} 图片存在标签同名冲突")
    samples: list[ValidationSample] = []
    if expected_count <= 0 or len(split_files["val"]) != expected_count:
        raise ValueError(f"验证集应有 {expected_count} 张，实际为 {len(split_files['val'])} 张")
    if {p.name.casefold() for p in split_files["train"]} & {p.name.casefold() for p in split_files["val"]}:
        raise ValueError("训练集与验证集存在同名图片，不能复评")
    modality_dirs: dict[str, Path] = {}
    for modality in ("infrared", "depth"):
        value = config[modality]
        modality_dirs[modality] = root / str(value["val"] if isinstance(value, dict) else value)
    label_records: list[tuple[str, str]] = []
    image_records: list[tuple[str, str]] = []
    for visible in split_files["val"]:
        infrared, depth = (modality_dirs[modality] / visible.name for modality in ("infrared", "depth"))
        label = visible.parent.parent / "labels" / f"{visible.stem}.txt"
        content = label.read_bytes()
        labels = read_labels(content, label.name)
        label_records.append((label.name, hashlib.sha256(content).hexdigest()))
        for modality, source in (("visible", visible), ("infrared", infrared), ("depth", depth)):
            image_records.append((f"{modality}/{source.name}", file_sha256(source)))
        samples.append(ValidationSample(visible, infrared, depth, labels))
    split_records = [(split, path.name) for split in ("train", "val") for path in split_files[split]]
    fingerprints = {
        "data_yaml_sha256": hashlib.sha256(config_content).hexdigest(),
        "validation_labels_sha256": records_sha256(label_records),
        "validation_images_sha256": records_sha256(image_records),
        "image_split_sha256": records_sha256(split_records),
    }
    return ValidationSnapshot(tuple(samples), fingerprints)


def score_predictions(predictions: Iterator[tuple[str, Results, np.ndarray]], output: Path) -> dict[str, object]:
    """按官方匹配规则累计原图检测框，返回整体指标和完整的 12 类统计。

    Args:
        predictions: 逐图提供文件名、原图坐标预测结果、五列归一化真值标签。
        output: 官方评估对象使用的输出目录；关闭图表和额外预测保存。

    Returns:
        整体指标、每类 AP 与目标数。验证集中没有真值的类别 AP 为 None，
        整体 mAP 按 Ultralytics 规则仅对存在真值的类别求平均。
    """
    validator = DetectionValidator(save_dir=output, args={"plots": False, "save": False, "save_txt": False})
    metrics = DetMetrics(names=dict(enumerate(CLASS_NAMES)))
    image_count: int = 0
    for name, result, labels in predictions:
        if result.boxes is None:
            raise ValueError(f"预测结果缺少检测框：{name}")
        predicted = result.boxes.data.detach().cpu().float()
        if predicted.ndim != 2 or predicted.shape[1] != 6 or not torch.isfinite(predicted).all():
            raise ValueError(f"预测结果不是有限数值的六列检测框：{name}")
        if len(predicted) > MAX_DETECTIONS:
            raise ValueError(f"预测框数量超过赛事上限：{name}")
        truth = torch.from_numpy(labels.copy())
        height, width = result.orig_shape
        truth_boxes = ops.xywh2xyxy(truth[:, 1:5]) * torch.tensor([width, height, width, height])
        pred = {"bboxes": predicted[:, :4], "cls": predicted[:, 5]}
        target = {"bboxes": truth_boxes, "cls": truth[:, 0]}
        metrics.update_stats({
            **validator._process_batch(pred, target),
            "target_cls": labels[:, 0], "target_img": np.unique(labels[:, 0]),
            "conf": predicted[:, 4].numpy(), "pred_cls": predicted[:, 5].numpy(), "im_name": name,
        })
        image_count += 1
    if not image_count:
        raise ValueError("没有验证图片，不能计算检测指标")
    metrics.process(save_dir=output, plot=False)
    per_class: list[dict[str, object]] = []
    class_positions = {int(class_id): index for index, class_id in enumerate(metrics.ap_class_index)}
    for class_id, class_name in enumerate(CLASS_NAMES):
        position = class_positions.get(class_id)
        precision, recall, ap50, ap50_95 = metrics.class_result(position) if position is not None else (None,) * 4
        per_class.append({
            "class_id": class_id, "class_name": class_name,
            "images": int(metrics.nt_per_image[class_id]), "instances": int(metrics.nt_per_class[class_id]),
            "precision": None if precision is None else float(precision),
            "recall": None if recall is None else float(recall),
            "ap50": None if ap50 is None else float(ap50),
            "ap50_95": None if ap50_95 is None else float(ap50_95),
        })
    return {
        "images": image_count, "instances": int(metrics.nt_per_class.sum()),
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()}, "per_class": per_class,
    }


def relative_name(path: Path) -> str:
    """记录仓库内相对位置，外部文件只记录名称，避免写入机器个人路径。"""
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else path.name


def write_json(path: Path, content: dict[str, object]) -> None:
    """排他写入 JSON；禁止非有限数值冒充有效评估指标。"""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """按首行字段排他写入 UTF-8 CSV，方便本地表格软件比较。"""
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_row(report: dict[str, object]) -> dict[str, object]:
    """将单项报告中的身份、主要参数和整体指标整理为 CSV 行。"""
    metadata = report["metadata"]
    return {
        "weights": metadata["weights"], "weights_sha256": metadata["weights_sha256"],
        "imgsz": metadata["parameters"]["imgsz"], "device": metadata["parameters"]["device"],
        "images": report["images"], "instances": report["instances"],
        **metadata["fingerprints"], **report["metrics"], "elapsed_seconds": report["elapsed_seconds"],
    }


def evaluate_checkpoint(
    weights: Path, imgsz: int, output: Path, device: str = "0", *, data: Path = DATA_PATH,
    expected_count: int = EXPECTED_COUNT,
) -> dict[str, object]:
    """在完整验证集上评估一个检查点，并返回与磁盘相同的完成报告。

    Args:
        weights: 已存在的本地五通道、12 类 PyTorch 检查点。
        imgsz: 32 的正整数倍，正式对比使用 1280 或 1536。
        output: 必须尚不存在的单项评估目录。
        device: 默认 CUDA:0；小型离线测试可使用 cpu。
        data: 当前数据集配置；默认 datasets/data.yaml。
        expected_count: 预期完整验证集数量，默认 256。

    Returns:
        含检查点与数据指纹、参数、整体指标和 12 类 AP 的报告。

    Raises:
        FileExistsError: 输出目录已存在。
        ValueError: 参数、模型或数据不合法，或评估期间输入文件发生变化。

    Note:
        每次重新读取源图和标签；已有 .cache/.npy 不影响结果。
        中断目录只含开始记录，不会生成 status=complete 的 evaluation.json。
    """
    weights, data, output = ((PROJECT_ROOT / path).resolve() for path in (weights, data, output))
    if output.exists():
        raise FileExistsError(f"评估目录已存在，不会覆盖：{output}")
    if imgsz <= 0 or imgsz % 32:
        raise ValueError("imgsz 必须为 32 的正整数倍")
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise FileNotFoundError(f"请指定已有的本地 .pt 检测权重：{weights}")
    if output.is_relative_to(data.parent) or data.parent.is_relative_to(output) or weights.is_relative_to(output):
        raise ValueError("评估目录不能包含数据或检查点，也不能写入数据目录")
    snapshot = inspect_validation(data, expected_count)
    checkpoint_hash = file_sha256(weights)
    model = YOLO(str(weights), task="detect")
    validate_model(model)
    metadata: dict[str, object] = {
        "weights": relative_name(weights), "weights_sha256": checkpoint_hash,
        "data": relative_name(data), "fingerprints": snapshot.fingerprints,
        "classes": list(CLASS_NAMES), "expected_images": expected_count,
        "framework": {"ultralytics": ultralytics.__version__, "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
        "implementation_sha256": {name: file_sha256(PROJECT_ROOT / name) for name in ("评估模型.py", "predict.py", "三模态训练.py")},
        "parameters": {
            "imgsz": imgsz, "device": device, "quantize": 32, "batch": 1, "rect": True,
            "conf": CONF_THRESHOLD, "iou": IOU_THRESHOLD, "nms": True, "max_det": MAX_DETECTIONS,
            "augment": False, "agnostic_nms": False, "classes": None, "channels": 5,
            "channel_order": "RGB,infrared,depth", "depth_max_mm": DEPTH_MAX_MM,
            "compile": False, "channels_last": False, "cache": False,
            "ap_iou_thresholds": [round(0.5 + step * 0.05, 2) for step in range(10)],
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    started_at: str = datetime.now().astimezone().isoformat()
    write_json(output / "started.json", {"status": "started", "created_at": started_at, "metadata": metadata})
    print(f"评估权重：{relative_name(weights)}，尺寸 {imgsz}，验证图共 {len(snapshot.samples)} 张")
    started = perf_counter()

    def predictions() -> Iterator[tuple[str, Results, np.ndarray]]:
        """逐张融合并预测，只保留单张五通道图和官方指标所需的小数组。"""
        for index, sample in enumerate(snapshot.samples, start=1):
            fused = fuse_modalities(sample.visible, sample.infrared, sample.depth)
            result = predict_image(model, fused, output, device, imgsz, CONF_THRESHOLD, IOU_THRESHOLD)
            yield sample.visible.name, result, sample.labels
            if index == 1 or index % 25 == 0 or index == len(snapshot.samples):
                print(f"评估完成 {index}/{len(snapshot.samples)}：{sample.visible.name}")

    scores = score_predictions(predictions(), output)
    if snapshot.fingerprints != inspect_validation(data, expected_count).fingerprints or checkpoint_hash != file_sha256(weights):
        raise ValueError("评估期间数据或检查点内容发生变化，拒绝保存完整评估结果")
    report: dict[str, object] = {
        "status": "complete", "created_at": started_at, "completed_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3), "metadata": metadata, **scores,
    }
    write_csv(output / "per_class.csv", report["per_class"])
    write_csv(output / "summary.csv", [summary_row(report)])
    write_json(output / "evaluation.json", report)
    print(f"完整评估已保存：{output}；mAP50-95={report['metrics']['metrics/mAP50-95(B)']:.6f}")
    return report


def main() -> None:
    """串行评估指定权重与尺寸，汇总同口径结果。"""
    parser = argparse.ArgumentParser(description="统一评估五通道权重")
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[1280, 1536])
    parser.add_argument("--device", default="0")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for index, weights in enumerate(args.weights):
        for size in args.imgsz:
            report = evaluate_checkpoint(weights, size, output / f"{index}_{weights.stem}_{size}", args.device, data=args.data)
            rows.append(summary_row(report))
    write_csv(output / "comparison.csv", rows)


# Windows 子进程和测试导入时不执行评估。
if __name__ == "__main__":
    main()
