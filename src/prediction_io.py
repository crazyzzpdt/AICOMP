"""三模态预测共用的读取调度、结果保存和赛事材料，不选择或加载模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# 离线执行，禁止框架自动下载依赖或检查更新。
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics.engine.results import Results

from src.modalities import CLASS_NAMES, IMAGE_SUFFIXES, configure_fp32
from tools.IntegrateAndPackage import TEAM_ID, PackageConfig, package as integrate_round2_package

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MAX_DETECTIONS: int = 100


@dataclass(frozen=True)
class OutputConfig:
    """两个入口共用的图像任务和输出设置；模型参数由各自配置定义。"""

    weights: Path = Path()
    source: Path = Path("datasets/test")
    output: Path | None = None
    expected_count: int = 1000
    imgsz: int = 1280
    height: int = 1280
    batch: int = 4
    workers: int = 8
    save_workers: int = 8
    prefetch_batches: int = 2
    pin_memory: bool = True
    device: str = "0"
    conf: float = 0.001
    max_det: int = 100
    visual_conf: float = 0.25
    png_compression: int = 1
    log_every: int = 25
    save: bool = True
    save_txt: bool = True
    save_conf: bool = True
    show_labels: bool = True
    show_conf: bool = True
    show_boxes: bool = True
    line_width: int | None = None
    verbose: bool = True
    report: Path = Path("docs/技术方案.md")


@dataclass
class PredictionSample:
    """保存单张 CPU 输入及坐标信息，不在线程之间传递 GPU 对象。"""

    path: Path
    image: np.ndarray
    visible: np.ndarray | None
    target: dict[str, torch.Tensor]


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


def prediction_batches(samples: list[tuple[Path, Path, Path]], config: OutputConfig,
                       reader: ThreadPoolExecutor, load_sample) -> Iterator[list[PredictionSample]]:
    """并行预读有限窗口并按清单顺序成批返回，读取异常直接交给主线程。"""
    paths = iter(samples)
    pending: deque[Future[PredictionSample]] = deque()
    capacity = config.batch * config.prefetch_batches
    for sample in paths:
        pending.append(reader.submit(load_sample, sample))
        if len(pending) == capacity:
            break
    batch: list[PredictionSample] = []
    while pending:
        batch.append(pending.popleft().result())
        next_paths = next(paths, None)
        if next_paths is not None:
            pending.append(reader.submit(load_sample, next_paths))
        if len(batch) == config.batch or not pending:
            yield batch
            batch = []


def prepare_output(output: Path) -> None:
    """独占创建输出根目录；即使已有目录为空也不复用。

    Raises:
        FileExistsError: 输出位置已经存在，需通过 --output 指定新位置。
    """
    if output.exists():
        raise FileExistsError(f"输出目录已存在，不会覆盖：{output}；请用 --output 指定新目录")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("images", "labels"):
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


def save_prediction(result: Results, image_name: str, output: Path, config: OutputConfig) -> None:
    """写入同名带框图片和六列标签；展示阈值不影响提交候选。

    Args:
        result: 已恢复原图坐标的检测结果。
        image_name: 配对清单中的原始图片文件名。
        output: 本次独占的输出根目录。
        config: 绘图、保存与压缩设置；展示阈值不改变TXT候选。

    Raises:
        FileExistsError: 同名结果已经存在。
        OSError: 图片无法编码或文件无法写入。
    """
    image_path, label_path = output / "images" / image_name, output / "labels" / f"{Path(image_name).stem}.txt"
    if image_path.exists() or label_path.exists():
        raise FileExistsError(f"预测结果已存在，拒绝覆盖：{image_name}")
    rows = prediction_rows(result)
    content: str = "".join(f"{int(row[0])} " + " ".join(f"{value:.10f}" for value in row[1:]) + "\n" for row in rows)
    with label_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    if not config.save:
        return
    # 绘图也使用实际提交的框，只有显示阈值不同；原始 Results 不做阈值修改。
    visible_rows = rows[rows[:, 5] >= config.visual_conf]
    height, width = result.orig_shape
    xyxy: np.ndarray = np.column_stack((
        (visible_rows[:, 1] - visible_rows[:, 3] / 2) * width, (visible_rows[:, 2] - visible_rows[:, 4] / 2) * height,
        (visible_rows[:, 1] + visible_rows[:, 3] / 2) * width, (visible_rows[:, 2] + visible_rows[:, 4] / 2) * height,
    ))
    display = Results(result.orig_img, path=image_name, names=result.names, boxes=np.column_stack((xyxy, visible_rows[:, 5], visible_rows[:, 0])))
    plotted: np.ndarray = display.plot(conf=config.show_conf, labels=config.show_labels, boxes=config.show_boxes, line_width=config.line_width, pil=False)
    parameters = [cv2.IMWRITE_PNG_COMPRESSION, config.png_compression] if image_path.suffix.lower() == ".png" else []
    success, encoded = cv2.imencode(image_path.suffix, plotted, parameters)
    if not success:
        raise OSError(f"预测图片编码失败：{image_path}")
    with image_path.open("xb") as handle:
        handle.write(encoded.tobytes())


def build_submission(output: Path, image_names: list[str], save_images: bool = True) -> Path:
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
    archive: Path = output / "submission.zip"
    if archive.exists():
        raise FileExistsError(f"提交包已存在，拒绝覆盖：{archive}")
    expected: set[str] = {f"{Path(name).stem}.txt" for name in image_names}
    if not expected or len(expected) != len(image_names):
        raise ValueError("提交清单为空或存在同名标签")
    if {p.name for p in (output / "labels").iterdir()} != expected:
        raise ValueError("标签文件数量或名称与待预测图片不一致，不能生成提交包")
    if save_images and {p.name for p in (output / "images").iterdir()} != set(image_names):
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


def prediction_source_hashes(source_paths: tuple[str, ...]) -> dict[str, str]:
    """记录入口实际依赖的源码，不将另一个模型框架混入材料。"""
    paths: dict[str, Path] = {}
    for name in source_paths:
        source = PROJECT_ROOT / name
        if not source.exists():
            raise FileNotFoundError(f"预测源码缺失：{source}")
        candidates = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in candidates:
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json"}:
                paths[path.relative_to(PROJECT_ROOT).as_posix()] = path
    framework = Path(ultralytics.__file__).parent
    paths.update(("ultralytics/" + path.relative_to(framework).as_posix(), path)
                 for path in sorted(framework.rglob("*"))
                 if path.is_file() and path.suffix in {".py", ".yaml", ".yml"})
    hashes = {}
    for name, path in paths.items():
        with path.open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return hashes


def build_round2_package(config: OutputConfig, submission: Path) -> Path:
    """调用tools中的官方复赛整合工具，并在成功后移除临时排行榜ZIP。"""
    run = config.weights.parent.parent
    package = integrate_round2_package(PackageConfig(
        package_dir=config.output,
        model=config.weights,
        submission=submission,
        run=run,
        report=(PROJECT_ROOT / config.report).resolve(),
    ))
    if not package.is_dir():
        raise RuntimeError(f"复赛整合工具没有生成预期目录：{package}")
    submission.unlink()
    return package


def parse_arguments(config: OutputConfig, argv: list[str]) -> OutputConfig:
    """仅覆盖显式提供的参数；命令行名称与入口字段一一对应。"""
    parser = argparse.ArgumentParser(description="三模态预测，默认值见对应入口", argument_default=argparse.SUPPRESS)
    paths = {"weights", "source", "output", "project", "report", "data"}
    integers = {"imgsz", "height", "batch", "workers", "save_workers", "prefetch_batches", "expected_count",
                "max_det", "png_compression", "log_every", "vid_stride", "line_width", "quantize"}
    floats = {"conf", "iou", "visual_conf"}
    for field in fields(config):
        name, value = field.name, getattr(config, field.name)
        flag = "--" + name.replace("_", "-")
        if name in {"classes", "embed"}:
            parser.add_argument(flag, type=int, nargs="+")
        elif name == "compile":
            parser.add_argument(flag, nargs="?", const=True)
            parser.add_argument("--no-compile", dest=name, action="store_false", default=argparse.SUPPRESS)
        elif isinstance(value, bool) or name == "rect":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(flag, type=Path if name in paths else int if name in integers else float if name in floats else str)
    return replace(config, **vars(parser.parse_args(argv)))


def run_prediction(config: OutputConfig, create_backend, source_paths: tuple[str, ...], entrypoint: str) -> None:
    """并行预测后调用tools复赛工具，最终只保留三个结果文件夹。

    Args:
        config: 入口文件中显式设置的预测参数。
        create_backend: 对应框架的加载函数，返回读图、前向函数及真实执行记录。
        source_paths: 该入口参与摘要记录的源码路径。
        entrypoint: 本次入口文件名，写入预测记录。

    Note:
        GPU 只在主线程执行。后台保存仅持有 CPU 结果，异常会阻止生成正式 ZIP。
        不自动试跑探测显存，不重训，不修改数据或覆盖旧输出。
    """
    if config.imgsz <= 0 or config.imgsz % 32:
        raise ValueError("imgsz 必须为 32 的正整数倍")
    if min(config.batch, config.workers, config.save_workers, config.prefetch_batches, config.expected_count, config.log_every) <= 0:
        raise ValueError("批次、线程数、预取批数、图像数量和日志间隔必须为正整数")
    if not 1 <= config.max_det <= MAX_DETECTIONS or not 0 <= config.png_compression <= 9:
        raise ValueError("max_det 必须在 1–100，png_compression 必须在 0–9")
    if not all(0 <= value <= 1 for value in (config.conf, config.visual_conf)):
        raise ValueError("conf、visual_conf 必须位于 [0,1]")
    if config.height <= 0:
        raise ValueError("height必须为正整数")
    if config.output is None:
        raise ValueError("必须指定预测输出目录")
    device = str(config.device).removeprefix("cuda:")
    if device != "cpu" and not device.isdigit():
        raise ValueError("device须为单张CUDA编号、cuda:编号或cpu")
    config = replace(config, device=device)
    if not config.save_txt or not config.save_conf:
        raise ValueError("赛事提交必须save_txt=True、save_conf=True，确保每图都有六列标签")
    if config.line_width is not None and config.line_width <= 0:
        raise ValueError("line_width须为正整数或None")
    weights, source, output = ((PROJECT_ROOT / path).resolve() for path in (config.weights, config.source, config.output))
    if output.exists():
        raise FileExistsError(f"输出目录已存在，不会覆盖：{output}；请用 --output 指定新目录")
    if not weights.is_file() or weights.suffix.lower() not in {".pt", ".pth"}:
        raise FileNotFoundError(f"请指定已有的本地 .pt/.pth 五通道检测权重：{weights}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("预测输出与官方源数据目录不能互相包含")
    samples = collect_samples(source)
    if len(samples) != config.expected_count:
        raise ValueError(f"测试集应有 {config.expected_count} 组，实际找到 {len(samples)} 组；请检查 --source")
    with weights.open("rb") as handle:
        weight_hash: str = hashlib.file_digest(handle, "sha256").hexdigest()
    configure_fp32()
    config = replace(config, weights=weights, source=source, output=output)
    source_hashes = prediction_source_hashes(source_paths)
    load_sample, forward_batch, backend_metadata = create_backend(config)
    print("FP32推理、TF32关闭；三模态预处理取自所选权重，不启用训练随机增强")
    prepare_output(output)
    print(f"预测权重：{weights}\n三模态配对完成，共 {len(samples)} 组；结果写入 {output}")
    model_batch = backend_metadata["effective_model_batch"]
    print(f"FP32 预测：模型批次={model_batch}，调度批次={config.batch}，读取线程={config.workers}，保存线程={config.save_workers}，预取={config.prefetch_batches} 批")
    started: float = time.perf_counter()
    completed: int = 0
    reported: int = 0
    batch_sizes: Counter[int] = Counter()
    pending_saves: deque[Future[None]] = deque()
    previous_cv_threads: int = cv2.getNumThreads()
    # 外层已经并行读写，避免每个线程再启动一组 OpenCV 线程争抢 CPU。
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="模态读取") as reader, \
                ThreadPoolExecutor(max_workers=config.save_workers, thread_name_prefix="结果保存") as writer:
            for batch in prediction_batches(samples, config, reader, load_sample):
                groups: dict[tuple[int, ...], list[PredictionSample]] = {}
                for sample in batch:
                    # 按实际输入尺寸分组，保持各模型的画布契约。
                    groups.setdefault(sample.image.shape, []).append(sample)
                for group in groups.values():
                    results = forward_batch(group)
                    if model_batch == 1:
                        batch_sizes[1] += len(group)
                    else:
                        batch_sizes[len(group)] += 1
                    for sample, result in zip(group, results, strict=True):
                        while pending_saves and (pending_saves[0].done() or len(pending_saves) >= config.batch * config.prefetch_batches):
                            pending_saves.popleft().result()
                        pending_saves.append(writer.submit(save_prediction, result, sample.path.name, output,
                                                           config))
                    completed += len(group)
                if config.verbose and (completed - reported >= config.log_every or completed == len(samples)):
                    elapsed = time.perf_counter() - started
                    print(f"已推理 {completed}/{len(samples)}，流水线累计 {elapsed:.1f} 秒；图片与标签后台保存中")
                    reported = completed
            # 所有保存异常必须在打包前传播，不能产生缺图缺标签的正式提交包。
            while pending_saves:
                pending_saves.popleft().result()
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(f"当前 batch={config.batch} 超出可用显存；请减小 batch 并指定新的 --output，已有结果保留") from error
    finally:
        cv2.setNumThreads(previous_cv_threads)
    elapsed = time.perf_counter() - started
    metadata: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(), "entrypoint": entrypoint,
        "weights": weights.name, "phase": "round2", "source_hashes": source_hashes,
        "run": weights.parent.parent.name, "weights_sha256": weight_hash, "source": source.name,
        "images": len(samples), "classes": list(CLASS_NAMES), "ultralytics": ultralytics.__version__,
        "torch": torch.__version__, "imgsz": config.imgsz, "height": config.height,
        "conf": config.conf, "visual_conf": config.visual_conf, "max_det": config.max_det,
        "device": config.device, "batch": config.batch, "quantize": 32,
        "inference_precision": "float32", "tf32": False, "augment": False,
        "sensor_augmentation_inference": False, "input_normalization": "divide_255_once",
        "workers": config.workers, "save_workers": config.save_workers,
        "prefetch_batches": config.prefetch_batches, "png_compression": config.png_compression,
        "actual_batch_sizes": dict(sorted(batch_sizes.items())),
        "pipeline_seconds": elapsed, "pipeline_images_per_second": len(samples) / elapsed,
        "configuration": json.loads(json.dumps(asdict(config), default=str)),
        **backend_metadata,
    }
    archive = build_submission(output, [sample[0].name for sample in samples], config.save)
    with archive.open("rb") as handle:
        metadata["submission_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    print(f"读取、推理、绘图与保存共 {elapsed:.1f} 秒，平均 {len(samples) / elapsed:.2f} 组/秒（不含模型加载与 ZIP 打包）")
    package = build_round2_package(config, archive)
    record = package / f"{TEAM_ID}-代码与数据" / "prediction.json"
    with record.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    expected = {"images", "labels", package.name}
    actual = {path.name for path in output.iterdir()}
    if actual != expected:
        raise RuntimeError(f"预测输出结构不符合约定：期望{sorted(expected)}，实际{sorted(actual)}")
    print(f"预测及复赛材料已完成：{output}")
    print(f"结果根目录只含 images、labels 和 {package.name}；submission.zip 位于团队文件夹的代码与数据目录。")
