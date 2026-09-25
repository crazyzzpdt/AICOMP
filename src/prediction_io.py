"""三模态预测共用的读取调度、结果保存和赛事材料，不选择或加载模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
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
    phase: str = "round2"
    export_materials: bool = True
    materials_only: bool = False
    team_id: str = ""
    team_name: str = ""
    captain: str = ""
    weights_url: str = ""
    technical_report: Path | None = None


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
    archive: Path = output / "比赛提交内容" / "submission.zip"
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


def export_round2_materials(config: OutputConfig, metadata: dict[str, object], archive: Path, source_paths: tuple[str, ...], entrypoint: str) -> Path:
    """分离导出模型和源码，并如实列出未备齐或未验证的复赛材料。

    Note:
        不复制赛事图像、标签、虚拟环境、Git或整个runs目录。训练源码取所选
        权重运行的code快照，不以当前入口冒充历史模型训练代码。
        本函数不运行模型、不上传网盘，也不把文件齐全等同于评审环境复现通过。
    """
    if metadata.get("source_hashes") != prediction_source_hashes(source_paths):
        raise ValueError("预测源码已变化或原记录缺少源码摘要；请恢复原源码后补材料，不能伪称代码与结果一致")

    def copy_file(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)

    def copy_sources(source: Path, target: Path) -> int:
        count = 0
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if path.is_symlink() or any(part in {".git", "__pycache__", ".venv"} for part in relative.parts):
                continue
            if path.is_file() and (path.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".toml", ".md", ".txt"}
                                  or path.name in {"LICENSE", "NOTICE"}):
                copy_file(path, target / relative)
                count += 1
        return count

    pending: list[str] = []
    identity = (config.team_id, config.team_name, config.captain)
    complete_identity = all(identity)
    if not complete_identity:
        pending.append("补充参赛团队编号、团队名称、队长姓名，并按附件2命名根目录")
    team_id = config.team_id or "待填团队编号"
    root_name = "-".join((config.team_id, config.team_name, "复赛", config.captain)) if complete_identity else "复赛材料待补队伍信息"
    # 每次仅打包也另建容器；不覆写上一次已核对的材料。
    container = config.output / "比赛提交内容" / f"审阅材料_{datetime.now():%Y%m%d_%H%M%S_%f}"
    root = container / root_name
    root.mkdir(parents=True, exist_ok=False)
    code = root / f"{team_id}-代码与数据"
    model_dir = root / f"{team_id}-模型文件"
    code.mkdir()
    model_dir.mkdir()
    copy_file(config.weights, model_dir / config.weights.name)
    with (model_dir / config.weights.name).open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != metadata["weights_sha256"]:
            raise ValueError("导出的模型与预测使用的模型摘要不一致，材料未完成")
    copy_file(archive, root / "submission.zip")
    copy_file(config.output / "prediction.json", root / "prediction.json")
    for name in source_paths:
        source = PROJECT_ROOT / name
        if source.is_dir():
            copy_sources(source, code / name)
        else:
            copy_file(source, code / name)
    copy_sources(PROJECT_ROOT / "docs", code / "docs")
    for name in ("pyproject.toml", "uv.lock"):
        if (PROJECT_ROOT / name).is_file():
            copy_file(PROJECT_ROOT / name, code / name)
    # 当前安装包含项目所需YOLO接口，不能仅写一个pip版本号就声称源码一致。
    copy_sources(Path(ultralytics.__file__).parent, code / "ultralytics")
    run = config.weights.parent.parent
    snapshot = run / "code"
    archived_run = (PROJECT_ROOT / "tools" / "archive" / "run_snapshots" / run.relative_to(PROJECT_ROOT / "runs")
                    if run.is_relative_to(PROJECT_ROOT / "runs") else run)
    archived_snapshot = archived_run / "code"
    if not snapshot.is_dir() and archived_snapshot.is_dir():
        snapshot = archived_snapshot
    if (snapshot / "train1.py").is_file() or (snapshot / "train2.py").is_file():
        copy_sources(snapshot, code / "训练源码")
    elif (run / "train1.py").is_file() or (run / "main.py").is_file() or (archived_run / "train1.py").is_file() or (archived_run / "main.py").is_file():
        # v4/v5将源码直接留在运行根目录，只复制同层Python，不递归带入图像和权重。
        source_run = run if list(run.glob("*.py")) else archived_run
        for path in sorted(source_run.glob("*.py")):
            if not path.is_symlink():
                copy_file(path, code / "训练源码" / path.name)
        pending.append("已导出旧运行根目录Python快照；需按其导入核对依赖完整性和当时框架版本")
    else:
        pending.append("所选模型缺少训练入口快照：从该版本Git恢复真实训练源码，不能使用当前入口替代")
    for name in ("args.yaml", "optimization_recipe.json"):
        if (run / name).is_file():
            copy_file(run / name, code / "训练记录" / name)
    packages = ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "pillow", "PyYAML",
                "scipy", "matplotlib", "tqdm", "psutil", "requests", "polars", "ultralytics-thop",
                "faster-coco-eval")
    if entrypoint == "predict2.py":
        packages += ("loguru", "tensorboard", "transformers", "calflops")
    dependencies: list[str] = []
    for package in packages:
        try:
            dependencies.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            pending.append(f"未发现{package}的安装元数据，需核对依赖")
    (code / "requirements.txt").write_text("\n".join(dependencies) + "\n", encoding="utf-8")
    if config.technical_report is not None:
        copy_file(config.technical_report, root / f"{team_id}-技术方案.PDF")
    else:
        template = PROJECT_ROOT / "docs" / "技术方案.md"
        draft = template.read_text(encoding="utf-8") if template.is_file() else "# 复赛技术方案（待定稿）\n"
        # 报告位于材料根目录，关联知识文档位于代码目录，不留下迁移后的断链。
        draft = re.sub(r'\]\(([^():\n]+\.md(?:#[^()\n]*)?)\)',
                       lambda match: f']({team_id}-代码与数据/docs/{match.group(1)})', draft)
        actual = {key: metadata.get(key) for key in ("run", "weights", "weights_sha256", "phase", "images", "backend",
                  "architecture", "fusion_version", "imgsz", "height", "conf", "iou", "max_det", "preprocess")}
        draft += "\n\n## 本次实际提交候选（自动记录）\n\n```json\n" + json.dumps(actual, ensure_ascii=False, indent=2) + "\n```\n"
        (root / f"{team_id}-技术方案.md").write_text(draft, encoding="utf-8")
        pending.append("已按用户要求提供技术方案Markdown草稿，待核对实际提交模型并定稿转PDF；MD不是PDF替代证明")
    if not config.weights_url:
        pending.append("补充仅供赛事评审访问的模型权重下载链接，不使用GitHub公开存储")
    pending.append("在评审目标环境按说明复现；当前仅打包，未验证依赖安装或训练/推理可运行性")
    pending.append("训练复现须另行准备官方数据、既有清洗清单/审计及该版本初始化权重；不随代码复制大文件")
    height = metadata.get("height", metadata["imgsz"])
    iou = metadata.get("iou")
    iou = 0.7 if iou is None else iou
    command = (f'python {entrypoint} --weights "../{team_id}-模型文件/{config.weights.name}" '
               f'--source "官方复赛测试集路径" --output predict_review --phase round2 '
               f'--imgsz {metadata["imgsz"]} --height {height} --batch {metadata["batch"]} '
               f'--conf {metadata["conf"]} --device {metadata["device"]} '
               f'--max-det {metadata["max_det"]} --expected-count {metadata["images"]} '
               f'--no-export-materials')
    if entrypoint == "predict1.py":
        command += (f' --iou {iou} --yolo-profile {metadata.get("yolo_profile") or "v4"}'
                    f' {"--rect" if metadata["rect"] else "--no-rect"}'
                    f' {"--multi-label" if metadata.get("multi_label") else "--no-multi-label"}')
        for key in ("agnostic_nms", "channels_last", "stream"):
            command += f' --{"" if metadata.get(key) else "no-"}{key.replace("_", "-")}'
        if metadata.get("classes_filter") is not None:
            command += " --classes " + " ".join(map(str, metadata["classes_filter"]))
    description = ("# 复赛项目说明\n\n"
        "本项目使用可见光、红外和深度图进行12类目标检测。仅单模型预测，不做投票集成。\n\n"
        f"模型运行：`{metadata['run']}`；SHA256：`{metadata['weights_sha256']}`。\n\n"
        f"权重下载链接：{config.weights_url or '待补（权重已单独放在模型文件目录）'}。\n\n"
        "## 环境与运行\n\n"
        f"生成环境Python={sys.version.split()[0]}，PyTorch={metadata['torch']}；GPU/CUDA平台须匹配。\n"
        "按requirements.txt准备依赖；PyTorch CUDA轮子来源见pyproject.toml，不能以CPU版本冒充。\n"
        "ultralytics/是本次预测实际使用的源码副本，优先于普通pip包；不含虚拟环境。\n"
        "先在联网准备环境阶段安装依赖，正式推理离线运行，不自动下载权重或依赖。\n\n"
        f"在本目录执行复现预测命令（替换官方测试集路径）：\n\n```powershell\n{command}\n```\n\n"
        "## 文件与训练复现\n\n"
        f"{entrypoint}是本模型的预测入口；src/保存对应框架实现和共用输出处理。\n"
        "训练源码/为所选模型的运行快照（如有），训练记录/为其实际配置；不要使用当前入口替代。\n"
        "训练前按原配置准备官方数据、既有审计与官方初始化权重，并在训练源码目录执行该版本真实训练入口；"
        "路径和源码指纹须按该快照处理，具体缺项见材料清单。代码与数据目录名沿用附件，实际不含数据。\n\n"
        "训练运行时须将本代码根目录加入PYTHONPATH，使自定义ultralytics源码可见；"
        "当前预测框架副本不自动等同于历史训练框架，须核对该模型原记录。docs/保留技术依据与历史适用范围。\n\n"
        "## 提交与注意事项\n\n"
        "根目录submission.zip与排行榜提交结果相同，只含同名六列TXT；空检测也有空TXT。\n"
        "模型文件与代码分开放置；大数据/环境不入代码目录。测试集不用于训练、标注或人工改结果。\n"
        "仅通过报名系统指定渠道向评审分享材料，不能把赛事数据、权重包上传公开GitHub。\n"
        "本材料包未经目标机器复现验证，不能据文件存在宣称训练复现成功。\n")
    (code / "README.md").write_text(description, encoding="utf-8")
    manifest = {"phase": "round2", "status": "needs_review", "pending": pending,
                "weights_sha256": metadata["weights_sha256"], "inference_verified_on_reviewer_machine": False,
                "files": []}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                manifest["files"].append({"path": path.relative_to(root).as_posix(),
                                          "sha256": hashlib.file_digest(handle, "sha256").hexdigest()})
    (root / "材料清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "待完善.md").write_text("# 复赛材料待核对\n\n" + "\n".join(f"- {item}" for item in pending) + "\n", encoding="utf-8")
    print(f"复赛审阅材料已导出：{root}\n尚有{len(pending)}项需核对，见待完善.md；未上传、未宣称完整交付。")
    return root


def parse_arguments(config: OutputConfig, argv: list[str]) -> OutputConfig:
    """仅覆盖显式提供的参数；命令行名称与入口字段一一对应。"""
    parser = argparse.ArgumentParser(description="三模态预测，默认值见对应入口", argument_default=argparse.SUPPRESS)
    paths = {"weights", "source", "output", "project", "technical_report", "data"}
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
    """并行读取、批量推理、后台保存，全部完成后生成提交 ZIP。

    Args:
        config: 入口文件中显式设置的预测参数。
        create_backend: 对应框架的加载函数，返回读图、前向函数及真实执行记录。
        source_paths: 该入口参与摘要校验与材料导出的源码路径。
        entrypoint: 本次入口文件名，用于生成复现命令。

    Note:
        GPU 只在主线程执行。后台保存仅持有 CPU 结果，异常会阻止生成正式 ZIP。
        不自动试跑探测显存，不重训，不修改数据或覆盖旧输出。
    """
    if config.phase not in {"round1", "round2"}:
        raise ValueError("phase必须是round1或round2")
    for value in (config.team_id, config.team_name, config.captain):
        if value and (re.search(r'[<>:"/\\|?*\x00-\x1f]', value) or value.strip() != value or
                      value.endswith(".") or value in {".", ".."}):
            raise ValueError("队伍信息不能含路径分隔符、首尾空格或Windows非法文件名字符")
    if config.technical_report is not None:
        report = (PROJECT_ROOT / config.technical_report).resolve(strict=True)
        with report.open("rb") as handle:
            if report.suffix.lower() != ".pdf" or handle.read(5) != b"%PDF-":
                raise ValueError("技术方案须为真实PDF文件，不把Markdown改后缀当作PDF")
        config = replace(config, technical_report=report)
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
    if output.exists() and not config.materials_only:
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
    if config.materials_only:
        metadata = json.loads((output / "prediction.json").read_text(encoding="utf-8"))
        archive = output / "比赛提交内容" / "submission.zip"
        if (config.phase != "round2" or metadata.get("phase") != "round2" or
                metadata.get("weights_sha256") != weight_hash or metadata.get("images") != len(samples) or
                metadata.get("source") != source.name):
            raise ValueError("仅补材料要求原复赛记录、同一权重与测试集；不能把初赛结果重新标成复赛")
        with archive.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != metadata.get("submission_sha256"):
                raise ValueError("排行榜结果包已改变或缺少摘要，拒绝与其他模型材料混用")
        export_round2_materials(replace(config, weights=weights, source=source, output=output), metadata, archive, source_paths, entrypoint)
        return
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
        "weights": weights.name, "phase": config.phase, "source_hashes": source_hashes,
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
    with (output / "prediction.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"读取、推理、绘图与保存共 {elapsed:.1f} 秒，平均 {len(samples) / elapsed:.2f} 组/秒（不含模型加载与 ZIP 打包）")
    print(f"排行榜结果包校验通过，共 {len(samples)} 个TXT：{archive}\n只将此ZIP上传结果入口，代码/权重不混入TXT包。")
    if config.phase == "round2" and config.export_materials:
        export_round2_materials(config, metadata, archive, source_paths, entrypoint)
