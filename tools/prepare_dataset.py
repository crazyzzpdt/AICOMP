"""审阅并清洗官方新版标签，将三模态按留痕计划落到 datasets。

默认仅生成审阅记录；确认后用 --apply-review 指定记录目录进行备份和落位。
v8 使用 --audit-v8 导出来源审计，--review-v8 导出弱类审阅图；
仅在 split_plan.json 已审阅后用 --apply-v8 落位，并完整备份旧划分。
重新官方下载后使用 --refresh-official，保留现有划分，比较全部来源并重建副本。
正式训练由 main.py 启动，原始数据不修改。
在项目根目录执行：uv run python -m tools.prepare_dataset
"""

# 内置库
import argparse
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil

# 自己的模块：以模块方式运行，避免工具脚本修改sys.path。
from aic.data import CLASS_NAMES, IMAGE_SUFFIXES, file_hash


# 忽略小数舍入量级误差，只裁剪超出归一化边界万分之一的可见框。
BOUNDARY_TOLERANCE: float = 0.0001
# 2026-09-17 对原图审阅确认：大客车误标 person；仅对完全匹配的官方原行修订。
REVIEWED_CLASS_CORRECTIONS: dict[str, tuple[int, str, int, str]] = {
    "000003_026_00000001.txt": (1, "0 0.623698 0.336574 0.752604 0.678704", 6,
                              "原图中框覆盖江铃大客车及后视镜，按赛事四轮汽车定义改为 car；后方行人框保留"),
}


def correct_reviewed_classes(filename: str, text: str) -> tuple[str, list[dict[str, object]]]:
    """只执行已逐图确认且与原标签严格匹配的类别修订。"""
    correction = REVIEWED_CLASS_CORRECTIONS.get(filename)
    if correction is None:
        return text, []
    number, expected, category, reason = correction
    rows = text.splitlines()
    if len(rows) < number or rows[number - 1] != expected:
        raise ValueError(f"人工修订源行与已审阅版本不同：{filename}:{number}")
    updated = f"{category} " + expected.split(" ", 1)[1]
    rows[number - 1] = updated
    return "\n".join(rows) + "\n", [{"line": number, "action": "class_correction", "before": expected,
                                      "after": updated, "reason": reason}]


def clean_label_text(text: str) -> tuple[str, list[dict[str, object]]]:
    """裁剪明确越出画面的框并去除完全重复行，保留类别与非重复目标。

    Args:
        text: 官方新版 YOLO 标签内容。

    Returns:
        清洗后内容及逐行动作、原值和新值。

    Raises:
        ValueError: 标注结构非法，或框与图像无交集。
    """
    output: list[str] = []
    changes: list[dict[str, object]] = []
    seen: set[tuple[float, ...]] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        values = tuple(float(value) for value in line.split())
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"第 {number} 行不是五列有限数值")
        category, cx, cy, width, height = values
        if not category.is_integer() or not 0 <= category < len(CLASS_NAMES) or min(width, height) <= 0:
            raise ValueError(f"第 {number} 行类别或尺寸非法")
        if values in seen:
            changes.append({"line": number, "action": "duplicate", "before": line, "after": None})
            continue
        seen.add(values)
        x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        left, top, right, bottom = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
        if right <= left or bottom <= top:
            raise ValueError(f"第 {number} 行框完全位于图像之外，需人工处理")
        if max(-x1, -y1, x2 - 1, y2 - 1) > BOUNDARY_TOLERANCE:
            coordinates = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
            updated = f"{int(category)} " + " ".join(f"{value:.9f}" for value in coordinates)
            changes.append({"line": number, "action": "clip", "before": line, "after": updated})
            output.append(updated)
        else:
            output.append(line)
    return ("\n".join(output) + "\n" if changes and output else text), changes


def inspect_cleaning(dataset: Path, raw: Path, official_labels: Path) -> list[dict[str, object]]:
    """在任何替换前校验全部配对、尺寸、标签来源并生成清洗计划。"""
    from PIL import Image

    records: list[dict[str, object]] = []
    stems: set[str] = set()
    for split in ("train", "val"):
        images = _image_files(dataset / split / "images")
        if not images:
            raise ValueError(f"{split} 图像目录为空")
        label_stems = {path.stem for path in (dataset / split / "labels").glob("*.txt")}
        if label_stems != {path.stem for path in images}:
            raise ValueError(f"{split} 图片与标签未一一对应")
        for image in images:
            if image.stem in stems:
                raise ValueError(f"跨划分重复词干：{image.stem}")
            stems.add(image.stem)
            sources = [raw / modality / image.name for modality in ("visible", "infrared", "depth")]
            label = dataset / split / "labels" / f"{image.stem}.txt"
            source_label = official_labels / label.name
            sizes: list[tuple[int, int]] = []
            hashes: dict[str, str] = {}
            for source in sources:
                with Image.open(source) as opened:
                    sizes.append(opened.size)
                    opened.verify()
                hashes[source.parent.name] = file_hash(source)
            if len(set(sizes)) != 1 or file_hash(image) != hashes["visible"]:
                raise ValueError(f"图像来源或三模态尺寸异常：{image.name}")
            original = source_label.read_text(encoding="utf-8-sig")
            corrected, manual_changes = correct_reviewed_classes(label.name, original)
            cleaned, changes = clean_label_text(corrected)
            changes = manual_changes + changes
            current = label.read_text(encoding="utf-8-sig")
            if current not in (original, cleaned):
                raise ValueError(f"存在未记录的手工标签改动，请先核验：{label}")
            for modality in ("infrared", "depth"):
                destination = dataset / split / modality / image.name
                if destination.exists() and file_hash(destination) != hashes[modality]:
                    raise ValueError(f"本地模态与官方原图不一致：{destination}")
            records.append({"split": split, "image": image.name, "size": list(sizes[0]),
                            "source_hashes": hashes, "official_label_sha256": file_hash(source_label),
                            "before_sha256": file_hash(label), "original": original, "cleaned": cleaned,
                            "changes": changes})
    return records


def write_review_images(dataset: Path, records: list[dict[str, object]], report: Path) -> None:
    """导出待改边界框局部图与索引，支持逐项人工确认。"""
    from PIL import Image, ImageDraw

    review = report / "review"
    review.mkdir()
    tiles: list[object] = []
    index: list[dict[str, object]] = []
    for record in records:
        for change in record["changes"]:
            if change["action"] != "clip":
                continue
            source = dataset / record["split"] / "images" / record["image"]
            with Image.open(source) as opened:
                frame = opened.convert("RGB")
            category, cx, cy, width, height = map(float, change["before"].split())
            w, h = frame.size
            box = ((cx - width / 2) * w, (cy - height / 2) * h,
                   (cx + width / 2) * w, (cy + height / 2) * h)
            crop = (max(0, int(box[0]) - 40), max(0, int(box[1]) - 40),
                    min(w, math.ceil(box[2]) + 40), min(h, math.ceil(box[3]) + 40))
            draw = ImageDraw.Draw(frame)
            draw.rectangle(tuple(max(0, min(value, (w - 1) if position % 2 == 0 else (h - 1)))
                                 for position, value in enumerate(box)), outline="red", width=3)
            frame = frame.crop(crop)
            frame.thumbnail((390, 230))
            tile = Image.new("RGB", (400, 265), "white")
            tile.paste(frame, (0, 30))
            number = len(index) + 1
            ImageDraw.Draw(tile).text((5, 5), f"{number}: {record['split']} {CLASS_NAMES[int(category)]} row {change['line']}", fill="black")
            tile.save(review / f"{number:03d}.jpg")
            tiles.append(tile)
            index.append({"review_id": number, "split": record["split"], "image": record["image"], **change})
    for start in range(0, len(tiles), 12):
        sheet = Image.new("RGB", (1600, 795), "white")
        for offset, tile in enumerate(tiles[start:start + 12]):
            sheet.paste(tile, ((offset % 4) * 400, (offset // 4) * 265))
        sheet.save(review / f"sheet_{start // 12 + 1}.jpg")
    (report / "review_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_cleaning(dataset: Path, raw: Path, official_labels: Path, report: Path, records: list[dict[str, object]]) -> None:
    """备份后原子替换标签，并将三模态落到 datasets 的两个划分中。

    Note:
        records 必须来自已审阅的 inspect_cleaning；源指纹与当前标签再次核验，
        以防审阅期间文件变化。备份与逐行记录先于任何标签替换保存。
    """
    for record in records:
        label = dataset / record["split"] / "labels" / (Path(record["image"]).stem + ".txt")
        if file_hash(label) != record["before_sha256"]:
            raise ValueError(f"审阅后标签已改变：{label}")
        if file_hash(official_labels / label.name) != record["official_label_sha256"]:
            raise ValueError(f"官方源标注已改变：{label.name}")
        for modality, fingerprint in record["source_hashes"].items():
            if file_hash(raw / modality / record["image"]) != fingerprint:
                raise ValueError(f"审阅后原始模态发生变化：{modality}/{record['image']}")
    backup = report / "before"
    backup.mkdir()
    shutil.copy2(dataset / "data.yaml", backup / "data.yaml")
    for split in ("train", "val"):
        shutil.copytree(dataset / split / "labels", backup / split / "labels")
    (report / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    changed_files: int = 0
    for record in records:
        split = record["split"]
        label = dataset / split / "labels" / (Path(record["image"]).stem + ".txt")
        if label.read_text(encoding="utf-8-sig") != record["cleaned"]:
            temporary = label.with_suffix(".txt.cleaning")
            temporary.write_text(record["cleaned"], encoding="utf-8")
            temporary.replace(label)
            changed_files += 1
        for modality in ("infrared", "depth"):
            destination = dataset / split / modality / record["image"]
            destination.parent.mkdir(exist_ok=True)
            _link_or_validate(raw / modality / record["image"], destination)
        record["after_sha256"] = file_hash(label)
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    content = ("train: train/images\nval: val/images\ninfrared:\n  train: train/infrared\n  val: val/infrared\n"
               "depth:\n  train: train/depth\n  val: val/depth\nchannels: 5\nnames:\n" + names + "\n")
    temporary_yaml = dataset / "data.yaml.cleaning"
    temporary_yaml.write_text(content, encoding="utf-8")
    temporary_yaml.replace(dataset / "data.yaml")
    # 旧缓存移入审计目录，避免同长度标签变化被框架的尺寸哈希漏掉。
    for split in ("train", "val"):
        cache = dataset / split / "labels.cache"
        if cache.exists():
            cache.replace(backup / split / "labels.cache")
    (report / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "applied", "images": len(records), "changed_files": changed_files,
               "clipped_boxes": sum(change["action"] == "clip" for record in records for change in record["changes"]),
               "duplicate_rows": sum(change["action"] == "duplicate" for record in records for change in record["changes"]),
               "class_corrections": sum(change["action"] == "class_correction" for record in records for change in record["changes"])}
    (report / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"清洗落位完成：{len(records)} 组，修改 {changed_files} 个标签；备份与记录：{report}")


# 一、文件复制与校验
def _image_files(images_dir: Path) -> list[Path]:
    """返回数据集划分中的真实图像，忽略 Ultralytics 的 NPY 缓存。"""
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _remove_image_cache(images_dir: Path) -> None:
    """删除与旧通道配置绑定的 Ultralytics 图像缓存。"""
    for cache_path in images_dir.glob("*.npy"):
        cache_path.unlink()


def _link_or_validate(source: Path, destination: Path) -> None:
    """创建图像硬链接；已有文件仅在指向同一文件时复用。"""
    if destination.exists():
        if destination.samefile(source):
            return
        raise FileExistsError(f"目标图像已存在但不是原始图像的副本：{destination}")
    try:
        destination.hardlink_to(source)
    except OSError:
        # 跨卷或文件系统不支持硬链接时退化为复制，不影响旧训练集。
        shutil.copy2(source, destination)


def _copy_label_or_validate(source: Path, destination: Path) -> None:
    """原子替换为新版标签，同时断开旧标签可能存在的硬链接。"""
    if destination.exists():
        try:
            if destination.samefile(source):
                return
        except OSError:
            pass
    temporary_path = destination.with_suffix(".txt.new")
    shutil.copy2(source, temporary_path)
    temporary_path.replace(destination)


def _data_yaml(output_dir: Path, raw_dataset: Path) -> str:
    """生成供自定义五通道数据集读取的 YAML 配置。"""
    infrared_dir = Path(os.path.relpath(raw_dataset / "infrared", output_dir)).as_posix()
    depth_dir = Path(os.path.relpath(raw_dataset / "depth", output_dir)).as_posix()
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    return (
        "train: train/images\n"
        "val: val/images\n"
        f"infrared: {infrared_dir}\n"
        f"depth: {depth_dir}\n"
        "channels: 5\n"
        "names:\n"
        f"{names}\n"
    )


# 二、构建新版标注数据集
def prepare_multimodal_dataset(
    old_dataset: Path, raw_dataset: Path, new_labels: Path, output_dir: Path
) -> Path:
    """保持既有图像划分，并将标签切换为官方更新版本。

    图像目录保持原位，标签通过临时文件原子替换，避免写入旧标签硬链接指向的
    官方原文件。原始旧标签仍保存在官方下载目录中。

    Args:
        old_dataset: 现有 RGB 训练集根目录，用于复用已核验的 train/val 划分。
        raw_dataset: 包含 infrared 与 depth 文件夹的官方下载训练集目录。
        new_labels: 官方更新后的 YOLO 标签目录。
        output_dir: 训练数据根目录，正常应与 ``old_dataset`` 相同。

    Returns:
        已更新的三模态训练集根目录。

    Raises:
        FileNotFoundError: 图像、对应模态或新版标签缺失。
        FileExistsError: 已有图像不能安全复用。
    """
    for split in ("train", "val"):
        source_images = old_dataset / split / "images"
        if not source_images.is_dir():
            raise FileNotFoundError(f"找不到现有 {split} 图像目录：{source_images}")
        destination_images = output_dir / split / "images"
        destination_labels = output_dir / split / "labels"
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)
        _remove_image_cache(destination_images)

        for image_path in _image_files(source_images):
            stem = image_path.stem
            infrared_path = raw_dataset / "infrared" / image_path.name
            depth_path = raw_dataset / "depth" / image_path.name
            label_path = new_labels / f"{stem}.txt"
            for required_path, description in (
                (infrared_path, "红外图"),
                (depth_path, "深度图"),
                (label_path, "新版标签"),
            ):
                if not required_path.is_file():
                    raise FileNotFoundError(f"{description}缺失：{required_path}")
            _link_or_validate(image_path, destination_images / image_path.name)
            _copy_label_or_validate(label_path, destination_labels / label_path.name)

    yaml_path = output_dir / "data.yaml"
    yaml_content = _data_yaml(output_dir, raw_dataset)
    if not yaml_path.exists() or yaml_path.read_text(encoding="utf-8") != yaml_content:
        temporary_path = yaml_path.with_suffix(".yaml.new")
        temporary_path.write_text(yaml_content, encoding="utf-8")
        temporary_path.replace(yaml_path)
    return output_dir


def audit_v8(dataset: Path, raw: Path, official_labels: Path) -> None:
    """核验全部官方训练样本并导出场景缩略图，供有记录地调整验证划分。

    Note:
        不读取测试集，不执行模型；标签语义不确定时只记录，不自动纠正。
    """
    from PIL import Image, ImageDraw

    records = inspect_cleaning(dataset, raw, official_labels)
    report = dataset.parent / "runs/dataset_cleaning" / f"v8_{datetime.now():%Y%m%d_%H%M%S}"
    report.mkdir(parents=True, exist_ok=False)
    (report / "proposed.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        stem = Path(record["image"]).stem
        key = stem.rsplit("_", 1)[0] if "_" in stem else f"numeric:{stem}"
        groups.setdefault(key, []).append(record)
    group_summary = []
    for key, members in sorted(groups.items()):
        counts = Counter(int(line.split()[0]) for row in members for line in row["cleaned"].splitlines() if line.strip())
        group_summary.append({"group": key, "images": [row["image"] for row in members],
                              "splits": sorted({row["split"] for row in members}), "class_boxes": dict(counts)})
    (report / "groups.json").write_text(json.dumps(group_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    jpg = sorted((row for row in records if Path(row["image"]).suffix.lower() in {".jpg", ".jpeg"}), key=lambda row: row["image"])
    (report / "jpg_index.json").write_text(json.dumps([row["image"] for row in jpg], indent=2), encoding="utf-8")
    for start in range(0, len(jpg), 49):
        sheet = Image.new("RGB", (1400, 980), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(jpg[start:start + 49]):
            x, y = offset % 7 * 200, offset // 7 * 140
            with Image.open(raw / "visible" / row["image"]) as opened:
                tile = opened.convert("RGB")
            tile.thumbnail((196, 110))
            sheet.paste(tile, (x, y + 25))
            draw.text((x + 2, y + 3), f"{start + offset + 1}: {row['image']}", fill="black")
        sheet.save(report / f"jpg_scenes_{start // 49 + 1}.jpg")
    summary = {"images": len(records), "additional_label_changes": sum(
        (dataset / row["split"] / "labels" / (Path(row["image"]).stem + ".txt")).read_text(encoding="utf-8-sig") != row["cleaned"]
        for row in records), "jpg_images": len(jpg), "status": "audited_not_applied"}
    (report / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"v8 数据核验完成：{summary}；审阅目录：{report}")


def review_v8_weak_classes(report: Path, raw: Path) -> None:
    """导出船、球、三轮车的训练/验证场景及已有框，供人工检查而非模型评估。"""
    from PIL import Image, ImageDraw

    records = json.loads((report / "proposed.json").read_text(encoding="utf-8"))
    selected = sorted((row for row in records if any(int(line.split()[0]) in {1, 7, 11}
                      for line in row["cleaned"].splitlines() if line.strip())), key=lambda row: row["image"])
    (report / "weak_index.json").write_text(json.dumps([row["image"] for row in selected], indent=2), encoding="utf-8")
    for start in range(0, len(selected), 49):
        sheet = Image.new("RGB", (1750, 1190), "white")
        for offset, row in enumerate(selected[start:start + 49]):
            with Image.open(raw / "visible" / row["image"]) as opened:
                tile = opened.convert("RGB")
            width, height = tile.size
            draw = ImageDraw.Draw(tile)
            for line in row["cleaned"].splitlines():
                category, cx, cy, bw, bh = map(float, line.split())
                if int(category) in {1, 7, 11}:
                    draw.rectangle(((cx - bw / 2) * width, (cy - bh / 2) * height,
                                    (cx + bw / 2) * width, (cy + bh / 2) * height), outline="red", width=3)
            tile.thumbnail((246, 138))
            x, y = offset % 7 * 250, offset // 7 * 170
            sheet.paste(tile, (x, y + 30))
            caption = f"{start + offset + 1} {row['split']}: {Path(row['image']).stem}"
            ImageDraw.Draw(sheet).text((x + 2, y + 3), caption, fill="black")
        sheet.save(report / f"weak_scenes_{start // 49 + 1}.jpg")
    print(f"弱类审阅图已导出：{len(selected)} 张；{report}")


def apply_v8(dataset: Path, raw: Path, official_labels: Path, report: Path) -> None:
    """按已审阅清单落位清洗数据，保留完整旧 train/val 供恢复。

    Args:
        dataset: 项目 datasets 目录。
        raw: 官方训练图像目录。
        official_labels: 官方新版标签目录。
        report: 含 proposed.json 与 split_plan.json 的审计目录。

    Note:
        先核对源指纹，再构建完整暂存目录；只移动明确的 train/val 子目录。
        移动失败会回滚已切换目录，所有中间文件保留，不修改官方图像。
    """
    root = Path(__file__).resolve().parents[1]
    dataset, report = dataset.resolve(), report.resolve()
    if dataset != root / "datasets" or not report.is_relative_to(root / "runs/dataset_cleaning"):
        raise ValueError("v8 落位仅允许本项目 datasets 与 runs/dataset_cleaning 内的审计目录")
    records = json.loads((report / "proposed.json").read_text(encoding="utf-8"))
    plan = json.loads((report / "split_plan.json").read_text(encoding="utf-8"))
    if plan["proposed_sha256"] != file_hash(report / "proposed.json"):
        raise ValueError("划分计划与审阅数据指纹不一致")
    by_name = {row["image"]: row for row in records}
    if len(by_name) != len(records) or len(records) != 2000:
        raise ValueError("当前 v8 计划限定官方 2000 组，不能丢图或增加外部样本")
    assignments = {name: row["split"] for name, row in by_name.items()}
    changed_names: set[str] = set()
    for change in plan["moves"]:
        name, target = change["image"], change["to"]
        if name not in by_name or target not in {"train", "val"} or not change["reason"]:
            raise ValueError(f"划分变更缺少有效样本、目标或依据：{change}")
        if name in changed_names or change["from"] != by_name[name]["split"]:
            raise ValueError(f"重复迁移或原划分不匹配：{name}")
        changed_names.add(name)
        assignments[name] = target
    fingerprint_splits: dict[str, set[str]] = {}
    for row in records:
        fingerprint_splits.setdefault(row["source_hashes"]["visible"], set()).add(assignments[row["image"]])
    if any(len(splits) > 1 for splits in fingerprint_splits.values()):
        raise ValueError("新划分存在跨集完全相同的 RGB 图像，须将对应场景合并后再落位")
    # 全量来源核验在任何目录切换前完成，不接受审阅后的未知标签修改。
    current = inspect_cleaning(dataset, raw, official_labels)
    if current != records:
        raise ValueError("数据在审阅后发生变化，请重新生成审计而非覆盖")
    for split in ("train", "val"):
        present = {path.name for path in _image_files(dataset / split / "images")}
        if present != {row["image"] for row in records if row["split"] == split}:
            raise ValueError(f"{split} 图像清单在审阅后发生变化")
    staging, backup = report / "staging", report / "before"
    staging.mkdir(exist_ok=False)
    backup.mkdir(exist_ok=False)
    shutil.copy2(dataset / "data.yaml", backup / "data.yaml")
    counts: dict[str, object] = {}
    for split in ("train", "val"):
        for child in ("images", "infrared", "depth", "labels"):
            (staging / split / child).mkdir(parents=True)
        members = [row for row in records if assignments[row["image"]] == split]
        if not members:
            raise ValueError(f"新划分 {split} 为空")
        labels = Counter(int(line.split()[0]) for row in members for line in row["cleaned"].splitlines() if line.strip())
        counts[split] = {"images": len(members), "extensions": dict(Counter(Path(row["image"]).suffix.lower() for row in members)),
                         "class_boxes": {name: labels[index] for index, name in enumerate(CLASS_NAMES)}}
        for row in members:
            for child, modality in (("images", "visible"), ("infrared", "infrared"), ("depth", "depth")):
                _link_or_validate(raw / modality / row["image"], staging / split / child / row["image"])
            label = staging / split / "labels" / (Path(row["image"]).stem + ".txt")
            label.write_text(row["cleaned"], encoding="utf-8")
    manifest = {"status": "staged", "plan": plan, "counts": counts,
                "samples": [{"image": row["image"], "before_split": row["split"], "after_split": assignments[row["image"]],
                             "before_sha256": row["before_sha256"],
                             "after_sha256": file_hash(staging / assignments[row["image"]] / "labels" / (Path(row["image"]).stem + ".txt"))}
                            for row in records]}
    (report / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    completed: list[str] = []
    try:
        for split in ("train", "val"):
            source, saved, replacement = dataset / split, backup / split, staging / split
            if source.resolve().parent != dataset or saved.resolve().parent != backup or replacement.resolve().parent != staging:
                raise ValueError("目录解析超出预期位置，拒绝移动")
            source.rename(saved)
            try:
                replacement.rename(source)
            except BaseException:
                saved.rename(source)
                raise
            completed.append(split)
    except BaseException:
        for split in reversed(completed):
            (dataset / split).rename(staging / split)
            (backup / split).rename(dataset / split)
        raise
    manifest["status"] = "applied"
    (report / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "applied", "images": len(records), "counts": counts,
               "additional_label_changes": sum(bool(row["changes"]) for row in records),
               "moved_images": len(changed_names), "backup": "before"}
    (report / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"v8 清洗数据已落位 datasets：{counts}；旧划分和缓存完整保留：{backup}")


def refresh_official_dataset(dataset: Path, raw: Path, official_labels: Path) -> Path:
    """按现有划分重建最新官方下载副本，完整备份旧数据和缓存。

    Args:
        dataset: 项目 datasets 目录。
        raw: 重新下载的官方三模态目录。
        official_labels: 官方 new_labels_2000 标签目录，绝不读取旧 labels。

    Returns:
        本轮审计目录，包含来源对比、标签修改记录及完整旧数据。

    Note:
        不改变 train/val 划分；旧人工改类仅在图像指纹和源标签原行均匹配时沿用。
        新来源不匹配时保留官方类别并记入待审阅，不自动延用旧语义判断。
    """
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    dataset = dataset.resolve()
    if dataset != root / "datasets":
        raise ValueError("官方刷新仅允许项目 datasets 目录")
    assignments = {path.name: split for split in ("train", "val") for path in _image_files(dataset / split / "images")}
    if len(assignments) != 2000 or sum(len(_image_files(dataset / split / "images")) for split in ("train", "val")) != 2000:
        raise ValueError("现有划分必须完整包含 2000 张不重复图像")
    for modality in ("visible", "infrared", "depth"):
        if {path.name for path in _image_files(raw / modality)} != set(assignments):
            raise ValueError(f"重新下载的 {modality} 清单与现有划分不一致，禁止静默丢图或错配")
    if {path.stem for path in official_labels.glob("*.txt")} != {Path(name).stem for name in assignments}:
        raise ValueError("new_labels_2000 与全部训练图像词干不一一对应")
    prior_path = root / "runs/dataset_cleaning/v8_20260918_162439/proposed.json"
    prior = {row["image"]: row for row in json.loads(prior_path.read_text(encoding="utf-8"))} if prior_path.is_file() else {}
    report = root / "runs/dataset_cleaning" / f"official_refresh_{datetime.now():%Y%m%d_%H%M%S}"
    report.mkdir(parents=True, exist_ok=False)
    staging, backup = report / "staging", report / "before"
    staging.mkdir()
    backup.mkdir()
    shutil.copy2(dataset / "data.yaml", backup / "data.yaml")
    changed_sources: Counter[str] = Counter()
    pending: list[str] = []
    records: list[dict[str, object]] = []
    counts: dict[str, object] = {}
    for split in ("train", "val"):
        for child in ("images", "infrared", "depth", "labels"):
            (staging / split / child).mkdir(parents=True)
        boxes: Counter[int] = Counter()
        members = sorted(name for name, owner in assignments.items() if owner == split)
        for name in members:
            sizes: list[tuple[int, int]] = []
            source_hashes: dict[str, str] = {}
            old_hashes: dict[str, str | None] = {}
            for child, modality in (("images", "visible"), ("infrared", "infrared"), ("depth", "depth")):
                source, old = raw / modality / name, dataset / split / child / name
                with Image.open(source) as opened:
                    sizes.append(opened.size)
                    opened.verify()
                source_hashes[modality] = file_hash(source)
                old_hashes[modality] = file_hash(old) if old.is_file() else None
                changed_sources[modality] += old_hashes[modality] != source_hashes[modality]
                _link_or_validate(source, staging / split / child / name)
            if len(set(sizes)) != 1:
                raise ValueError(f"新下载三模态尺寸不一致：{name}")
            filename = Path(name).with_suffix(".txt").name
            source_label, old_label = official_labels / filename, dataset / split / "labels" / filename
            original = source_label.read_text(encoding="utf-8-sig")
            corrected, manual = original, []
            correction = REVIEWED_CLASS_CORRECTIONS.get(filename)
            if correction:
                number, expected, _, _ = correction
                matching = (prior.get(name, {}).get("source_hashes", {}).get("visible") == source_hashes["visible"]
                            and len(original.splitlines()) >= number and original.splitlines()[number - 1] == expected)
                if matching:
                    corrected, manual = correct_reviewed_classes(filename, original)
                else:
                    pending.append(filename)
            cleaned, changes = clean_label_text(corrected)
            destination = staging / split / "labels" / filename
            destination.write_text(cleaned, encoding="utf-8")
            old_text = old_label.read_text(encoding="utf-8-sig") if old_label.is_file() else None
            changed_sources["cleaned_labels"] += old_text != cleaned
            changed_sources["official_labels_vs_prior"] += prior.get(name, {}).get("official_label_sha256") != file_hash(source_label)
            boxes.update(int(line.split()[0]) for line in cleaned.splitlines() if line.strip())
            records.append({"image": name, "before_split": split, "after_split": split, "size": list(sizes[0]),
                            "source_hashes": source_hashes, "before_source_hashes": old_hashes,
                            "official_label_sha256": file_hash(source_label),
                            "before_sha256": file_hash(old_label) if old_label.is_file() else None,
                            "after_sha256": file_hash(destination), "changes": manual + changes})
        counts[split] = {"images": len(members), "extensions": dict(Counter(Path(name).suffix.lower() for name in members)),
                         "class_boxes": {name: boxes[index] for index, name in enumerate(CLASS_NAMES)}}
    fingerprint_splits: dict[str, set[str]] = {}
    for row in records:
        fingerprint_splits.setdefault(row["source_hashes"]["visible"], set()).add(row["after_split"])
    if any(len(splits) > 1 for splits in fingerprint_splits.values()):
        raise ValueError("最新下载来源出现跨集完全重复图像，需先审阅而非直接覆盖")
    manifest = {"status": "staged", "source": "official_redownload_new_labels_2000",
                "label_source": "数据集/训练集/new_labels_2000", "counts": counts,
                "changed_sources": dict(changed_sources), "pending_class_review": pending, "samples": records}
    manifest_path = report / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    completed: list[str] = []
    try:
        for split in ("train", "val"):
            source, saved, replacement = dataset / split, backup / split, staging / split
            if source.resolve().parent != dataset or saved.resolve().parent != backup or replacement.resolve().parent != staging:
                raise ValueError("目录解析超出预期位置，拒绝移动")
            source.rename(saved)
            try:
                replacement.rename(source)
            except BaseException:
                saved.rename(source)
                raise
            completed.append(split)
    except BaseException:
        for split in reversed(completed):
            (dataset / split).rename(staging / split)
            (backup / split).rename(dataset / split)
        raise
    manifest["status"] = "applied"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in manifest.items() if key != "samples"}
    summary["changed_label_files"] = sum(bool(row["changes"]) for row in records)
    summary["actions"] = dict(Counter(change["action"] for row in records for change in row["changes"]))
    summary["backup"] = "before"
    (report / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"官方重新下载数据已落位：{report}；来源差异：{dict(changed_sources)}；待复核类别：{pending}")
    return report


def main() -> None:
    """先导出审阅包，再按指定审阅包执行可追溯清洗。"""
    parser = argparse.ArgumentParser(description="审阅并清洗三模态训练标签")
    parser.add_argument("--refresh-official", action="store_true", help="从重新下载的三模态及 new_labels_2000 重建，完整备份当前副本")
    parser.add_argument("--apply-review", type=Path, help="已经逐图审阅的审计目录，必须包含 approved.json")
    parser.add_argument("--audit-v8", action="store_true", help="核验清洗数据并导出 JPG 场景审阅图，不运行模型")
    parser.add_argument("--apply-v8", type=Path, help="应用审阅目录中的 split_plan.json，整份旧划分搬入备份")
    parser.add_argument("--review-v8", type=Path, help="从已有审计导出弱类场景及标注，不运行模型")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset = project_root / "datasets"
    raw = project_root / "数据集" / "训练集" / "AIC2026_Train_2000"
    official_labels = project_root / "数据集" / "训练集" / "new_labels_2000"
    if args.refresh_official:
        refresh_official_dataset(dataset, raw, official_labels)
    elif args.audit_v8:
        audit_v8(dataset, raw, official_labels)
    elif args.review_v8:
        review_v8_weak_classes(args.review_v8.resolve(), raw)
    elif args.apply_v8:
        apply_v8(dataset, raw, official_labels, args.apply_v8.resolve())
    elif args.apply_review:
        report = args.apply_review.resolve()
        records = json.loads((report / "proposed.json").read_text(encoding="utf-8"))
        approval = json.loads((report / "approved.json").read_text(encoding="utf-8"))
        if approval.get("proposed_sha256") != file_hash(report / "proposed.json"):
            raise ValueError("审阅确认与当前清洗提案的指纹不一致")
        apply_cleaning(dataset, raw, official_labels, report, records)
    else:
        records = inspect_cleaning(dataset, raw, official_labels)
        report = project_root / "runs" / "dataset_cleaning" / datetime.now().strftime("%Y%m%d_%H%M%S")
        report.mkdir(parents=True, exist_ok=False)
        (report / "proposed.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        write_review_images(dataset, records, report)
        print(f"清洗审阅包已生成，共 {len(records)} 组：{report}")


if __name__ == "__main__":
    main()
