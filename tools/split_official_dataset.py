"""按类别覆盖和已知场景组缩减验证集，不修改任何官方标签或图像。

运行：uv run python -m tools.split_official_dataset [--apply]
默认生成计划；--apply核对全部来源摘要后移动200组三模态及标签，失败时回滚。
数值JPG仅复用历史人工场景记录，其余未知场景不能宣称已完成视觉去重。
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "runs/dataset_cleaning/official_labels_split_1700_300_20260924/manifest.json"
OUTPUT = ROOT / "runs/dataset_cleaning/official_labels_split_1900_100_v27"
CLASSES = ("person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball", "light", "garbage can", "uav", "tricycle")


def digest(path: Path) -> str:
    """读取文件摘要，不改动硬链接源文件。"""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_plan(audit: dict) -> dict:
    """仅使用原验证集的完整组，以整数规划保留100张及稀有类覆盖。"""
    rows = audit["samples"]
    parents = {row["image"]: row["image"] for row in rows}

    def find(name: str) -> str:
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    def merge(names: list[str]) -> None:
        for name in names[1:]:
            parents[find(name)] = find(names[0])

    prefixes = defaultdict(list)
    for name in parents:
        stem = Path(name).stem
        key = stem.rsplit("_", 1)[0].replace("_suppl", "") if "_" in stem else f"numeric:{stem}"
        prefixes[key].append(name)
    for names in prefixes.values():
        merge(names)
    reviewed = json.loads((ROOT / "runs/dataset_cleaning/v8_20260918_162439/split_plan.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for move in reviewed["moves"]:
        if move["image"] in parents:
            groups[move["group"]].append(move["image"])
    for names in groups.values():
        merge(names)
    groups = defaultdict(list)
    for row in rows:
        groups[find(row["image"])].append(row)
    candidates = [members for _, members in sorted(groups.items())
                  if all(row["after_split"] == "val" for row in members)]
    boxes, images = [], []
    for members in candidates:
        group_boxes, group_images = Counter(), Counter()
        for row in members:
            path = ROOT / "datasets/val/labels" / (Path(row["image"]).stem + ".txt")
            categories = [int(line.split()[0]) for line in path.read_text().splitlines() if line.strip()]
            group_boxes.update(categories)
            group_images.update(set(categories))
        boxes.append([group_boxes[i] for i in range(12)])
        images.append([group_images[i] for i in range(12)])
    box_array, image_array = np.array(boxes).T, np.array(images).T
    totals = box_array.sum(axis=1)
    # 稀有类保留至少一半可用框；其余类至少10框，并尽量覆盖3张含类图。
    minimum_boxes = np.where(totals <= 40, np.ceil(totals * 0.5), 10)
    minimum_images = np.minimum(3, image_array.sum(axis=1))
    if np.any(totals == 0):
        raise ValueError("完整验证候选组缺少类别，须审阅划分，不能退化为随机抽100张")
    sizes = np.array([len(members) for members in candidates])
    matrix = np.vstack([sizes, box_array, image_array])
    lower = np.concatenate(([100], minimum_boxes, minimum_images))
    upper = np.concatenate(([100], np.full(24, np.inf)))
    # 只按标签覆盖确定，不读模型分数；轻微序号项使同分方案选择稳定。
    utility = (np.minimum(box_array, 10) / np.maximum(totals[:, None], 1)).sum(axis=0)
    result = milp(-utility + np.arange(len(candidates)) * 1e-8,
                  integrality=np.ones(len(candidates)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper), options={"time_limit": 30})
    if result.x is None:
        raise ValueError("100张整组方案不满足稀有类约束；保留原划分，不移动文件")
    chosen = result.x > 0.5
    actual = matrix @ chosen.astype(int)
    if np.any(actual < lower) or np.any(actual > upper):
        raise ValueError("划分不满足完整组/类别覆盖约束")
    selected = sorted(row["image"] for members, keep in zip(candidates, chosen) if keep for row in members)
    return {"prior_manifest_sha256": digest(PRIOR), "val_images": selected,
            "class_boxes": dict(zip(CLASSES, actual[1:13].astype(int).tolist())),
            "class_images": dict(zip(CLASSES, actual[13:].astype(int).tolist())),
            "selected_groups": [[row["image"] for row in members] for members, keep in zip(candidates, chosen) if keep],
            "limitations": ["按文件名前缀及既有人工组约束，不证明所有未知场景视觉独立",
                            "100张AP方差更大，不与旧300张AP直接比较"]}


def apply_plan(audit: dict, plan: dict) -> None:
    """先核对原数据，再原地移动；完整保存旧清单，异常时回滚已移动文件。"""
    dataset = ROOT / "datasets"
    selected = set(plan["val_images"])
    moves = []
    updated = deepcopy(audit)
    for row in updated["samples"]:
        before = row["after_split"]
        after = "val" if row["image"] in selected else "train"
        label_name = Path(row["image"]).stem + ".txt"
        official = ROOT / "数据集/训练集/new_labels_2000" / label_name
        if digest(official) != row["after_sha256"]:
            raise ValueError(f"标签不是官方原文：{label_name}")
        for folder, filename, expected in (
                ("labels", label_name, row["after_sha256"]),
                *((folder, row["image"], row["source_hashes"][modality]) for folder, modality in
                  (("images", "visible"), ("infrared", "infrared"), ("depth", "depth")))):
            source = dataset / before / folder / filename
            if digest(source) != expected:
                raise ValueError(f"当前数据与原审计不符：{source}")
            if before != after:
                target = dataset / after / folder / filename
                if target.exists() or not source.resolve().is_relative_to(dataset.resolve()) or not target.resolve().is_relative_to(dataset.resolve()):
                    raise ValueError(f"目标冲突或越界：{target}")
                moves.append((source, target))
        row["before_split"], row["after_split"] = before, after
    for split in ("train", "val"):
        expected = {row["image"] for row in audit["samples"] if row["after_split"] == split}
        for folder in ("images", "infrared", "depth"):
            actual = {p.name for p in (dataset / split / folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
            if actual != expected:
                raise ValueError(f"{split}/{folder}文件清单与审计不一致")
    shutil.copy2(PRIOR, OUTPUT / "before_manifest.json")
    shutil.copy2(dataset / "data.yaml", OUTPUT / "before_data.yaml")
    updated["counts"] = {split: {"images": sum(row["after_split"] == split for row in updated["samples"]),
                                "extensions": dict(Counter(Path(row["image"]).suffix for row in updated["samples"] if row["after_split"] == split))}
                         for split in ("train", "val")}
    updated["split_plan"] = plan
    completed = []
    try:
        for source, target in moves:
            shutil.move(str(source), str(target))
            completed.append((source, target))
        # 标签缓存只归档，不沿用旧划分；不删除用户缓存或任何图像。
        for split in ("train", "val"):
            cache = dataset / split / "labels.cache"
            if cache.exists():
                target = OUTPUT / f"before_{split}_labels.cache"
                shutil.move(str(cache), str(target))
                completed.append((cache, target))
        (OUTPUT / "manifest.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    except BaseException:
        for source, target in reversed(completed):
            shutil.move(str(target), str(source))
        raise
    print(json.dumps({"counts": updated["counts"], "moved_files": len(moves), "label_edits": 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if (OUTPUT / "manifest.json").exists():
        raise SystemExit("划分已落位，不重复执行；先阅读已有审计")
    audit = json.loads(PRIOR.read_text(encoding="utf-8"))
    plan = make_plan(audit)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "split_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"val_count": len(plan["val_images"]), "class_boxes": plan["class_boxes"], "class_images": plan["class_images"]}, ensure_ascii=False))
    if args.apply:
        apply_plan(audit, plan)
