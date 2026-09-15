"""
数据落位脚本：把 训练集/AIC2026_Train_2000 的 visible 图像与 labels 标注
按约 9:1 划分到 datasets/{train,val}/{images,labels}，
使 datasets/data.yaml 可直接用于 Ultralytics 训练（当前为 RGB-only 基线）。

划分规则（按参赛手册要求：相同/相邻场景的样本不得分跨训练与验证）：
  1. 场景组 = 文件名去掉末段帧号后的前缀组（如 000006_010_xxx、shuming_985_xxx），
     纯数字命名的样本按 ID 间隔<=50 归并为连续段；场景组为原子单位，绝不拆分；
  2. 每个类别"最小的场景组"预留给验证集，保证稀有类在两侧都有覆盖；
  3. 其余组按 (val进度, train进度) 比例贪心分配（大组先分，配平到约 9:1）；
  4. 兜底修复：若某类仍在一侧缺失，整组换入，等量组换回。

落位方式：同盘优先硬链接（零额外空间），失败自动回退为复制。
可重复运行：运行前清空并重建 datasets/{train,val} 下四个子目录。
"""
# 内置库
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "数据集" / "训练集" / "AIC2026_Train_2000"
DST = BASE / "datasets"
# 标注源："labels" = 原始下载版；"new_labels_2000" = 更新版（2026-09-02，382 文件有差异）
LABEL_SUBDIR = "labels"

IMG_EXTS = (".jpg", ".jpeg", ".png")
NUMERIC_RUN_GAP = 50      # 纯数字 ID 间隔超过该值视为不同序列
RUN_MAX_LEN = 90          # 连续段超过该长度时，在段内间隔最大处切开（最大间隔最可能是场景分界）
VAL_FRACTION = 0.10       # 验证集目标比例
VAL_SLACK = 40            # val 允许超出目标比例的样本数（吸收组粒度误差）

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def scene_key(stem: str) -> str:
    """场景组键：去掉末段帧号的前缀；纯数字命名返回 '#num#'。"""
    if re.fullmatch(r"\d+", stem):
        return "#num#"
    return stem.rsplit("_", 1)[0]


def build_groups(stems: list[str]) -> list[list[str]]:
    """把样本归并为场景组：前缀组 + 纯数字连续段。返回按首样本排序的组列表。"""
    prefix_groups: dict[str, list[str]] = {}
    for s in stems:
        if scene_key(s) != "#num#":
            prefix_groups.setdefault(scene_key(s), []).append(s)

    groups: list[list[str]] = [sorted(prefix_groups[k]) for k in sorted(prefix_groups)]

    numeric_stems = sorted((s for s in stems if re.fullmatch(r"\d+", s)), key=int)
    run: list[str] = []
    for s in numeric_stems:
        if run and int(s) - int(run[-1]) > NUMERIC_RUN_GAP:
            groups.extend(split_run(run))
            run = []
        run.append(s)
    if run:
        groups.extend(split_run(run))

    return sorted(groups, key=lambda g: g[0])


def split_run(run: list[str]) -> list[list[str]]:
    """超长连续段在内部间隔最大处切成不超过 RUN_MAX_LEN 的段。"""
    if len(run) <= RUN_MAX_LEN:
        return [run]
    gaps = sorted(((int(run[i + 1]) - int(run[i]), i) for i in range(len(run) - 1)),
                  reverse=True)
    need = -(-len(run) // RUN_MAX_LEN) - 1  # ceil(len/RUN_MAX_LEN) - 1 个切点
    cuts = sorted(i for _, i in gaps[:need])
    chunks, prev = [], 0
    for ci in cuts:
        chunks.append(run[prev:ci + 1])
        prev = ci + 1
    chunks.append(run[prev:])
    return chunks


def boxes_of_split(groups: list[list[str]], box_of: dict) -> Counter:
    c = Counter()
    for g in groups:
        for s in g:
            c.update(box_of.get(s, {}))
    return c


def place(src: Path, dst: Path) -> str:
    """落位单个文件：优先硬链接（同盘零拷贝），失败自动回退为复制。

    Returns:
        实际落位方式："link" 硬链接 / "copy" 复制。
    """
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> None:
    """执行落位全流程：统计 → 场景分组 → 划分 → 硬链接落位 → 复核输出。"""
    img_dir, lbl_dir = SRC / "visible", SRC / LABEL_SUBDIR
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    stems = [p.stem for p in imgs]
    n = len(stems)

    # 每样本的类别框数
    box_of: dict[str, Counter] = {}
    for s in stems:
        c = Counter()
        txt = lbl_dir / f"{s}.txt"
        if txt.exists():
            for line in txt.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    c[int(line.split()[0])] += 1
        box_of[s] = c

    groups = build_groups(stems)
    assert sum(len(g) for g in groups) == n, "分组后样本数不一致"
    total_boxes = Counter()
    for g in groups:
        for s in g:
            total_boxes.update(box_of[s])
    n_val = round(n * VAL_FRACTION)

    # 1) 预留：每个类别最小的场景组划给 val（组必须足够小，防止巨型组把 val 撑爆）
    val_idx: set[int] = set()
    for c in sorted(total_boxes):
        best = min((i for i, g in enumerate(groups) if any(box_of[s].get(c) for s in g)),
                   key=lambda i: (len(groups[i]), groups[i][0]), default=None)
        if best is not None and len(groups[best]) <= n_val // 2:
            val_idx.add(best)
    v_sz = sum(len(groups[i]) for i in val_idx)
    print(f"[预留] {len(val_idx)} 个场景组划入 val，共 {v_sz} 张")

    # 2) 其余组按两侧进度比例贪心（大组先分，配平到约 9:1）
    t_target, v_target = n - n_val, n_val
    t_sz = n - v_sz
    train_idx: set[int] = set(range(len(groups))) - val_idx
    for gi in sorted((i for i in range(len(groups)) if i not in val_idx),
                     key=lambda i: (-len(groups[i]), groups[i][0])):
        g = groups[gi]
        train_idx.discard(gi)
        v_progress, t_progress = v_sz / max(v_target, 1), t_sz / max(t_target, 1)
        if v_progress <= t_progress and v_sz + len(g) <= v_target + VAL_SLACK:
            val_idx.add(gi)
            v_sz += len(g)
        else:
            train_idx.add(gi)
            t_sz += len(g)

    val_groups = [groups[i] for i in sorted(val_idx)]
    train_groups = [groups[i] for i in sorted(train_idx)]

    # 3) 兜底修复：某类在一侧缺失时整组换入，等量组换回
    for _ in range(30):
        tb = boxes_of_split(train_groups, box_of)
        vb = boxes_of_split(val_groups, box_of)
        val_missing = [c for c in total_boxes if vb[c] == 0]
        train_missing = [c for c in total_boxes if tb[c] == 0]
        if not val_missing and not train_missing:
            break
        if val_missing:
            c0, src_side, dst_side = val_missing[0], train_groups, val_groups
        else:
            c0, src_side, dst_side = train_missing[0], val_groups, train_groups
        cand = [g for g in src_side if any(box_of[s].get(c0) for s in g)]
        if not cand:
            break  # 该类全部集中在另一侧的巨型组中，无法整组换入，接受现状
        g_move = min(cand, key=lambda g: (len(g), g[0]))
        scarce = set(train_missing + val_missing + [c0])
        back = [g for g in dst_side
                if not any(any(box_of[s].get(c) for s in g) for c in scarce)]
        if back and sum(len(g) for g in dst_side) - len(g_move) > n_val // 2:
            g_back = min(back, key=lambda g: (abs(len(g) - len(g_move)), g[0]))
            src_side.remove(g_move)
            dst_side.remove(g_back)
            dst_side.append(g_move)
            src_side.append(g_back)
        else:
            src_side.remove(g_move)
            dst_side.append(g_move)

    # 4) 体积微调：val 占比超出 6%~14% 时搬运组，但不得搬空任何类别在任一侧的覆盖
    for _ in range(100):
        v_sz = sum(len(g) for g in val_groups)
        if 0.06 <= v_sz / n <= 0.14:
            break
        if v_sz / n > 0.14:
            src_side, dst_side = val_groups, train_groups
        else:
            src_side, dst_side = train_groups, val_groups
        side_boxes = boxes_of_split(src_side, box_of)

        def safe(g: list[str]) -> bool:
            gboxes = Counter()
            for s in g:
                gboxes.update(box_of[s])
            return all(side_boxes[c] - gboxes[c] > 0 for c in gboxes)

        cand = [g for g in src_side if safe(g)]
        if not cand:
            break  # 再搬会丢类别覆盖，接受当前体积
        g_move = max(cand, key=lambda g: (len(g), g[0]))
        src_side.remove(g_move)
        dst_side.append(g_move)

    val_set = {s for g in val_groups for s in g}
    assert not (val_set & {s for g in train_groups for s in g}), "train/val 场景组重叠！"

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            target = DST / split / sub
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)

    counts = {"train": 0, "val": 0}
    linked = copied = 0
    missing_lbl = []
    for img in imgs:
        split = "val" if img.stem in val_set else "train"
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            missing_lbl.append(img.stem)
            continue
        if place(img, DST / split / "images" / img.name) == "link":
            linked += 1
        else:
            copied += 1
        place(lbl, DST / split / "labels" / lbl.name)
        counts[split] += 1

    def class_hist(split: str) -> dict:
        hist = Counter()
        for txt in (DST / split / "labels").glob("*.txt"):
            for line in txt.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    hist[int(line.split()[0])] += 1
        return dict(sorted(hist.items()))

    v_final = sum(len(g) for g in val_groups)
    print(f"场景组数: {len(groups)} (val {len(val_groups)} / train {len(train_groups)})")
    print(f"总样本 {n} | train {counts['train']} | val {counts['val']} "
          f"(val 占比 {counts['val'] / n:.1%})")
    print(f"落位方式: 硬链接 {linked} / 复制 {copied}")
    print(f"train 类别分布: {class_hist('train')}")
    print(f"val   类别分布: {class_hist('val')}")
    tb = boxes_of_split(train_groups, box_of)
    vb = boxes_of_split(val_groups, box_of)
    one_sided = [c for c in sorted(total_boxes) if tb[c] == 0 or vb[c] == 0]
    if one_sided:
        print(f"⚠️ 类别 {one_sided} 仅存在于一侧（该类样本过度集中，无法安全整组换入）")
    if missing_lbl:
        print(f"⚠️ {len(missing_lbl)} 张图片缺标注未入库: {missing_lbl[:8]} ...")
    else:
        print("图片-标注配对完整，无缺漏。")


if __name__ == "__main__":
    main()
