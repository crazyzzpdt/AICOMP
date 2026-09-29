"""框架无关的三模态训练裁剪；不导入YOLO、D-FINE或模型运行依赖。"""

from dataclasses import dataclass
import math
import random

import numpy as np


@dataclass(frozen=True)
class SmallObjectCrop:
    """训练期原图裁剪；默认关闭，旧配方与所有验证输入不变。"""

    probability: float = 0.0  # 主样本尝试概率；不是额外重复采样比例
    min_fraction: float = 0.6  # 同时裁取原宽、高的至少60%，保留局部上下文
    max_fraction: float = 0.8  # 最大保留80%；目标线性放大约1.25–1.67倍
    max_object_size: float = 64.0  # 按完整图长边缩放后的框面积平方根筛选，单位像素
    min_visibility: float = 0.8  # 任意相交框保留不足80%时放弃窗口，不制造无标注碎片
    attempts: int = 6  # 窗口失败后有限重试，全部失败就使用原增强流程

    def __post_init__(self) -> None:
        if not (0 <= self.probability <= 1 and 0 < self.min_fraction <= self.max_fraction < 1
                and math.isfinite(self.max_object_size) and self.max_object_size > 0
                and 0 < self.min_visibility <= 1 and type(self.attempts) is int and 1 <= self.attempts <= 20):
            raise ValueError("小目标裁剪概率、尺寸、可见比例或尝试次数不合法")


def choose_small_object_crop(boxes: np.ndarray, classes: np.ndarray, shape: tuple[int, int],
                             imgsz: int, config: SmallObjectCrop) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """由归一化xywh选择原图窗口，返回同步框；不修改官方标注数组。"""
    if not len(boxes):
        return None
    height, width = shape
    xywh = boxes.astype(np.float32, copy=True) * np.array([width, height, width, height], np.float32)
    xyxy = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    wh = xyxy[:, 2:] - xyxy[:, :2]
    area = wh.prod(axis=1)
    eligible = np.flatnonzero((area > 0) & (np.sqrt(area) * imgsz / max(shape) < config.max_object_size))
    if not len(eligible):
        return None
    categories = classes.reshape(-1)
    for _ in range(config.attempts):
        # 先在本图符合尺寸的类别中等概率选类，避免person框数多就独占所有窗口。
        category = random.choice(np.unique(categories[eligible]).tolist())
        target = random.choice(eligible[categories[eligible] == category].tolist())
        fraction = random.uniform(config.min_fraction, config.max_fraction)
        crop_w, crop_h = max(1, round(width * fraction)), max(1, round(height * fraction))
        x1, y1, x2, y2 = xyxy[target]
        min_x, max_x = max(0, math.ceil(x2 - crop_w)), min(width - crop_w, math.floor(x1))
        min_y, max_y = max(0, math.ceil(y2 - crop_h)), min(height - crop_h, math.floor(y1))
        if min_x > max_x or min_y > max_y:
            continue
        left, top = random.randint(min_x, max_x), random.randint(min_y, max_y)
        clipped = xyxy.copy()
        clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(left, left + crop_w)
        clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(top, top + crop_h)
        visible_area = (clipped[:, 2:] - clipped[:, :2]).prod(axis=1)
        keep = visible_area > 0
        if np.any(visible_area[keep] / np.maximum(area[keep], 1e-9) < config.min_visibility):
            continue
        clipped = clipped[keep] - np.array([left, top, left, top], np.float32)
        normalized = np.concatenate(((clipped[:, :2] + clipped[:, 2:]) / 2,
                                     clipped[:, 2:] - clipped[:, :2]), axis=1)
        normalized /= np.array([crop_w, crop_h, crop_w, crop_h], np.float32)
        return (left, top, crop_w, crop_h), keep, normalized
    return None
