"""将 datasets 现有划分切换为新版标注并写入三模态配置。

仅在需要重新落位 datasets/train、datasets/val 时运行；正式训练由 main.py 启动。
"""

# 内置库
import os
from pathlib import Path
import shutil


IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
CLASS_NAMES: tuple[str, ...] = (
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage can",
    "uav",
    "tricycle",
)


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


def main() -> None:
    """更新 datasets 中的现有 train、val 和 data.yaml。"""
    project_root = Path(__file__).resolve().parent
    output_dir = prepare_multimodal_dataset(
        old_dataset=project_root / "datasets",
        raw_dataset=project_root / "数据集" / "训练集" / "AIC2026_Train_2000",
        new_labels=project_root / "数据集" / "训练集" / "new_labels_2000",
        output_dir=project_root / "datasets",
    )
    print(f"三模态训练集已就绪：{output_dir}")


if __name__ == "__main__":
    main()
