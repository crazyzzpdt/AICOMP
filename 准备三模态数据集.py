"""在 datasets 内非破坏性构建使用新版标注的三模态训练集。

仅在需要重新生成 datasets/multimodal_new_labels 时运行本文件；正式训练由 main.py 启动。
"""

# 内置库
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
    """复制新版标签；已有标签内容不一致时停止，避免覆盖人工结果。"""
    if destination.exists():
        if destination.read_bytes() == source.read_bytes():
            return
        raise FileExistsError(f"目标标签已存在且内容不同：{destination}")
    shutil.copy2(source, destination)


def _data_yaml(output_dir: Path, raw_dataset: Path) -> str:
    """生成供自定义五通道数据集读取的 YAML 配置。"""
    infrared_dir = (raw_dataset / "infrared").resolve().as_posix()
    depth_dir = (raw_dataset / "depth").resolve().as_posix()
    output_path = output_dir.resolve().as_posix()
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    return (
        f"path: {output_path}\n"
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
    """从既有划分构建新版标注的独立三模态训练集。

    图像只建立硬链接（或不支持时复制），标签只从 ``new_labels`` 复制；因此
    ``old_dataset`` 中的旧标注与原始数据不会被修改。

    Args:
        old_dataset: 现有 RGB 训练集根目录，用于复用已核验的 train/val 划分。
        raw_dataset: 包含 infrared 与 depth 文件夹的官方下载训练集目录。
        new_labels: 官方更新后的 YOLO 标签目录。
        output_dir: 新建三模态训练集的目标根目录。

    Returns:
        新三模态训练集根目录。

    Raises:
        FileNotFoundError: 图像、对应模态或新版标签缺失。
        FileExistsError: 已有目标文件不能安全复用。
    """
    for split in ("train", "val"):
        source_images = old_dataset / split / "images"
        if not source_images.is_dir():
            raise FileNotFoundError(f"找不到现有 {split} 图像目录：{source_images}")
        destination_images = output_dir / split / "images"
        destination_labels = output_dir / split / "labels"
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)

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
    if yaml_path.exists() and yaml_path.read_text(encoding="utf-8") != yaml_content:
        raise FileExistsError(f"目标数据配置已存在且内容不同：{yaml_path}")
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return output_dir


def main() -> None:
    """在 datasets 下生成 multimodal_new_labels。"""
    project_root = Path(__file__).resolve().parent
    output_dir = prepare_multimodal_dataset(
        old_dataset=project_root / "datasets",
        raw_dataset=project_root / "数据集" / "训练集" / "AIC2026_Train_2000",
        new_labels=project_root / "数据集" / "训练集" / "new_labels_2000",
        output_dir=project_root / "datasets" / "multimodal_new_labels",
    )
    print(f"三模态训练集已就绪：{output_dir}")


if __name__ == "__main__":
    main()
