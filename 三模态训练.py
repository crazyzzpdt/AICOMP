"""为 YOLO26l 提供 RGB、红外、深度五通道融合训练能力。

由 main.py 导入 MultimodalDetectionTrainer；不可直接作为独立训练入口运行。
"""

# 内置库
import math
from copy import copy
from functools import partial
from pathlib import Path

# 三方库
import cv2
import numpy as np
import torch
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import get_hash
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


# 沿用既有深度编码上限，本轮不同时改变深度尺度。
DEPTH_MAX_MM: int = 20_000
# 旧运行 73 轮达到最佳、之后长期停滞；将衰减周期与总轮数上限分开。
LR_DECAY_EPOCHS: int = 200
# 每个进程至多保留 8 张缩放图，降低五通道 Mosaic 缓冲占用。
IMAGE_BUFFER_LIMIT: int = 8


# 一、三模态图像读取与融合
def read_image(path: Path, flags: int) -> np.ndarray:
    """读取单张图像，并兼容 Windows 中文路径。

    Args:
        path: 待读取的图像文件。
        flags: OpenCV 图像读取模式。

    Returns:
        成功解码的图像数组。

    Raises:
        FileNotFoundError: 图像不存在或无法解码。
    """
    # Windows 下 OpenCV 的 imread 可能无法处理中文路径，改由 NumPy 读取文件字节后再解码。
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise FileNotFoundError(f"无法读取三模态图像：{path}") from error
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise FileNotFoundError(f"无法读取三模态图像：{path}")
    return image


def convert_to_single_channel(image: np.ndarray) -> np.ndarray:
    """将红外或深度图转换为单通道表示。

    Args:
        image: 红外或低分辨率深度图。

    Returns:
        输入图像的单通道表示。
    """
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_depth_channel(depth: np.ndarray) -> np.ndarray:
    """将深度图转换为模型使用的 8 位单通道。

    PNG 深度图保留原始 uint16 毫米值读取，并在 0 至 20000 毫米内线性归一化；
    官方 JPG 深度图没有毫米值，因此仅转为灰度，不假设其距离尺度。

    Args:
        depth: 读取后的官方深度图。

    Returns:
        范围为 0 至 255 的 uint8 深度通道。
    """
    if depth.dtype == np.uint16:
        clipped = np.clip(depth, 0, DEPTH_MAX_MM)
        return np.rint(clipped * 255.0 / DEPTH_MAX_MM).astype(np.uint8)
    return convert_to_single_channel(depth)


def adapt_rgb_stem_weights(source: torch.Tensor, target_channels: int) -> torch.Tensor:
    """将 RGB 预训练首层权重扩展为多通道输入权重。

    原有 RGB 权重完整保留，新增模态权重从零开始学习，避免初始时红外、深度
    直接改变 COCO 预训练响应；零初始化的新增输入权重仍可正常获得梯度。

    Args:
        source: 形状为 ``[输出通道, 3, 卷积高, 卷积宽]`` 的 RGB 预训练权重。
        target_channels: 目标输入通道数，必须不少于三。

    Returns:
        适配到目标输入通道数的首层权重。

    Raises:
        ValueError: 输入不是三通道 RGB 权重，或目标通道数少于三。
    """
    if source.ndim != 4 or source.shape[1] != 3:
        raise ValueError("只能迁移输入通道为三通道的 RGB 预训练权重")
    if target_channels < 3:
        raise ValueError("目标输入通道数不能少于三")

    if target_channels == 3:
        return source.clone()
    modality_weights = source.new_zeros((source.shape[0], target_channels - 3, *source.shape[2:]))
    return torch.cat((source, modality_weights), dim=1)


def fuse_modalities(visible_path: Path, infrared_path: Path, depth_path: Path) -> np.ndarray:
    """将同一词干的 RGB、红外和深度图合成为五通道输入。

    Args:
        visible_path: 官方可见光 RGB 图路径。
        infrared_path: 官方红外图路径。
        depth_path: 官方深度图路径。

    Returns:
        RGB、红外、归一化深度顺序的 uint8 五通道图像。

    Raises:
        ValueError: 三张图尺寸不一致。
    """
    if len({visible_path.name, infrared_path.name, depth_path.name}) != 1:
        raise ValueError("三模态必须使用同名样本，不能混合不同帧")
    visible = cv2.cvtColor(read_image(visible_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    infrared = read_image(infrared_path, cv2.IMREAD_UNCHANGED)
    depth = read_image(depth_path, cv2.IMREAD_UNCHANGED)
    infrared_channel = convert_to_single_channel(infrared)
    depth_channel = normalize_depth_channel(depth)

    # 官方同名三模态已对齐；尺寸异常应报告，不能通过拉伸掩盖错配。
    if not visible.shape[:2] == infrared_channel.shape[:2] == depth_channel.shape[:2]:
        raise ValueError(f"三模态尺寸不一致：{visible_path.name}")
    return np.dstack((visible, infrared_channel, depth_channel))


# 二、Ultralytics 五通道数据集
class MultimodalYOLODataset(YOLODataset):
    """读取同名 RGB、红外、深度图并交给 Ultralytics 检测增强流程。"""

    def __init__(self, *args: object, data: dict[str, object], **kwargs: object) -> None:
        """初始化模态根目录后构建 YOLO 标签数据集。

        Args:
            args: 传递给 Ultralytics YOLODataset 的位置参数。
            data: 含 ``infrared``、``depth`` 与 ``channels: 5`` 的数据集配置。
            kwargs: 传递给 Ultralytics YOLODataset 的关键字参数。

        Raises:
            KeyError: 数据集配置缺少三模态目录。
            ValueError: 数据集通道数不是五。
        """
        if data.get("channels") != 5:
            raise ValueError("三模态早期融合训练的数据集 channels 必须为 5")
        data_root = Path(str(data.get("path", Path.cwd()))).resolve()
        infrared_dir = Path(str(data["infrared"]))
        depth_dir = Path(str(data["depth"]))
        self.infrared_dir = (data_root / infrared_dir).resolve() if not infrared_dir.is_absolute() else infrared_dir
        self.depth_dir = (data_root / depth_dir).resolve() if not depth_dir.is_absolute() else depth_dir
        if not self.infrared_dir.is_dir() or not self.depth_dir.is_dir():
            raise FileNotFoundError("红外或深度图目录不存在，请检查 datasets/data.yaml")
        super().__init__(*args, data=data, **kwargs)
        self.max_buffer_length = min(self.max_buffer_length, IMAGE_BUFFER_LIMIT)

    def get_label_files(self) -> list[str]:
        """建立可见光图、红外图、深度图与新版标签的同名映射。"""
        label_files = super().get_label_files()
        self.infrared_files = [self.infrared_dir / Path(image_path).name for image_path in self.im_files]
        self.depth_files = [self.depth_dir / Path(image_path).name for image_path in self.im_files]
        for modality_path, description in (
            *[(path, "红外图") for path in self.infrared_files],
            *[(path, "深度图") for path in self.depth_files],
        ):
            if not modality_path.is_file():
                raise FileNotFoundError(f"{description}缺失，无法构成三模态样本：{modality_path}")
        return label_files

    def get_cache_hash(self) -> str:
        """让标签缓存随任一模态文件变化而失效。"""
        files = self.label_files + self.im_files + [str(path) for path in self.infrared_files + self.depth_files]
        files.append(f"fusion-v2-classes-{','.join(map(str, self.data['names'].values()))}")
        return get_hash(files)

    def load_fused_image(self, index: int) -> np.ndarray:
        """读取索引对应的原始尺寸五通道图像。"""
        visible_path = Path(self.im_files[index])
        return fuse_modalities(
            visible_path,
            self.infrared_dir / visible_path.name,
            self.depth_dir / visible_path.name,
        )

    def cache_images_to_disk(self, index: int) -> None:
        """将五通道融合图写入 NPY 缓存，避免后续反复解码三个源文件。"""
        cache_path = self.npy_files[index]
        if cache_path.exists() and not self.cache_is_current(index):
            cache_path.unlink()
        if not cache_path.exists():
            try:
                np.save(cache_path.as_posix(), self.load_fused_image(index), allow_pickle=False)
            except Exception as error:
                cache_path.unlink(missing_ok=True)
                LOGGER.warning(f"{self.prefix}无法缓存三模态图像 {cache_path}：{error}")

    def cache_is_current(self, index: int) -> bool:
        """源图或融合实现更新后重新生成缓存，避免复用旧 BGR 输入。"""
        visible_path = Path(self.im_files[index])
        sources = (visible_path, self.infrared_dir / visible_path.name, self.depth_dir / visible_path.name, Path(__file__))
        cache_path = self.npy_files[index]
        return cache_path.is_file() and cache_path.stat().st_mtime_ns >= max(path.stat().st_mtime_ns for path in sources)

    def load_image(
        self, index: int, rect_mode: bool = True, resize_short: bool = False
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """加载五通道图像，并保持 Ultralytics 的原始缩放和缓存语义。"""
        image, cache_path = self.ims[index], self.npy_files[index]
        if image is None:
            if self.cache_is_current(index):
                try:
                    image = np.load(cache_path, allow_pickle=False)
                    if image.ndim != 3 or image.shape[2] != 5:
                        raise ValueError(f"缓存通道数为 {image.shape[-1] if image.ndim >= 3 else 1}，期望为 5")
                except Exception as error:
                    LOGGER.warning(f"{self.prefix}移除失效三模态缓存 {cache_path}：{error}")
                    cache_path.unlink(missing_ok=True)
                    image = self.load_fused_image(index)
            else:
                image = self.load_fused_image(index)

            height_original, width_original = image.shape[:2]
            if rect_mode:
                if resize_short:
                    ratio = self.imgsz / min(height_original, width_original)
                    if ratio != 1:
                        width, height = (
                            (math.ceil(width_original * ratio), self.imgsz)
                            if height_original < width_original
                            else (self.imgsz, math.ceil(height_original * ratio))
                        )
                        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
                else:
                    ratio = self.imgsz / max(height_original, width_original)
                    if ratio != 1:
                        width = min(math.ceil(width_original * ratio), self.imgsz)
                        height = min(math.ceil(height_original * ratio), self.imgsz)
                        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            elif not (height_original == width_original == self.imgsz):
                image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            if self.augment and self.cache != "ram":
                self.ims[index] = image
                self.im_hw0[index] = (height_original, width_original)
                self.im_hw[index] = image.shape[:2]
                self.buffer.append(index)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    old_index = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[old_index], self.im_hw0[old_index], self.im_hw[old_index] = None, None, None

            return image, (height_original, width_original), image.shape[:2]
        return image, self.im_hw0[index], self.im_hw[index]


# 三、三模态训练器
def learning_rate_factor(epoch: int, decay_epochs: int, final_ratio: float, cosine: bool) -> float:
    """按独立收敛周期衰减学习率，到达下限后保持不反弹。"""
    progress: float = min(max(epoch / max(decay_epochs, 1), 0.0), 1.0)
    remaining: float = (1.0 + math.cos(math.pi * progress)) / 2.0 if cosine else 1.0 - progress
    return final_ratio + (1.0 - final_ratio) * remaining


class MultimodalDetectionTrainer(DetectionTrainer):
    """构建五通道数据集，并将 RGB 预训练首层迁移到三模态模型。"""

    # 框架生命周期方法名必须与父类一致。
    def _setup_scheduler(self) -> None:
        """保留训练轮数上限，将实际学习率衰减限制在前 200 轮。"""
        self.lf = partial(
            learning_rate_factor,
            decay_epochs=min(self.epochs, LR_DECAY_EPOCHS),
            final_ratio=self.args.lrf,
            cosine=self.args.cos_lr,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def get_dataloader(
        self, dataset_path: str, batch_size: int = 16, rank: int = -1, mode: str = "train"
    ) -> InfiniteDataLoader:
        """为本机单卡构建低预取加载器，验证进程数不再翻倍。

        Note:
            五通道增强曾出现 CPU 内存分配失败，因此每个进程仅预取一批，关闭锁页。
            当前实现限定单卡训练；Windows 多进程入口仍由 main.py 保护。
        """
        if rank != -1:
            raise ValueError("当前三模态加载器面向本机单卡，请使用 device=0")
        if mode not in {"train", "val"}:
            raise ValueError(f"不支持的数据加载模式：{mode}")
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        batch_size = min(batch_size, len(dataset))
        shuffle: bool = mode == "train" and not dataset.rect
        workers: int = min(self.args.workers, math.ceil(len(dataset) / batch_size))
        generator = torch.Generator().manual_seed(self.args.seed)
        LOGGER.info(f"{mode}: 加载进程 {workers}，每进程预取 1 批，锁页内存关闭")
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            prefetch_factor=1 if workers else None,
            pin_memory=False,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=bool(self.args.compile and mode == "train"),
        )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> MultimodalYOLODataset:
        """构建保持三模态空间同步增强的检测数据集。"""
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return MultimodalYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=copy(self.args),
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_model(self, cfg: str | None = None, weights: object = None, verbose: bool = True) -> DetectionModel:
        """创建五通道检测模型并迁移 RGB 预训练参数。"""
        model = self.set_model_names_for_load(
            DetectionModel(cfg, nc=self.data["nc"], ch=5, verbose=verbose and RANK == -1)
        )
        if weights:
            source_model = (weights.get("ema") or weights["model"]) if isinstance(weights, dict) else weights
            # 仅复制模型外壳与名称表，复用权重；不改原检查点名称或比赛类别编号。
            load_source = copy(source_model)
            source_names: dict[int, str] = dict(source_model.names)
            if self.args.cls_remap and source_model.model[0].conv.in_channels == 3:
                for class_id, class_name in source_names.items():
                    if class_name.strip().lower() == "sports ball" and "ball" not in source_names.values():
                        source_names[class_id] = "ball"
                        LOGGER.info("预训练类别迁移：sports ball → ball（比赛类别编号保持不变）")
            load_source.names = source_names
            model.load(load_source, verbose=verbose)
            source_weights = source_model.float().state_dict()["model.0.conv.weight"]
            target_weights = model.state_dict()["model.0.conv.weight"]
            # 五通道检查点直接恢复；只有首次从 RGB 权重训练时扩展首层。
            if source_weights.shape[1] == 3:
                target_weights.copy_(adapt_rgb_stem_weights(source_weights, target_weights.shape[1]))
            elif source_weights.shape != target_weights.shape:
                raise ValueError("检查点首层与当前五通道模型不兼容")
        return model
