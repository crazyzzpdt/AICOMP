"""覆盖已确认的训练/预测边界问题，不进行模型训练或全量数据评估。"""

# 内置库
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
import json
from types import SimpleNamespace

# 三方库
import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics

# 自己的模块
from src.yolo.aic import data
from src.yolo.aic import training
from src.yolo.aic.model import EARLY_FUSION_VERSION, EVALUATION_PROTOCOL, EarlyFusionDetectionModel, letterbox_native_fused, restore_boxes
from src.yolo.aic.training import FusionDetectionTrainer, FusionRecipe, MultimodalYOLODataset, PolishEarlyStopping


def test_native_padding_and_round_trip() -> None:
    """防止预测再用辅助通道零补边，或将真实内容坐标还原错位。"""
    image = np.full((1080, 1920, 5), [10, 20, 30, 40, 50], dtype=np.uint8)
    canvas, geometry = letterbox_native_fused(image, 1280)
    assert canvas.shape == (1280, 1280, 5)
    np.testing.assert_array_equal(canvas[0, 0], [114] * 5)
    np.testing.assert_array_equal(canvas[280, 0], [10, 20, 30, 40, 50])
    boxes = restore_boxes(torch.tensor([[0., 280., 1280., 1000.]]), geometry, (1080, 1920))
    torch.testing.assert_close(boxes, torch.tensor([[0., 0., 1920., 1080.]]))


def test_native_validator_clips_the_same_content_as_prediction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """防止验证保留补边内的框，或把框架的高宽缩放误读为宽高。"""
    trainer = FusionDetectionTrainer.__new__(FusionDetectionTrainer)
    trainer.test_loader, trainer.save_dir, trainer.args = None, tmp_path, get_cfg()
    trainer.callbacks, trainer.native_square = {}, True
    validator = trainer.get_validator()
    validator.domains = []
    validator.args.plots = True  # 此用例只检查裁框；混淆矩阵在独立快照用例覆盖。
    # 只隔离框架的指标累积；被测裁框及实际预测数据结构保持真实。
    monkeypatch.setattr(DetectionValidator, "update_metrics", lambda *args: None)
    rows = [{"bboxes": torch.tensor([[-10., 0., 50., 90.], [0., 0., 5., 5.]]),
             "conf": torch.tensor([0.9, 0.8]), "cls": torch.tensor([0., 1.]), "extra": torch.empty(2, 0)}]
    batch = {"ori_shape": [(40, 80)], "ratio_pad": [((1.5, 0.5), (10, 20))], "im_file": ["sample.png"]}
    validator.update_metrics(rows, batch)
    torch.testing.assert_close(rows[0]["bboxes"], torch.tensor([[10., 20., 50., 80.]]))
    torch.testing.assert_close(rows[0]["conf"], torch.tensor([0.9]))


def test_cache_false_does_not_load_existing_npy(tmp_path: Path) -> None:
    """cache=False必须读取当前模态，不能因为磁盘上恰好有旧缓存而复用。"""
    dataset = MultimodalYOLODataset.__new__(MultimodalYOLODataset)
    cached = tmp_path / "old.npy"
    np.save(cached, np.full((2, 2, 5), 7, dtype=np.uint8))
    dataset.ims, dataset.npy_files = [None], [cached]
    dataset.cache, dataset.augment, dataset.imgsz = False, False, 2
    dataset.cache_is_current = lambda index: True
    dataset.load_fused_image = lambda index: np.full((2, 2, 5), 9, dtype=np.uint8)
    image, _, _ = dataset.load_image(0)
    assert np.all(image == 9)
    assert cached.is_file(), "关闭缓存不应删除用户已有缓存"


def test_label_cache_rejects_equal_length_label_edit(tmp_path: Path) -> None:
    """标签改类但字节数不变时，框架的文件尺寸哈希不能继续复用旧类别。"""
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    dataset = MultimodalYOLODataset.__new__(MultimodalYOLODataset)
    dataset.label_files, dataset.im_files = [str(label)], []
    dataset.infrared_files, dataset.depth_files = [], []
    dataset.data = {"names": {0: "person", 1: "boat"}}
    before = dataset.get_cache_hash()
    label.write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    assert before != dataset.get_cache_hash()


def test_source_fingerprint_rejects_same_size_modality_change(tmp_path: Path) -> None:
    """同名、同尺寸字节数的辅助模态被替换，也必须被来源审计拒绝。"""
    hashes = {}
    for folder, modality in (("images", "visible"), ("infrared", "infrared"), ("depth", "depth")):
        directory = tmp_path / "train" / folder
        directory.mkdir(parents=True)
        path = directory / "sample.png"
        path.write_bytes(b"abcd")
        hashes[modality] = data.file_hash(path)
    rows = [{"image": "sample.png", "after_split": "train", "source_hashes": hashes}]
    data.verify_source_images(tmp_path, rows)
    (tmp_path / "train/infrared/sample.png").write_bytes(b"abce")
    with pytest.raises(ValueError, match="infrared"):
        data.verify_source_images(tmp_path, rows)


def test_native_polish_explicitly_updates_real_augmentation() -> None:
    """显式收尾值必须传入重建的增强链，而不是仅写进recipe。"""
    dataset = MultimodalYOLODataset.__new__(MultimodalYOLODataset)
    dataset.polish_scale, dataset.polish_translate = 0.1, 0.025
    dataset.build_transforms = lambda hyp: (hyp.mosaic, hyp.scale, hyp.translate)
    hyp = SimpleNamespace(mosaic=0.5, scale=0.3, translate=0.1, copy_paste=0., mixup=0., cutmix=0.)
    dataset.close_mosaic(hyp)
    assert dataset.transforms == (0., 0.1, 0.025)


def test_native_polish_none_preserves_previous_recipe() -> None:
    """未显式启用收尾扰动时，不能暗中修改v13/v14的尺度和位移。"""
    dataset = MultimodalYOLODataset.__new__(MultimodalYOLODataset)
    dataset.polish_scale, dataset.polish_translate = None, None
    dataset.build_transforms = lambda hyp: (hyp.mosaic, hyp.scale, hyp.translate)
    hyp = SimpleNamespace(mosaic=0.5, scale=0.3, translate=0.1, copy_paste=0., mixup=0., cutmix=0.)
    dataset.close_mosaic(hyp)
    assert dataset.transforms == (0., 0.3, 0.1)


def test_patience_zero_disables_plateau_stop() -> None:
    """沿用YOLO的patience=0禁用早停语义，不应在第20轮意外结束。"""
    stopper = PolishEarlyStopping(1, 0, 0.0)
    assert not any(stopper(epoch, 0.3) for epoch in range(1, 61))
    assert not stopper.possible_stop


def test_patience_counts_real_plateau() -> None:
    """正常耐心计数仍能结束平台期，不因禁用分支修复而失效。"""
    stopper = PolishEarlyStopping(1, 3, 0.0)
    for epoch in range(1, 20):
        assert not stopper(epoch, 0.3)
    assert stopper(20, 0.3)


def test_best_snapshot_identifies_epoch_without_another_forward(tmp_path: Path) -> None:
    """最佳曲线必须由同轮已计算指标生成，并明确标注轮次，不能冒用末轮图。"""
    metrics = DetMetrics(names={0: "person"})
    metrics.update_stats({"tp": np.ones((1, 10), dtype=bool), "conf": np.array([0.9]),
                          "pred_cls": np.array([0.]), "target_cls": np.array([0.]),
                          "target_img": np.array([0.]), "im_name": "sample.png"})
    metrics.process()
    matrix = ConfusionMatrix(names={0: "person"})
    training.save_validation_artifacts(metrics, matrix, tmp_path, epoch=7)
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["epoch"] == 7
    assert report["metrics"]["metrics/mAP50-95(B)"] > 0.9
    assert (tmp_path / "BoxPR_curve.png").is_file()
    assert (tmp_path / "confusion_matrix.png").is_file()


def test_polish_can_stop_before_twenty_epochs() -> None:
    """短收尾的最低轮数不能仍被历史20轮常量锁死。"""
    stopper = PolishEarlyStopping(1, 3, 0.0, min_epochs=5)
    for epoch in range(1, 5):
        assert not stopper(epoch, 0.39)
    assert stopper(5, 0.39)
    with pytest.raises(ValueError):
        PolishEarlyStopping(1, 3, 0.0, min_epochs=0)


def test_polish_schedule_does_not_relax_main_training() -> None:
    """全程不拼图只对显式polish开放，原主训练的日程保护保持有效。"""
    trainer = FusionDetectionTrainer.__new__(FusionDetectionTrainer)
    trainer.recipe = FusionRecipe(training_stage="polish", architecture="early_v10", geometry="native_square",
                                  image_height=1280, image_width=1280, min_stop_epochs=5,
                                  initial_weights_sha256="a" * 64)
    trainer.args = get_cfg(overrides={"epochs": 20, "mosaic": 0., "close_mosaic": 0, "cls_remap": False})
    trainer._validate_training_stage()
    trainer.args.mosaic = 0.5
    with pytest.raises(ValueError):
        trainer._validate_training_stage()
    trainer.args.mosaic = 0.
    trainer.recipe = FusionRecipe()
    with pytest.raises(ValueError):
        trainer._validate_training_stage()


@pytest.fixture
def polish_pair(tmp_path: Path) -> tuple[FusionDetectionTrainer, EarlyFusionDetectionModel, EarlyFusionDetectionModel]:
    """用真实卷积及BN的小状态检验严格迁移；完整检测器由限定GPU闭环覆盖。"""
    source = EarlyFusionDetectionModel.__new__(EarlyFusionDetectionModel)
    torch.nn.Module.__init__(source)
    source.model = torch.nn.Sequential(torch.nn.Sequential(OrderedDict(
        conv=torch.nn.Conv2d(5, 4, 3, bias=False), bn=torch.nn.BatchNorm2d(4))))
    source.names = dict(enumerate(data.CLASS_NAMES))
    source.early_fusion_version, source.content_hw = EARLY_FUSION_VERSION, (1280, 1280)
    source.fusion_training = {"signature": {"version": EARLY_FUSION_VERSION,
        "evaluation_protocol": EVALUATION_PROTOCOL, "recipe": {"geometry": "native_square"},
        "audit": {"manifest_sha256": "audit", "yaml_sha256": "yaml"}}, "completed": True}
    with torch.no_grad():
        source.model[0].conv.weight.fill_(0.125)
        source.model[0].bn.running_mean.fill_(2.)
    path = tmp_path / "parent.pt"
    source.pt_path = str(path)
    torch.save({"epoch": 14, "ema": source}, path)
    trainer = FusionDetectionTrainer.__new__(FusionDetectionTrainer)
    trainer.recipe = FusionRecipe(training_stage="polish", architecture="early_v10", geometry="native_square",
                                  image_height=1280, image_width=1280, min_stop_epochs=5,
                                  initial_weights_sha256=data.file_hash(path))
    trainer.recipe_version = EARLY_FUSION_VERSION
    trainer.args = SimpleNamespace(model=str(path))
    trainer.audit = {"manifest_sha256": "audit", "yaml_sha256": "yaml"}
    target = deepcopy(source)
    with torch.no_grad():
        target.model[0].conv.weight.zero_()
        target.model[0].bn.running_mean.zero_()
    return trainer, source, target


def test_polish_preserves_auxiliary_weights_and_buffers(polish_pair: tuple) -> None:
    """已学习的IR/Depth与BN统计必须继承，不能仅迁移RGB或再次置零。"""
    trainer, source, target = polish_pair
    record = trainer._load_polish_weights(target, source)
    for name, expected in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], expected), name
    assert torch.count_nonzero(target.model[0].conv.weight[:, 3:]) > 0
    assert record["epoch"] == 15
    assert record["weights_sha256"] == trainer.recipe.initial_weights_sha256


def test_polish_entry_assigns_names_without_remapping(polish_pair: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """关闭类别重映射后，新建检测器仍须标记真实12类并走新阶段加载分支。"""
    trainer, source, target = polish_pair
    trainer.early_fusion, trainer.resume, trainer.resume_metadata = True, False, None
    trainer.rgb_diagnostic, trainer.quality_fusion = False, False
    trainer.data = {"nc": 12, "names": dict(enumerate(data.CLASS_NAMES))}
    trainer.args.cls_remap = False
    target.names = {i: str(i) for i in range(12)}
    # 只缩小昂贵的网络构建；入口分支、类别赋值、来源检查及参数复制均执行真实代码。
    monkeypatch.setattr(EarlyFusionDetectionModel, "__new__", staticmethod(lambda cls, *args, **kwargs: target))
    monkeypatch.setattr(EarlyFusionDetectionModel, "__init__", lambda *args, **kwargs: None)
    result = trainer.get_model(cfg={"scale": "l"}, weights=source, verbose=False)
    assert result.names == source.names
    assert torch.equal(result.model[0].conv.weight, source.model[0].conv.weight)
    assert trainer.resume_metadata is None
    assert trainer.parent_initialization["epoch"] == 15


@pytest.mark.parametrize("bad", ["names", "audit", "geometry", "hash"])
def test_polish_rejects_unmatched_parent(polish_pair: tuple, bad: str) -> None:
    """不能静默继承错误类别、划分、几何或被替换的同名权重。"""
    trainer, source, target = polish_pair
    if bad == "names":
        source.names[7] = "wrong"
    elif bad == "audit":
        source.fusion_training["signature"]["audit"]["manifest_sha256"] = "other"
    elif bad == "geometry":
        source.fusion_training["signature"]["recipe"]["geometry"] = "fixed_rect"
    else:
        Path(trainer.args.model).write_bytes(b"replaced")
    with pytest.raises(ValueError):
        trainer._load_polish_weights(target, source)
