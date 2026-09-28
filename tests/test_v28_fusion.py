"""v28有界契约检查：合成小图，不启动正式训练或读取赛事测试集。"""

import os
from types import SimpleNamespace

os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

import pytest
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from src.modalities import FLOAT_PREPROCESS_VERSION
from src.yolo.aic.data import CLASS_NAMES
from src.yolo.aic.model import EVALUATION_PROTOCOL
from src.yolo.aic.reliability import LocalQualityFusion, RELIABILITY_FUSION_VERSION
from src.yolo.aic.training import FusionDetectionTrainer, FusionRecipe, copy_ema_to_cpu
from src.yolo.prediction import FusionPredictor


@pytest.fixture(scope="module")
def prepared():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    source = YOLO("orgin_models/yolo26x.pt").model
    trainer = FusionDetectionTrainer.__new__(FusionDetectionTrainer)
    trainer.recipe = FusionRecipe(architecture="reliability_v28", geometry="native_square",
                                  continuous_depth=True, freeze_bn_stats=True, image_height=128,
                                  image_width=128, early_backbone_lr=1e-5, auxiliary_lr=1e-4)
    trainer.rgb_diagnostic = trainer.quality_fusion = trainer.early_fusion = False
    trainer.reliability_fusion = trainer.continuous_depth = True
    trainer.resume = False
    trainer.resume_metadata = None
    trainer.recipe_version = RELIABILITY_FUSION_VERSION
    trainer.args = get_cfg(overrides={"cls_remap": True, "cls_pw": 0.0, "nms": True})
    trainer.data = {"nc": 12, "names": dict(enumerate(CLASS_NAMES)), "channels": 5}
    model = trainer.get_model(weights=source, verbose=False)
    model.args = trainer.args
    return trainer, model


def test_zero_residual_keeps_rgb_and_optimizer_covers_all_parameters(prepared):
    trainer, model = prepared
    image = torch.rand(1, 5, 128, 128)
    model.eval()
    with torch.inference_mode():
        reference = DetectionModel._predict_once(model, image[:, :3])[0]
        actual = model(image)[0]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    optimizer = trainer.build_optimizer(model, lr=5e-5, momentum=0.9, decay=5e-4)
    lookup = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    assert set(lookup) == {id(p) for p in model.parameters()}
    for name, parameter in model.named_parameters():
        expected = 1e-4 if not name.startswith("model.") else 1e-5 if int(name.split(".")[1]) <= 10 else 5e-5
        assert lookup[id(parameter)]["lr"] == expected


def test_depth_holes_block_depth_and_local_alignment_stays_finite():
    torch.manual_seed(1)
    block = LocalQualityFusion(32, 16, align=True).train()
    torch.nn.init.normal_(block.project.weight, std=0.01)
    torch.nn.init.normal_(block.gate[-1].weight, std=0.01)
    rgb, ir, depth = torch.rand(1, 32, 8, 8), torch.rand(1, 16, 8, 8), torch.rand(1, 16, 8, 8)
    cues = torch.rand(1, 4, 8, 8)
    cues[:, 3] = 0
    # 无支持区连门控也不能读取不可用的深度特征。
    first = block(rgb, ir, depth, cues)
    second = block(rgb, ir, torch.zeros_like(depth), cues)
    torch.testing.assert_close(first, second)
    assert block.last_gates[2] == 0 and torch.isfinite(first).all()


def test_detection_loss_gradients_and_checkpoint_recompute(prepared):
    trainer, model = prepared
    model.train()
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()
    # 第一步零投影仅更新投影本身；打开小投影后检查辅助分支真实参与检测损失。
    with torch.no_grad():
        for block in model.fusion_blocks:
            block.project.weight.normal_(std=0.001)
        model.rgb_detail[-1].weight.normal_(std=0.001)
    batch = {"img": torch.rand(1, 5, 128, 128), "batch_idx": torch.tensor([0, 0]),
             "cls": torch.tensor([[0.], [7.]]), "bboxes": torch.tensor([[.5, .5, .2, .3], [.2, .3, .1, .1]])}
    loss, _ = model(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    for module in (model.ir_encoder, model.depth_encoder, model.rgb_detail, *model.fusion_blocks):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    # 同批检测损失与梯度一致，确认节省显存的重算没有破坏训练。
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()
                 if p.grad is not None and not name.startswith("model.")}
    model.zero_grad(set_to_none=True)
    model.checkpoint_auxiliary = False
    repeated_loss, _ = model(batch)
    repeated_loss.sum().backward()
    torch.testing.assert_close(loss, repeated_loss)
    for name, p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name])
    model.checkpoint_auxiliary = True
    model.zero_grad(set_to_none=True)


def test_fp32_save_reload_and_prediction_contract(prepared, tmp_path):
    _, model = prepared
    model.eval()
    model.fusion_training = {"signature": {
        "version": RELIABILITY_FUSION_VERSION, "evaluation_protocol": EVALUATION_PROTOCOL,
        "recipe": {"geometry": "native_square", "continuous_depth": True, "architecture": "reliability_v28"},
        "float_preprocessing": {"version": FLOAT_PREPROCESS_VERSION},
        "reliability_fusion": {"version": RELIABILITY_FUSION_VERSION},
    }}
    saved = tmp_path / "bounded_checkpoint.pt"
    torch.save(copy_ema_to_cpu(model), saved)
    loaded = torch.load(saved, map_location="cpu", weights_only=False)
    assert all(p.dtype == torch.float32 for p in loaded.parameters())
    predictor = FusionPredictor(loaded, "cpu", (128, 128))
    image = torch.rand(1, 5, 128, 128) * 255
    targets = [{"orig_size": torch.tensor([128, 128]), "geometry": torch.tensor([1., 1., 0., 0.])}]
    config = SimpleNamespace(channels_last=False, classes=None, agnostic_nms=False)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(image / 255)[0], model(image / 255)[0], rtol=0, atol=0)
    rows = predictor.predict_batch(image, targets, .001, 100, .7, config)
    assert len(rows) == 1 and torch.isfinite(rows[0]["boxes"]).all()
    assert len(rows[0]["boxes"]) <= 100 and predictor.input_geometry == "float_native"
    loaded.fusion_training["signature"]["reliability_fusion"]["version"] = "wrong"
    with pytest.raises(ValueError, match="v28"):
        FusionPredictor(loaded, "cpu", (128, 128))
