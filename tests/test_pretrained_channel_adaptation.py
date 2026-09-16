"""验证 RGB 预训练首层迁移到五通道输入的规则。"""

# 内置库
from copy import deepcopy
from types import SimpleNamespace

# 三方库
import pytest
import torch
from ultralytics.nn.tasks import DetectionModel

# 自己的模块
from 三模态训练 import MultimodalDetectionTrainer, adapt_rgb_stem_weights


@pytest.mark.parametrize("source_channels", [3, 5])
def test_ball_head_transfer_preserves_both_branches_and_source_names(source_channels: int) -> None:
    """验证 RGB 别名迁移和五通道恢复均保留双分支参数与比赛类别编号。"""
    cfg = {"nc": 80, "end2end": True, "reg_max": 1,
           "backbone": [[-1, 1, "Conv", [96, 3, 2]]], "head": [[[0], 1, "Detect", ["nc"]]]}
    names: dict[int, str] = {index: f"target_{index}" for index in range(12)}
    names[7] = "ball"
    source = DetectionModel(deepcopy(cfg), ch=source_channels, nc=80 if source_channels == 3 else 12, verbose=False)
    source.names = {index: f"source_{index}" for index in range(80)} if source_channels == 3 else names.copy()
    ball_id: int = 32 if source_channels == 3 else 7
    source.names[ball_id] = "sports ball" if source_channels == 3 else "ball"
    original_names = source.names.copy()
    with torch.no_grad():
        for branch_name in ("cv3", "one2one_cv3"):
            output = getattr(source.model[-1], branch_name)[0][-1]
            output.weight[ball_id].fill_(0.125)
            output.bias[ball_id].fill_(-2.0)
    trainer = object.__new__(MultimodalDetectionTrainer)
    trainer.args = SimpleNamespace(cls_remap=True)
    trainer.data = {"nc": 12, "names": names}
    target = trainer.get_model(deepcopy(cfg), weights=source, verbose=False)
    assert source.names == original_names
    assert target.names == names
    for branch_name in ("cv3", "one2one_cv3"):
        expected = getattr(source.model[-1], branch_name)[0][-1]
        actual = getattr(target.model[-1], branch_name)[0][-1]
        torch.testing.assert_close(actual.weight[7], expected.weight[ball_id])
        torch.testing.assert_close(actual.bias[7], expected.bias[ball_id])
    if source_channels == 5:
        for key, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value)


def test_adapt_rgb_stem_weights_preserves_pretrained_response() -> None:
    """新增模态初始不干扰 RGB 响应，但首层模态权重仍能获得梯度。"""
    source = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    adapted = adapt_rgb_stem_weights(source, target_channels=5)

    expected = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[0.0]], [[0.0]]]])
    torch.testing.assert_close(adapted, expected)
    image = torch.arange(1.0, 6.0).reshape(1, 5, 1, 1)
    adapted.requires_grad_(True)
    output = torch.nn.functional.conv2d(image, adapted)
    torch.testing.assert_close(output, torch.nn.functional.conv2d(image[:, :3], source))
    output.sum().backward()
    assert torch.all(adapted.grad[:, 3:] != 0)


def test_adapt_rgb_stem_weights_rejects_non_rgb_pretrained_layer() -> None:
    """输入权重不是 RGB 三通道时不能静默套用迁移规则。"""
    source = torch.zeros((1, 4, 1, 1))

    try:
        adapt_rgb_stem_weights(source, target_channels=5)
    except ValueError as error:
        assert "三通道" in str(error)
    else:
        raise AssertionError("非 RGB 权重必须拒绝迁移")
