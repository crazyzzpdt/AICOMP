"""验证 RGB 预训练首层迁移到五通道输入的规则。"""

# 三方库
import torch

# 自己的模块
from 三模态训练 import adapt_rgb_stem_weights


def test_adapt_rgb_stem_weights_preserves_response_scale() -> None:
    """RGB 权重须缩放后保留，红外和深度以 RGB 均值初始化。"""
    source = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    adapted = adapt_rgb_stem_weights(source, target_channels=5)

    expected = torch.tensor([[[[0.6]], [[1.2]], [[1.8]], [[1.2]], [[1.2]]]])
    torch.testing.assert_close(adapted, expected)


def test_adapt_rgb_stem_weights_rejects_non_rgb_pretrained_layer() -> None:
    """输入权重不是 RGB 三通道时不能静默套用迁移规则。"""
    source = torch.zeros((1, 4, 1, 1))

    try:
        adapt_rgb_stem_weights(source, target_channels=5)
    except ValueError as error:
        assert "三通道" in str(error)
    else:
        raise AssertionError("非 RGB 权重必须拒绝迁移")
