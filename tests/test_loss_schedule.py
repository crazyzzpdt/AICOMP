"""验证独立双检测头损失日程及恢复状态，不依赖正式训练。"""

# 内置库
from copy import copy

# 三方库
import pytest
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.nn.tasks import DetectionModel

# 自己的模块
import 三模态训练 as multimodal


def test_loss_horizon_does_not_change_epoch_cap_and_restores_progress() -> None:
    """调整损失进度不能改变早停上限，恢复第 150 轮须保持对应双头比例。"""
    cfg = {"nc": 12, "end2end": True, "reg_max": 1,
           "backbone": [[-1, 1, "Conv", [96, 3, 2]]], "head": [[[0], 1, "Detect", ["nc"]]]}
    model = DetectionModel(cfg, ch=5, nc=12, verbose=False)
    model.args = copy(DEFAULT_CFG)
    model.args.epochs = 5000
    multimodal.configure_loss_schedule(model, 300, 150)
    assert model.args.epochs == 5000
    assert model.criterion.one2one.hyp.epochs == 300
    assert model.criterion.one2many.hyp.epochs == 5000
    assert model.criterion.o2m == pytest.approx(0.4488294314)
    assert model.criterion.o2o == pytest.approx(0.5511705686)
    model.criterion.update()
    assert model.criterion.updates == 151
    assert model.criterion.o2m < 0.4488294314
    multimodal.configure_loss_schedule(model, 300, 500)
    assert model.criterion.o2m == pytest.approx(0.1)
    assert model.criterion.o2o == pytest.approx(0.9)
