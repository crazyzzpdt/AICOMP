# 运行快照工具区

`run_snapshots/` 保存从历史 `runs/` 产物中移出的 Python 源码快照，按原运行相对路径归档。
它们只用于追溯历史训练，不是当前训练入口，也不应直接执行。当前入口固定为根目录
`train1.py`（YOLO）、`train2.py`（D-FINE）、`predict1.py`（YOLO）和`predict2.py`（D-FINE）。

快照目录被 Git 忽略，避免把训练产物和历史重复代码提交到仓库；正式复赛材料由预测入口按所选权重单独复制需要的源码。
