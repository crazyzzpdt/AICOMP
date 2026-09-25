# 源码目录

| 位置 | 职责 |
|---|---|
| `modalities.py` | 框架无关的连续浮点三模态读取、几何与传感器增强 |
| `prediction_io.py` | 模态配对、有限预取、图片/TXT保存、ZIP和材料；不选择或加载模型 |
| `yolo/prediction.py` | YOLO专用权重校验、预处理和预测；入口为根目录predict1.py |
| `yolo/aic/` | YOLO模型、数据集、训练器和旧权重反序列化兼容 |
| `dfine/prediction.py` | D-FINE专用预测及历史输入兼容；入口为根目录predict2.py |
| `dfine/training.py` | 项目D-FINE五通道训练适配 |
| `dfine/runtime.py` | 项目D-FINE构建、源码验证和坐标解码 |
| `D-FINE/src/`、`D-FINE/configs/` | 上游D-FINE实现与配置；目录层级保留，避免破坏源码清单及相对配置 |

`D-FINE`（带连字符）是上游源码目录；`dfine`（小写）是项目可导入包，二者有明确分工。上游工具在根目录 `tools/dfine/`，历史代码在 `tools/archive/run_snapshots/`。

保留有效的 `yolo/aic` 模块路径，避免已有YOLO权重保存的类路径失效。已删除无源码的 `common/` 和 `yolo/aic/aic/`，不创建无职责占位包。

本轮仅目录重组和引用更新，没有验证模型运行效果。源码摘要变化后，历史断点仍应使用其对应源码快照恢复，不绕过校验；旧权重推理路径保留。
