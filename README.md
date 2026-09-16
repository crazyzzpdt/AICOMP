# AIC 城市场景三模态目标检测

使用 YOLO26l 融合 RGB、红外和深度，检测官方 12 个类别。输入为五通道（RGB 3 + 红外 1 + 深度 1），训练入口为 `main.py`。

## 当前状态（2026-09-17）

- 数据：官方 2000 组，train 1744 / val 256；使用 `new_labels_2000`。图片与标签来源一致，未发现批量错配；仍有少量官方标注瑕疵及验证分布不足。
- 已完成 v2：第 79 轮最佳本地 mAP50-95=0.37746，第 179 轮早停；本地验证分数不等同官方测试榜单分数。
- v3 已完成：第 150 轮 CSV 最佳 mAP50-95=0.37290，第 250 轮早停；未超过旧运行第 79 轮的 0.37746。两次 ball AP 都为 0，暂保留旧 best.pt 作为提交候选，后续统一评估确认。
- 当前代码配置：1280、batch=4、workers=4、BF16、disk 缓存、AdamW、lr0=0.0003、nbs=16；5000 轮、patience=100，200 轮学习率衰减，第 101 轮关闭 Mosaic。epochs 仍影响 YOLO26 双头损失日程，不是纯训练上限。
- 下一步先统一评估、定向复核弱类标注，再做 lr0=0.0001 等单变量实验；候选参数尚未实施。本次文档更新不修改数据或启动训练。

## 使用

先在本地准备官方数据和 `orgin_models/yolo26l.pt`，然后在项目根目录执行：

```powershell
uv sync
uv run pytest tests/
uv run python main.py
```

新训练输出到 `runs/detect/AIC_RGBIRDepth_yolo26l_1280_v3/`，重名自动递增，选用当次 `weights/best.pt`。仅恢复同配方中断运行时，将 `main.py` 的 `RESUME_PATH` 改为实际 `weights/last.pt`；新配方保持 `None`。

`准备三模态数据集.py` 仅用于按现有划分重新复制官方新版标签、刷新配置与缓存，不需每次开训运行。`三模态训练.py` 提供加载器和训练器，不是第二个训练入口。资源释放由用户管理，不随训练自动执行。

## 本地目录

```text
datasets/
├── train/{images,labels}
├── val/{images,labels}
└── data.yaml
数据集/训练集/
├── AIC2026_Train_2000/{visible,infrared,depth,labels}
└── new_labels_2000/
orgin_models/                       # 本地预训练权重
runs/                              # 当次权重、日志与曲线
```

RGB 图为原始 visible 的硬链接，新版 TXT 为独立副本；红外和深度通过 `datasets/data.yaml` 引用官方训练目录。数据、缓存、权重及训练产物不入库，唯一数据配置例外为 `datasets/data.yaml`。

## 文档导航

- [数据来源、问题与划分](docs/数据集成分与划分记录.md)：新版类别统计、ball 核验、边缘框、验证集不足。
- [训练配方与实验记录](docs/训练配置与数据集复核.md)：两次完成训练对比、球类退化诊断、双头损失日程及下一轮实验；历史配置单独保留。
- [参数参考](docs/Ultralytics训练参数参考.md)、[目录规范](docs/YOLO数据集目录结构与配置规范.md)：区分框架默认值和项目设置。
- [训练通用经验](docs/YOLO训练通用经验.md)、[数据工作笔记](docs/丹方.md)：历史示例不直接套用本赛题。
- [本机环境](docs/电脑训练环境.md)、[历史资源释放记录](docs/训练前释放资源.md)。
- [赛题要求摘录](docs/比赛任务要求.md)、[官方资料](docs/比赛资料/官方赛题规则.md)：官方内容与项目核验结论分开维护。

## 预测与初赛提交

```powershell
uv run python predict.py
```

`predict.py` 自动配对官方初赛 1000 组三模态，生成 `predict/images` 带框图片、`predict/labels` 六列标签，以及 `predict/比赛提交内容/submission.zip`。**初赛只提交该 ZIP**，不上传图片或本地运行 JSON。已有输出目录会报错，不覆盖；重复预测使用 `--output predict_v3` 等新目录。

默认采用现有验证指标略高的旧运行 `AIC_RGBIRDepth_yolo26l_1280/weights/best.pt`；可用 `--weights` 显式选择其他已训练五通道权重。参数、默认权重选择依据和防覆盖说明见 [预测与赛事提交](docs/预测与赛事提交.md)。

五通道权重不能只传 RGB 图片调用通用预测命令；本脚本复用同名三模态预处理。赛事数据不上传公开平台，测试数据仅用于合规推理。
