# AIC 城市场景三模态目标检测

使用 YOLO26l 融合 RGB、红外和深度，检测官方 12 个类别。输入为五通道（RGB 3 + 红外 1 + 深度 1），训练入口为 `main.py`。

## 当前状态（2026-09-17）

初始化说明（2026-09-18）：v5 已从 `orgin_models/yolo26l.pt` 的官方 COCO 预训练重新开始，未续训 v4；使用官方预训练仍可能过拟合。弱类改善不代表整体提高，训练分类损失下降本身不是问题。

v5_full 已完成 234 轮，最高 mAP50=0.59835（41 轮）、mAP50-95=0.40625（34 轮）。ball 从 v4 的 AP50=0 提升至约 0.423，但总体没有明显突破，后期过拟合。不要续训 last.pt；先使用 best.pt 进行用户自行决定的合规提交。详见 [完成复盘](docs/v5训练完成复盘.md)。以下“尚未训练”为历史状态。

最新线上反馈：v4 得分 55.6000，epoch50.pt 为 53.2540，保留 v4 为正式基线。下一轮已写入 main.py，运行名 AIC_RGBIRDepth_yolo26l_1280_v5_full；包含弱类重采样、原尺寸同步目标裁剪、RGB 增强、模态随机缺失和分层学习率，cls_pw=0、mosaic=0.25。第 161 轮起关闭 Mosaic/裁剪/模态缺失，patience=200。目标 60 分以上，不保证实际提分。详见 [v5 配方与 ball 诊断](docs/v5训练方案与球类诊断.md)。尚未执行新配方训练或推理。

- 数据：官方 2000 组，train 1744 / val 256；使用新版标注的可追溯清洗副本，修改 57 个标签文件（64 个边缘框裁剪、1 行去重、1 处大客车类别修订），原始数据保留。
- 已完成 v2：第 79 轮最佳本地 mAP50-95=0.37746，第 179 轮早停；本地验证分数不等同官方测试榜单分数。
- v3 最佳 mAP50-95=0.37290；v4 已完成 200 轮，CSV 最高 mAP50=0.60632（42 轮），最高 mAP50-95=0.40370（100 轮），三次最终 best.pt 的 ball AP50 均为 0。
- 当前代码配置：1280、batch=4、workers=4、BF16、disk 缓存、AdamW、lr0=0.0001（骨干 0.00002）、nbs=16；5000 轮、patience=200，200 轮学习率衰减，第 161 轮开始完整模态收尾。epochs 仍影响 YOLO26 双头损失日程，不是纯训练上限。
- epoch50.pt 的本地 mAP50 更高，但用户报告线上更低，不能只按本地 mAP50 替换正式模型。当前已实施 v5，等待用户开训。

## 使用

先在本地准备官方数据和 `orgin_models/yolo26l.pt`，然后在项目根目录执行：

```powershell
uv sync
uv run python main.py
```

当前运行名为 `AIC_RGBIRDepth_yolo26l_1280_v5_full`，新训练重名自动递增。best.pt 按 mAP50-95 选取，另存 best_map50.pt/best_map5095.pt。仅恢复同配方中断运行时设置 RESUME_PATH；旧 v5_weakclass 配方与已完成且剥离优化器的 last.pt 不作为原状态续训入口。

`准备三模态数据集.py` 默认生成清洗审阅包，确认记录后用 `--apply-review` 落位；日常训练无需运行。模块中的历史重建函数会恢复官方标注，不应拿来覆盖清洗版。`三模态训练.py` 提供加载器和训练器，资源释放由用户管理。

按用户要求，评估脚本、tests、config_checks 与 pytest 缓存已移出原位置，可从 runs/code_cleanup/ 恢复。不会自动执行测试、复评、额外推理或训练。pytest 的开发依赖保留，不影响训练，也不会自行执行；本轮不更新环境或锁文件。

## 本地目录

```text
datasets/
├── train/{images,infrared,depth,labels}
├── val/{images,infrared,depth,labels}
└── data.yaml
数据集/训练集/
├── AIC2026_Train_2000/{visible,infrared,depth,labels}
└── new_labels_2000/
orgin_models/                       # 本地预训练权重
runs/                              # 当次权重、日志与曲线
```

三种模态均位于 datasets 的对应划分内，以官方原图硬链接复用；标签是独立清洗副本。清洗备份与逐行记录在 `runs/dataset_cleaning/20260917_012026/`。硬链接图像不能原地改写。仅 datasets/data.yaml 入库，数据与运行产物不入库。

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

predict.py 的默认权重由用户手动维护；推荐显式传入要提交的单个模型。v5 训练完成后可执行：

```powershell
uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_yolo26l_1280_v5_full/weights/best.pt --output predict_v5
```

可选 `--multi-label` 保留同框多个类别候选，默认关闭；可能改善召回，也可能增加误检。独立候选使用新输出目录，不覆盖基准包。参数与权重选择见 [预测与赛事提交](docs/预测与赛事提交.md)。

五通道权重不能只传 RGB 图片调用通用预测命令；本脚本复用同名三模态预处理。赛事数据不上传公开平台，测试数据仅用于合规推理。
