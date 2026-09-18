# AIC 城市场景三模态目标检测

使用五通道（RGB 3 + 红外 1 + 深度 1）检测官方 12 类。当前训练入口 `main.py` 使用 D-FINE-L；历史 YOLO26l 权重、训练组件和预测兼容保留。

最新候选为 **v8**：1280、最多60轮、学习率5e-5/骨干5e-6、10轮耐心早停；关闭v7的定向重复、完整目标裁剪和低清模拟。清洗数据已落位 `datasets`，1709 train / 291 val，保留全部2000组及已有修订，本次新增标签修订0；旧完整数据和缓存保存在 `runs/dataset_cleaning/v8_20260918_162439/before/`。详见 [v8训练方案与数据清洗](docs/v8训练方案与数据清洗.md)。未运行测试、模型评估、推理或训练，用户自行执行 `uv run python main.py`。

v7线上52.4010，v6两份最佳权重55.1490/55.1360，v4的55.6000仍为线上基线。v8验证集发生变化，本地AP不能直接与旧划分排名；不承诺超过60分。predict.py默认权重由用户维护。

## 历史 v6 启动记录（2026-09-18）

v5 best.pt 已确认线上 **51.0830**，SHA256 与提交记录一致；v4 **55.6000** 仍是正式基线。main.py 已切换为 D-FINE-L Objects365 E25 五通道微调：1280、60 轮、3 轮预热、AdamW 主学习率 1e-4/骨干 1e-5、batch=2、有效批次16。关闭 v5 的重采样、定向裁剪与模态缺失。官方源码与权重、必要依赖已在本机准备，尚未启动新训练或运行模型测试，不能声称已提分或验证显存可用。

配置依据、类别初始化和离线复现见 [D-FINE三模态实施方案](docs/D-FINE三模态实施方案.md)。下面 v5 的配置描述是历史，不是当前入口。

## 历史 YOLO v4/v5 记录

初始化说明（2026-09-18）：v5 已从 `orgin_models/yolo26l.pt` 的官方 COCO 预训练重新开始，未续训 v4；使用官方预训练仍可能过拟合。弱类改善不代表整体提高，训练分类损失下降本身不是问题。

v5_full 已完成 234 轮，最高 mAP50=0.59835（41 轮）、mAP50-95=0.40625（34 轮）。ball 从 v4 的 AP50=0 提升至约 0.423，但总体没有明显突破，后期过拟合。不要续训 last.pt；先使用 best.pt 进行用户自行决定的合规提交。详见 [完成复盘](docs/v5训练完成复盘.md)。以下“尚未训练”为历史状态。

最新线上反馈：v4 得分 55.6000，epoch50.pt 为 53.2540，保留 v4 为正式基线。下一轮已写入 main.py，运行名 AIC_RGBIRDepth_yolo26l_1280_v5_full；包含弱类重采样、原尺寸同步目标裁剪、RGB 增强、模态随机缺失和分层学习率，cls_pw=0、mosaic=0.25。第 161 轮起关闭 Mosaic/裁剪/模态缺失，patience=200。目标 60 分以上，不保证实际提分。详见 [v5 配方与 ball 诊断](docs/v5训练方案与球类诊断.md)。尚未执行新配方训练或推理。

- 数据：官方 2000 组，train 1744 / val 256；使用新版标注的可追溯清洗副本，修改 57 个标签文件（64 个边缘框裁剪、1 行去重、1 处大客车类别修订），原始数据保留。
- 已完成 v2：第 79 轮最佳本地 mAP50-95=0.37746，第 179 轮早停；本地验证分数不等同官方测试榜单分数。
- v3 最佳 mAP50-95=0.37290；v4 已完成 200 轮，CSV 最高 mAP50=0.60632（42 轮），最高 mAP50-95=0.40370（100 轮），三次最终 best.pt 的 ball AP50 均为 0。
- 当前代码配置：1280、batch=4、workers=4、BF16、disk 缓存、AdamW、lr0=0.0001（骨干 0.00002）、nbs=16；5000 轮、patience=200，200 轮学习率衰减，第 161 轮开始完整模态收尾。epochs 仍影响 YOLO26 双头损失日程，不是纯训练上限。
- epoch50.pt 的本地 mAP50 更高，但用户报告线上更低，不能只按本地 mAP50 替换正式模型。当前已实施 v5，等待用户开训。

## 使用

本机已经准备现有 datasets、官方 `vendor/D-FINE` 固定源码和 `orgin_models/dfine_l_obj365_e25.pth`。在项目根目录执行：

```powershell
uv sync
uv run python main.py
```

当前运行名为 `AIC_RGBIRDepth_dfine_l_1280_v8`，重名时新建时间戳目录。best.pth 按 mAP50-95 选取，另存 best_map50.pth、last.pth 和每5轮检查点。仅恢复同配方中断运行时设置 RESUME_PATH，恢复也写新目录；已完成或早停的任务拒绝恢复。入口会核对数据审计，正式训练另存分域指标和审计快照。新机器请先按实施方案准备公开依赖及数据，训练本身不会联网。

`准备三模态数据集.py` 默认生成清洗审阅包；v8 已通过 `--audit-v8`、`--review-v8` 与审阅后的 `--apply-v8` 完成落位，日常训练无需再运行。模块中的历史重建函数会恢复官方标注，不应拿来覆盖清洗版。`三模态训练.py` 提供共享加载组件和历史 YOLO 训练器，资源释放由用户管理。

按用户要求，评估脚本、tests、config_checks 与 pytest 缓存已移出原位置，可从 runs/code_cleanup/ 恢复。不会自动执行测试、复评、额外推理或训练。pytest 的开发依赖保留，不会自行执行。D-FINE 新增依赖已写入 pyproject.toml/uv.lock，没有替换现有 torch/torchvision/Ultralytics。

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
vendor/D-FINE/                      # 固定版本官方源码（不入库）
runs/                              # 当次权重、日志与曲线
```

三种模态均位于 datasets 的对应划分内，以官方原图硬链接复用；标签是独立清洗副本。清洗备份与逐行记录在 `runs/dataset_cleaning/20260917_012026/`。硬链接图像不能原地改写。仅 datasets/data.yaml 入库，数据与运行产物不入库。

## 文档导航

- [项目知识索引](docs/项目知识索引.md)：统一当前状态、线上反馈、推理恢复、清洗事实和历史文档范围。

- [v8当前方案与数据清洗](docs/v8训练方案与数据清洗.md)：新配置、数据备份、划分变化、启动与预测命令。
- [D-FINE历史实施](docs/D-FINE三模态实施方案.md)：固定源码、预训练来源、类别迁移。

- [数据来源、问题与划分](docs/数据集成分与划分记录.md)：新版类别统计、ball 核验、边缘框、验证集不足。
- [训练配方与实验记录](docs/训练配置与数据集复核.md)：两次完成训练对比、球类退化诊断、双头损失日程及下一轮实验；历史配置单独保留。
- [参数参考](docs/Ultralytics训练参数参考.md)、[目录规范](docs/YOLO数据集目录结构与配置规范.md)：区分框架默认值和项目设置。
- [训练通用经验](docs/YOLO训练通用经验.md)、[数据工作笔记](docs/丹方.md)：历史示例不直接套用本赛题。
- [本机环境](docs/电脑训练环境.md)、[历史资源释放记录](docs/训练前释放资源.md)。
- [赛题要求摘录](docs/比赛任务要求.md)、[官方资料](docs/比赛资料/官方赛题规则.md)：官方内容与项目核验结论分开维护。

## 预测与初赛提交

已恢复历史YOLO逐张原生单标签推理路径，使用：

```powershell
uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_yolo26l_1280_v4_clean_lr1e4/weights/best.pt --imgsz 1280 --yolo-profile v4 --output predict_v4_restored
```

YOLO默认选v4路径；现有批量自定义后处理用 `--yolo-profile current`。D-FINE使用独立原生后端，不能套用YOLO后处理，当前用户v7默认权重保持不变。历史来源和边界见 [预测文档](docs/预测与赛事提交.md)。

```powershell
uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_dfine_l_1280_v8/weights/best.pth --imgsz 1280 --batch 18 --output predict_v8
```

上述命令在 v8 训练产生权重后使用，若运行目录带时间戳须换成实际目录。predict.py 保留用户当前选择的 v7 best.pth 路径，不自动选择最新运行；历史五通道 YOLO .pt 仍可通过 --weights 使用，输出结构不变。

预测入口采用与 main.py 相同的 `PredictionConfig` 分节配置，实现在 `三模态预测.py`。当前用户入口为 FP32、1536、batch=18、8个读取线程、8个保存线程、预取10批；预测v8须如上显式切换权重和1280。可用 `--batch` 调整，不自动试跑探测显存。本轮未测试、未测速，实际吞吐量由正式预测结束时日志记录。

`predict.py` 自动配对官方初赛 1000 组三模态，生成 `predict/images` 带框图片、`predict/labels` 六列标签，以及 `predict/比赛提交内容/submission.zip`。**初赛只提交该 ZIP**，不上传图片或本地运行 JSON。已有输出目录会报错，不覆盖；重复预测使用 `--output predict_v3` 等新目录。

predict.py 的默认权重由用户手动维护；推荐显式传入要提交的单个模型。v5 训练完成后可执行：

```powershell
uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_yolo26l_1280_v5_full/weights/best.pt --output predict_v5
```

可选 `--multi-label` 保留同框多个类别候选，默认关闭；可能改善召回，也可能增加误检。独立候选使用新输出目录，不覆盖基准包。参数与权重选择见 [预测与赛事提交](docs/预测与赛事提交.md)。

五通道权重不能只传 RGB 图片调用通用预测命令；本脚本复用同名三模态预处理。赛事数据不上传公开平台，测试数据仅用于合规推理。
