# AIC 城市场景三模态目标检测

## 当前：v13原生1280完整训练（2026-09-20）

v11线上44.4710。v12第10轮仍创新高，mAP50=0.55075、mAP50-95=0.33257、ball AP95=0.45121，却因使用不同验证划分标定的0.34门槛被提前停止。

当前main.py保持v12全部配置，只移除跨划分阶段门槛。模型最多训练200轮，前100轮使用Mosaic，后100轮普通场景收敛，由AP95最佳权重和`patience=100`控制。运行`uv run python main.py`；完成后运行`uv run python predict.py --output predict_v13`。

本轮未执行测试或模型任务，v13尚无成绩，不承诺60分。完整证据见[v12复盘与v13训练方案](docs/v12复盘与v13训练方案.md)。下方v12及更早配置为历史。

2026-09-19 报错修正：main.py已将imgsz同步为IMAGE_HW[0]=1920，保留宽1920、高1080；输出project改为按入口位置解析的绝对runs/detect，避免全局runs_dir重复拼接。其余v9配方和清洗数据不变，额外约200张人工复核仍待完成。只核对代码与差异，未运行测试或训练，不能声称已经通过实际开训验证。详见[v9实施说明](docs/v9融合训练实施.md)。

代码结构（2026-09-19）：根目录只保留`main.py`训练和`predict.py`预测、保存及提交打包两个入口。共享模型/读图和训练组件放在`aic/`；清洗工具放在`tools/prepare_dataset.py`，运行`uv run python -m tools.prepare_dataset`。不再保留中文中转模块，详细迁移与旧权重说明见[代码结构与运行](docs/代码结构与运行.md)。

最新执行（2026-09-18）：用户已批准 **YOLO26l 门控融合 v9**，并要求训练和预测保留1920×1080内容。`main.py`已恢复原生`model.train(...)`分节风格：RGB预训练主干+红外/深度轻量分支，画布补齐为1920×1088，batch=2/有效16、120轮，第81轮完整画面收尾；`predict.py`同步v9 best.pt及相同尺寸。只做静态核对，未测试或启动任何模型任务；显存与分数未验证。详见 [v9融合训练实施](docs/v9融合训练实施.md)。

使用五通道（RGB 3 + 红外 1 + 深度 1）检测官方12类，内部为三个真实参与学习的特征分支和同一检测器。历史YOLO与D-FINE预测后端保留；当前无引用的旧v5专用`训练优化.py`已删除，可从Git提交`08a4180`恢复。原三模态/融合模块已经按职责归并，旧D-FINE只保留预测支持，不再保留训练循环。

清洗数据继续使用`datasets`的1709 train / 291 val，保留2000组及已有57个标签文件修订。本轮不改数据，审计仍为`official_refresh_20260918_214843`。此前空间清理删除两个before中的images及code_cleanup备份；审计和历史标签保留，恢复图像须从官方源按清单重建，不能直接搬回完整旧目录。

v8线上53.5740、v7为52.4010、v6两份候选55.1490/55.1360；v4的55.6000仍为线上基线。v9尚无训练/赛事结果，不承诺超过60分。v8改变过验证划分，旧划分本地AP不能与新划分直接排名。以下v4–v8描述属于历史。

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

本机沿用现有datasets、本地`orgin_models/yolo26l.pt`和已安装依赖。在项目根目录执行：

```powershell
uv run python main.py
```

当前运行名为`AIC_RGBIRDepth_yolo26l_1280_v13_native_full`，重名由YOLO递增目录。best.pt按mAP50-95选取，另存best_map50.pt、last.pt和周期检查点。只恢复同配方中断任务；已完成/早停的last.pt拒绝恢复。新机器需自行准备同版本依赖和官方数据；不要无故升级本机已适配的框架。

训练完成确认实际权重路径后运行`uv run python predict.py --output predict_v13`。历史模型预测须显式传权重和原尺寸。

`准备三模态数据集.py` 默认生成清洗审阅包；v8 已通过 `--audit-v8`、`--review-v8` 与审阅后的 `--apply-v8` 完成落位，日常训练无需再运行。模块中的历史重建函数会恢复官方标注，不应拿来覆盖清洗版。`三模态训练.py` 提供共享加载组件和历史 YOLO 训练器，资源释放由用户管理。

评估脚本、tests、config_checks已不在工作位置，其code_cleanup备份随后经用户授权删除。不会自动执行测试、复评、额外推理或训练。本轮没有安装或更新依赖，D-FINE依赖留作历史模型预测兼容。

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

预测入口采用与 main.py 相同的 `PredictionConfig` 分节配置，实现也已合并在 `predict.py`。以下1536和batch=18是历史快照，当前v9默认1920×1080、batch=4、预取2批。当前用户入口为 FP32、1536、batch=18、8个读取线程、8个保存线程、预取10批；预测v8须如上显式切换权重和1280。可用 `--batch` 调整，不自动试跑探测显存。本轮未测试、未测速，实际吞吐量由正式预测结束时日志记录。

`predict.py` 自动配对官方初赛 1000 组三模态，生成 `predict/images` 带框图片、`predict/labels` 六列标签，以及 `predict/比赛提交内容/submission.zip`。**初赛只提交该 ZIP**，不上传图片或本地运行 JSON。已有输出目录会报错，不覆盖；重复预测使用 `--output predict_v3` 等新目录。

predict.py 的默认权重由用户手动维护；推荐显式传入要提交的单个模型。v5 训练完成后可执行：

```powershell
uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_yolo26l_1280_v5_full/weights/best.pt --output predict_v5
```

可选 `--multi-label` 保留同框多个类别候选，默认关闭；可能改善召回，也可能增加误检。独立候选使用新输出目录，不覆盖基准包。参数与权重选择见 [预测与赛事提交](docs/预测与赛事提交.md)。

五通道权重不能只传 RGB 图片调用通用预测命令；本脚本复用同名三模态预处理。赛事数据不上传公开平台，测试数据仅用于合规推理。
