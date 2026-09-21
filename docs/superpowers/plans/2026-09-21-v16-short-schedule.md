# v16 短日程训练实施计划

完成追加：用户已跑完60轮（1.974小时），最佳第36轮AP95=0.3676631044/AP50=0.5931457553，低于v14；三个阶段门槛均通过，没有提前停止。未做独立复评、赛事分数未知。当前入口已转v17受限采样，不重复启动本配方；以下“未运行”保留为当时交付记录。

> 执行方式：使用writing-plans记录、executing-plans在当前工作区执行。用户已确认v16；本轮仅交付配置及文档，不创建工作树、不调用子代理、不测试、不训练、不推理、不提交推送。

**Goal：** 在固定清洗版1709/291与相同评估协议下，检验60轮学习率衰减及第21轮关闭Mosaic能否超过v14，不承诺赛事60分。

**Architecture：** 官方YOLO26l RGB预训练重新迁移，五通道首层早期融合，IR/Depth新增卷积零初始化后参与训练，ball语义迁移保留。不是v14/v15续训，不改变融合结构或重洗数据。

**Tech Stack：** 本机Ultralytics8.4.152、PyTorch2.14.0+cu132；不升级、不联网下载。

## 全局约束与参数

仅修改main.py配置及相关说明；aic/、predict.py实现及用户默认值保持不变。用户已自行使用v4预测复赛测试集并提交，复赛分数未提供，不能沿用初赛55.6000填入。

| 参数 | v16 |
|---|---|
| model / resume / stage | orgin_models/yolo26l.pt / False / main |
| run | AIC_RGBIRDepth_yolo26l_1280_v16_short_schedule |
| 数据、验证 | 当前datasets/data.yaml、1709/291，审计不变 |
| imgsz / batch / nbs / workers | 1280 / 4 / 16 / 4 |
| epochs / warmup / cos_lr / lrf | 60 / 5 / True / 0.01 |
| AdamW主lr / 骨干第1–10层lr | 1e-4 / 2e-5 |
| mosaic / close_mosaic | 0.5 / 40，即第21轮关闭 |
| scale / translate / fliplr / HSV | 0.3 / 0.1 / 0.5 / 全0，关闭拼图后仍保持 |
| polish_scale / polish_translate | None / None，不额外改变几何增强强度 |
| patience / min_delta / 最低耐心停止轮数 | 40 / 0 / 20，沿用v14，不用v15的8/0.0005/5 |
| 阶段筛选 | 沿用v14在本周期内的10/20/50轮历史最佳AP95≥0.32/0.34/0.36 |
| cls_remap / cls_pw / 双头权重 | True / 0 / 0.8与0.2 |
| 验证 | val_batch2、FP32、conf0.001/iou0.7/max_det100，单标签、内容裁框 |
| 保存 | 每5轮留检查点，每轮last，AP95/AP50各自最佳，已有结果不覆盖 |

选择60轮是时间预算下的假设，不是已证明最优。与v14同时改变总日程及Mosaic关闭时间，只能验证组合；val_batch2与既有独立复评一致，训练时峰值要与重载基线区别记录。阶段门槛用于节约明显弱配方预算，不保证达到0.60，也可能放弃晚到的改善；阶段筛选可以早于耐心停止。

## 任务1：入口配置

- [x] 通读main.py，核对既有训练器的main阶段限制、60轮衰减和第21轮关闭条件；确认本地官方权重存在，仅查文件不加载网络。
- [x] MODEL_PATH改官方权重；删除v15父权重摘要常量，recipe中initial_weights_sha256=None、training_stage=main、cls_remap=True；保持RESUME_PATH=None。
- [x] 按上表设置main.py，保留全部YOLO分节参数、原审计路径和历史检查点保护，不新增训练器或脚本。

## 任务2：交付与边界

- [x] 同步README、知识索引、AGENTS/CLAUDE当前状态与历史分数说明，标明v16尚未运行、v4复赛分数未知。
- [x] 只检查实际文本差异、文件路径与配置引用；不运行pytest、AST/compile/import、模型前向或任何训练/预测/复评。不把v15的18项通过记成v16测试结果。
- [x] 提供用户自行开训及显式复赛预测命令；预测仍使用相同输入/后处理，不自动推广尚未训练的v16。

## 用户运行与验收

```powershell
uv run python main.py
```

按v14约121秒/轮估算，60轮约2小时；硬件负载、早停等会影响实际耗时。第21轮应在正式日志看到关闭Mosaic；optimization_recipe.json应记录polish_start_epoch=21、主/骨干初始lr=1e-4/2e-5。不是保证能跑通的测试结果。

v16完成后查看best_validation/metrics.json与results.csv，候选再按同291张、FP32/batch2、相同协议进行用户明确授权的独立复评，基线v14为0.3917876103；至少+0.001作为工程接受阈值，不是统计显著性。若未提高，保留v14，不自动重跑或改验证集。

训练完成且选择v16后，显式指定复赛源，避免predict.py历史默认源仍为初赛：

```powershell
uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_yolo26l_1280_v16_short_schedule/weights/best.pt --source "数据集/复赛测试集" --imgsz 1280 --height 1280 --output predict_v16_round2
```

若训练目录自动增加后缀，权重路径填写实际目录。复赛源只用于最终预测，不能用于训练、伪标签或调参；当前打包器输出预测TXT ZIP，不表示已经满足复赛要求的全部代码、报告等交付项。

## 执行记录

main.py配置已修改，官方预训练文件存在，既有训练器无需改动即可表达本日程；仅依据源码读取判断，尚未验证运行。完整分节参数保留，未修改aic/、predict.py、数据集、旧权重、旧运行目录。本轮禁止模型执行；交付前只核对文本和实际差异，不创建新的测试脚本或后台训练任务。v15历史自动任务已完成，不复用其调度器启动v16。
