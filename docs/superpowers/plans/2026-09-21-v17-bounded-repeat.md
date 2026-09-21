# v17 全类别受限长尾采样实施计划

## 完成复盘（2026-09-21，优先于下方交付状态）

用户完成55轮，最佳第15轮AP95=0.3851431884、AP50=0.6172845371，末轮AP95=0.36226。每轮1798主样本（1709基本+89额外），55轮计划/实际主索引匹配。船和垃圾桶改善但车/座椅/标志等回退，总体低于v14训练最佳0.3921497554。不是采样未执行，也不能据此推断所有重采样必然无效。未独立复评，未提供赛事成绩。后续v18关闭重复，转向分模态首层学习率；下方“尚未训练”是当时交付状态。

> 用户已批准全类别受限采样并要求参考v4；使用writing-plans、executing-plans在现有工作区实现。用户自行训练，本轮不创建工作树、不委派、不测试、不执行Python检查、不训练/推理/复评、不提交推送。用户禁测要求优先于技能的TDD步骤。

**Goal：** 在当前清洗1709/291、同协议评估下，以v14为基准检验受限长尾采样是否带来总体AP95净收益。

**Architecture：** 沿用官方YOLO26l迁移、五通道首层融合、骨干低学习率。仅在训练加载器加入有界采样；不复制数据文件，不修改标签、验证集、模型结构或预测实现。

**Tech Stack：** 本机Ultralytics8.4.152、PyTorch2.14.0；不升级依赖。

## v4参数复核与决策

已读取`runs/detect/AIC_RGBIRDepth_yolo26l_1280_v4_clean_lr1e4/args.yaml`。保留其1280、batch4/nbs16、AdamW主lr1e-4、lrf0.01、预热5、weight_decay0.0005、mosaic0.5/scale0.3/translate0.1/fliplr0.5/HSV0。

- v4为epochs5000/close_mosaic4900，即第101轮关拼图，实际完成200轮；不把名义5000轮当作成功原因。v17恢复v14的200轮余弦/close_mosaic100、第101轮关拼图，撤回v16的60轮组合。
- 保留v14骨干2e-5与patience40；v4全模型1e-4/patience100不一起搬回，以避免新增混杂变量及无效长训。
- cls_pw保持0，不叠加v4的0.25和长尾采样；auto_augment/erasing为分类参数，不因v4文件中出现而移植到检测增强。
- cache保持False，避免恢复大体积磁盘缓存；nms保持True及现有FP32内容裁框协议，不依据v4的null改变输出分支。
- val_batch2沿用v16及v14独立复评，与v14训练时batch1区分；不宣称严格单变量实验。
- 沿用v14全部阶段预算门槛10/20/50/75/100/125/150/175/200轮：0.32/0.34/0.36/0.38/0.40/0.45/0.50/0.55/0.60。门槛不是效果预测，不放宽以无限重试。

## 采样接口与限制

`FusionRecipe`新增`repeat_threshold=0.10`、`repeat_max_factor=2.0`、`repeat_extra_fraction=0.10`、`repeat_group_limit=2`、`repeat_ball_cap=1.25`；旧配方threshold默认0关闭。

训练标签统计每类出现图片数n_c，N为训练图数：`r_c=min(2,max(1,sqrt(0.10/(n_c/N))))`；单图取所属类别最大值，不相加。含ball图片整体再限到1.25，避免通过共现类别绕开球类保护。空标签图片保持1。

参考[Detectron2官方RepeatFactorTrainingSampler实现](https://detectron2.readthedocs.io/en/latest/_modules/detectron2/data/samplers/distributed_sampler.html)。本项目额外的限额和固定轮长不是官方算法原样复现，也不保证达到未受限的期望倍率。

每图额外权重为r_i-1。按组汇总权重并截断到每组2个，向下取整总需求，再限制到floor(0.1N)及候选容量，得到固定K。每轮按seed+epoch加权无放回选K个额外主样本，组内最多2个；与全体N个基本索引合并打乱。实际概率受到组/总量约束，r_i不是实际出现次数保证。所有轮次N+K固定，不改框架优化器/累积计数逻辑。

场景组：文件名去掉最后一个下划线字段作为序列；无此结构保留独立名称。已审阅的六张广场图、门岗对和路口对显式合组。仅基于现有训练审阅，不读取复赛测试图；不声称完成全量视觉去重。

限制针对主样本索引，原生Mosaic还会选附加图，故不能声称每张源图在全部拼图中最多出现两次。`sampling_history.csv`区分原始含类图数、计划/实际主样本含类次数及增强后实际目标框数；后者包含Mosaic及几何过滤影响。`sampling_plan.json`存倍率/分组/固定K，`sampling_epochs.jsonl`存轮次/尝试号/额外文件名/索引摘要。

加载器仅在开启采样时使用有限epoch迭代，防止InfiniteDataLoader提前预取下一轮旧计划；保留workers4/预取1/不锁页，持久worker仅在关Mosaic或清理时重启。epoch_start显式设置采样轮次，重试有尝试号，正常完成轮末核对计划与实际主样本次数。断点按完整源码/配方签名保护，不恢复v14–v16旧last。

## 实施任务

- [x] 通读main.py、aic/training.py及本机加载器/训练循环，读取v4参数和v14配方。
- [ ] 在aic/training.py内实现采样器、有限加载器和逐轮审计；不新增根目录脚本，不改变默认关闭路径。
- [ ] main.py切v17，从官方模型重迁移，恢复v14日程并显式加入采样参数，保留全部YOLO分节配置。
- [ ] 同步当前说明、v16结果与v17操作边界；仅阅读代码及Git实际差异，不运行检查脚本/测试。

## 用户运行及验收

```powershell
uv run python main.py
```

输出新目录`runs/detect/AIC_RGBIRDepth_yolo26l_1280_v17_bounded_repeat`，重名自动加后缀。比较`best_validation/metrics.json`总体AP95及全部12类，不按类别挑权重合并。v14训练峰值0.39215、独立复评0.3917876103是不同口径；v17未经训练不能预填成绩。v4旧256张验证与当前291张不同，初赛55.6000不是复赛分数。

后续选择v17时，使用实际训练输出目录：

```powershell
uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_yolo26l_1280_v17_bounded_repeat/weights/best.pt --source "数据集/复赛测试集" --imgsz 1280 --height 1280 --output predict_v17_round2
```

本轮不改变predict.py默认值，不新增TTA、多模型融合或测试集伪标签。不保证AP95=0.60或复赛60分；没有模型运行证据时不宣称已验证可运行。
