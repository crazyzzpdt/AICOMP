# D-FINE-L 三模态实施方案

## 依据与范围

用户已确认 v5 best.pt 的初赛得分为 51.0830；提交记录为 2026-09-17 23:07:50，SHA256 为 `33caf130bfb8b8cb0b6c35e31809fc2ac54ea094d5a347c535bbaf71693614d4`。1280、conf=0.001、iou=0.7、单标签 NMS、FP32、无 TTA，1000 张。不能归因为默认 0.25 阈值；0.25 仅是图片显示阈值。best.pt 来自早期最佳，不把末轮退化直接当作该提交降分的原因。

v4 55.6000 保留为正式基线。新路线是 D-FINE-L Objects365 E25 预训练，五通道早期融合，60 轮温和微调，不承诺达到 60 分。官方规则以 mAP50-95 为主，禁止额外训练数据和不同模型/检查点简单集成。

## 实施清单

- [x] 将官方 D-FINE 固定到 `956d1709314c2c6a4df6f34de232054578a7449f`，准备本地公开预训练权重与依赖；不上传赛事数据。
- [x] 新增 `D细化训练.py`：适配五通道输入、类别迁移、现有 datasets、60 轮训练与随训练验证、双指标最佳权重留存。
- [x] 更新唯一训练入口 `main.py`，保留原 YOLO 组件和全部训练产物。
- [x] 更新 `predict.py`，兼容 D-FINE 与旧 YOLO；共享坐标还原、六列标签与不覆盖的提交打包流程。
- [x] 同步使用文档；不运行测试、冒烟、额外推理或训练。检查范围见末节，不等同运行验证。

不执行技能模板中的云端训练、Hub 上传、自动测试或多代理评审；用户明确要求本机离线运行、不做冗余测试。本文件同时记录已确认的设计与实施进度，不额外生成重复计划。

## 参数与适配边界

1280 输入；物理 batch 初始 2、梯度累积至约 16，资源由用户调整。AdamW 主学习率 1e-4、骨干 1e-5、权重衰减 1e-4；3 轮预热、余弦衰减至 0.01 倍。60 轮固定预算，每轮保留最佳，不延续 YOLO 的 5000 轮双头日程。

保持现有 1744/256 图像划分与清洗标签，不生成另一份数据集。只做同步翻转、温和尺度变化和 RGB 颜色增强；不启用 v5 的弱类重复采样、定向裁剪或模态随机缺失。不是重新清洗或自动删除难例。

首层 RGB 权重完整迁移，新增两个输入通道从零学习，必须解除输入层冻结；小 batch 使用预训练骨干冻结归一化统计。官方 Objects365 权重是 366 个输出槽，类别映射须处理背景占位，不照抄 Person/Car 示例。宽泛或没有直接对应的类别不能假装一一等价。

官方 D-FINE 输入为像素缩放到 0–1，并未使用 ImageNet 均值标准差；五通道保持同一约定，深度沿用本项目编码。预测使用训练时相同的等比缩放与填充，去掉填充后恢复原图坐标。D-FINE 使用原生查询候选排序，不套用 YOLO NMS。

## 官方来源

- https://github.com/Peterande/D-FINE
- https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/configs/dfine/custom/objects365/dfine_hgnetv2_l_obj2custom.yml
- https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj365_e25.pth

Objects365 仅用于允许的公开预训练权重，不下载其训练数据。第三方源码保留自身 LICENSE，权重还需遵守来源数据条款。

## 本机执行

依赖已安装，保持原 torch 2.14.0、torchvision 与 Ultralytics。新增依赖主要用于官方源码导入、匹配和训练期间 COCO 指标计算；calflops/transformers 为官方源码导入链依赖，不调用其联网模型或 FLOPs 试跑。

```powershell
uv run python main.py
```

`batch=2` 是未经本机新架构显存测量的保守起点，不保证 1280 在所有样本上不 OOM。用户可改为1/2/4等，同时 effective_batch=16 保持整数倍；val_batch=1 为 FP32 验证。没有资源探测、自动试训或 OOM 后隐式重跑。

正常输出：

```text
runs/detect/AIC_RGBIRDepth_dfine_l_1280_v6/
├── args.yaml
├── optimization_recipe.json
├── train.log
├── results.csv
├── results.png
├── per_class_metrics.csv
├── code/
└── weights/{best.pth,best_map50.pth,last.pth,epoch10.pth,...}
```

权重包含模型、同轨迹 EMA、优化器和随机状态；best.pth 按 mAP50-95 选择，best_map50.pth 单独保存，不集成。只有 main.py 运行后才会生成上述目录。普通训练每轮一次验证，不另外复评训练集/测试集；官方损失依赖训练辅助输出，因此不伪造 val/loss 曲线。COCO 评估器与历史 Ultralytics AP 实现并非完全相同，不把微小本地变化直接等同线上收益。

中断恢复时 RESUME_PATH 指向同配方 last.pth，新建结果目录并复制原来的最佳权重，原运行不覆盖。配方、数据签名或源码版本不同会拒绝续训；已完成60轮拒绝恢复。旧 YOLO main.py 仍可从 Git 提交 df8f0e5 查阅，三模态训练.py、训练优化.py 与全部旧权重均保留；不要将新 .pth 交给 YOLO 加载。

训练完成后：

```powershell
uv run python predict.py --weights runs/detect/AIC_RGBIRDepth_dfine_l_1280_v6/weights/best.pth --output predict_v6
```

若实际训练目录因重名带时间戳，使用实际路径。预测默认路径仍由用户维护，不自动选最新模型。正式只提交 predict_v6/比赛提交内容/submission.zip；旧 YOLO .pt 也仍可显式指定。

## 类别迁移说明

| 比赛类别 | 官方 Objects365 初始化槽位（1 起始） |
|---|---|
| person / boat | 1 Person / 22 Boat |
| seat / sign / bicycle | 3 Chair / 90 Traffic Sign / 47 Bicycle |
| car / ball | 6 Car / 157 Other Balls |
| light / garbage can / tricycle | 7 Lamp / 45 Trash bin Can / 184 Tricycle |
| animal / uav | 无直接同义的整体类别，保留新头初始化；共享预训练骨干仍迁移 |

seat/sign/car/ball 等包含比代表类别更宽泛的对象，此表只是参数初始化，不修改比赛标签、不认为定义完全等价。球类来自 Other Balls，不漏掉编号偏移；light 按照官方“路灯和室内照明灯”定义，不映射到 Traffic Light。分类头各层和去噪嵌入同步处理。运行启动时遇到非366槽权重或未处理的参数不匹配会报错，不静默退化为随机骨干。

## 新机器准备（仅下载公开代码和权重，训练前完成）

已有对应目录时不要重复克隆或覆盖。首次准备可执行：

```powershell
git -c http.sslBackend=openssl clone https://github.com/Peterande/D-FINE.git vendor/D-FINE
git -C vendor/D-FINE checkout --detach 956d1709314c2c6a4df6f34de232054578a7449f
uv sync --frozen
```

再从上面的官方权重链接下载 dfine_l_obj365_e25.pth，放入 orgin_models。此文件大小为 128413144 字节，本机 SHA256 为 `af2ec45453ce9dfb3208f852ae199adfd8fe232c458016b4bcb27bced23a69db`。本机已经完成以上准备，不需要再次下载。

## 检查范围

2026-09-18 预测吞吐优化：`predict.py` 采用 `PredictionConfig` 分节入口，通用调度移到 `三模态预测.py`，`DFinePredictor` 增加批量接口。默认 batch=8、读取/保存各4线程、预取3批、当前批次锁页传输；继续使用1280和FP32，不改训练及验证循环。完整配置与运行方法见 [预测与赛事提交](预测与赛事提交.md)。此次仅做静态检查，没有运行模型推理或测速，性能收益与峰值显存尚未实测。

仅核对源码、配置、差异、文件哈希，并在 CPU 读取公开预训练检查点的键与形状：1173 个张量，输入层 (32,3,3,3)，分类头 (366,256)，去噪嵌入 (367,256)。main.py、D细化训练.py、predict.py 通过 AST 静态语法解析，Git 差异无空白格式错误；这不是运行测试。没有实例化新网络试跑、执行前向/反向、单元测试、冒烟、训练或额外评估。数据语义复核、难例挖掘、TTA/WBF 均未执行；不把代码编写完成说成已验证提分。
