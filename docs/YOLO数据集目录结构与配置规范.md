# YOLO 数据集目录结构与配置文件规范

> 适用：Ultralytics YOLO 系列训练（detect / segment / pose / classify / obb 均沿用同一组织方式）。

## 1. 标准目录结构

```text
<dataset_name>/                      # 数据集根目录，与 yaml 同名
├── <dataset_name>.yaml              # 数据集配置文件
├── 数据分配.txt                      # 数据来源与划分记录（自用备注，非必需）
├── train/
│   ├── images/                      # 训练集图片
│   ├── labels/                      # 训练集标注（YOLO txt）
│   └── labels.cache                 # Ultralytics 首次训练自动生成（可删，会重建）
├── val/
│   ├── images/
│   ├── labels/
│   └── labels.cache
└── test/
    ├── images/
    └── labels/
```

要点：

1. `train / val / test` 三级划分；`test` 可选，最少保留 `train + val`。
2. 每个划分下 `images/` 与 `labels/` 平级、**同名配对**：`images/xxx.jpg` ↔ `labels/xxx.txt`（仅扩展名不同）。
3. `labels.cache` 是首次训练时由 Ultralytics 生成的标签缓存，用于加速后续训练；删除后下次训练自动重建。
4. 配置文件命名 = 数据集目录名，训练时直接 `data=<dataset_name>.yaml` 即可。

## 2. 数据集配置文件（yaml）

实例 `fight_detector.yaml`：

```yaml
train: ./train/images
val: ./val/images
test: ./test/images
# 类别名称
names:
  0: fight
  1: normal
```

说明：

- 路径写相对路径（相对 yaml 所在目录），Ultralytics 会以 yaml 位置为基准解析；也可写绝对路径。
- `names` 用 `类别索引: 类别名`，索引与标注 txt 中每行第一个数字对应。
- 类别索引从 0 开始；类别顺序一经训练就固定，不要中途调换或增删。
- 只需训练/验证时可省略 `test:` 一行。

## 3. 标注格式（YOLO txt）

每张图片对应一个同名 txt，每行一个目标框：

```text
<class_id> <x_center> <y_center> <width> <height>
```

- 坐标与宽高均为 0~1 归一化值（相对整图，非像素）。
- `class_id` 即 yaml `names` 中的索引。
- 无目标的图片允许空 txt（作为负样本参与训练）。

## 4. 实例统计：fight_detector

| 划分  |   图片 | 标注 txt |                          合计（含 cache） |
| ----- | -----: | -------: | ----------------------------------------: |
| train | 25,878 |   25,878 |                                    51,757 |
| val   |  3,697 |    3,697 |                                     7,395 |
| test  |  1,849 |    1,849 |                                     3,698 |
| 合计  | 31,424 |   31,424 | 62,852（另有 yaml、数据分配.txt 各 1 个） |

- 类别：`0 = fight`（打架）、`1 = normal`（正常）。
- 图片与标注一一对应，无缺漏。
- 文件名含 `.rf.<hash>` 后缀，为 Roboflow（YOLO26 格式导出）的典型命名。

## 5. 数据来源与分配记录（数据分配.txt）

用 txt 记录每个来源数据集的去向，便于追溯：

| 来源数据集                                                | 分配去向                  |
| --------------------------------------------------------- | ------------------------- |
| fight-detection.v2i.yolo26                                | 31,426 张 → 全部进训练集  |
| Fighting_Detection_Dataset.v1i.yolo26                     | 254 张 → 进训练集         |
| RWF-2000_Only_Physical_violence.v15-rwf2000-2class.yolo26 | 4,746 张 → 校验（验证集） |
| fighting.v1i.yolo26                                       | 2,326 张 → 测试集         |

备注（原文）：实际使用的仅仅是 Fighting_Detection_Dataset.v1i.yolo26。

> 注意：txt 记录的计划量（31,680 / 4,746 / 2,326）与实际入库量（25,878 / 3,697 / 1,849）不一致，比例接近等比缩减，推测合并时剔除了重复或损坏样本。

## 6. 使用方法

```bash
# 训练
yolo detect train data=path/to/fight_detector.yaml model=yolo11n.pt imgsz=640 epochs=100

# 在测试集上评估
yolo detect val data=path/to/fight_detector.yaml model=best.pt split=test
```

```python
# Python API
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="path/to/fight_detector.yaml", imgsz=640, epochs=100)
```

## 7. 新数据集接入清单

1. 建目录：`<name>/{train,val,test}/{images,labels}`（test 可选）。
2. 按 `images/xxx.ext ↔ labels/xxx.txt` 同名规则放入图片与标注。
3. 写 `<name>.yaml`：train / val / test 路径 + `names` 类别表。
4. 写 `数据分配.txt` 记录来源与去向（可选但推荐）。
5. 校验：跑一次 `model.val(data="<name>.yaml", split="val")`，能正确统计图片数即结构无误。
