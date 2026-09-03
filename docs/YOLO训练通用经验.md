# YOLO 训练通用经验（实战沉淀）

> 沉淀自多个 YOLO 实战项目（火灾检测 fire_detector、暴力打架检测 fight_detector 等）。
> 环境：Ultralytics 8.4.x · YOLO26 · Windows 11 · RTX 5080 16G · torch 2.9.1+cu130。
> 所有代码改路径即可复用；目录规范详见同目录《YOLO数据集目录结构与配置规范.md》。

## 0. 全流程地图

```text
造数据(视频抽帧/外部数据集) → 建骨架 → 清洗合并(配对/统一id) → 半自动标注(本地预标注+CVAT复核)
→ 训练(预训练权重起步) → 推理验证(视频/图片/摄像头) → 复盘(results.csv + PR曲线) → 迭代
```

配套脚本索引：①建骨架 ②路径清单 ③配对清洗 ④统一类别id ⑤视频抽帧 ⑥ffmpeg转码
⑦本地预标注 ⑧CVAT云端标注 ⑨训练 ⑩推理 ⑪复盘

## 1. 数据集工程

### 1.1 一键生成数据集骨架

```python
from pathlib import Path

def create_yolo_dir(project_name: str):
    """生成 <name>/{train,val,test}/{images,labels} + 空 yaml"""
    for split in ("train", "val", "test"):
        Path(f"{project_name}/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{project_name}/{split}/labels").mkdir(parents=True, exist_ok=True)
    Path(f"{project_name}/{project_name}.yaml").write_text(
        "train: ./train/images\nval: ./val/images\ntest: ./test/images\nnames:\n",
        encoding="utf-8",
    )
```

> ⚠️ 实战踩过：手写 6 个 mkdir 容易复制粘贴出错（曾把 train/labels 写了 3 遍，val/test 的 labels 根本没建）。用循环生成最稳。

### 1.2 生成图片路径清单 txt

递归收集图片相对路径写成 txt，给 CVAT 上传 / 自定义数据列表用：

```python
import os

def generate_txt_for_cvat(target_dir, output_filename):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    lines = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.lower().endswith(exts):
                # join 后统一转正斜杠，跨系统不踩坑
                lines.append(os.path.join(root, f).replace('\', '/'))
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"已记录 {len(lines)} 条路径 -> {output_filename}")
```

### 1.3 images/labels 配对清洗（训练前必做）

按"文件名（去后缀）"配对，只保留**既有图又有标注**的样本，能揪出所有脏数据：

```python
import os, shutil

def collect_matched_data(image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_dict = {os.path.splitext(f)[0]: f
                  for f in os.listdir(image_dir) if f.lower().endswith(exts)}
    count = 0
    for f in os.listdir(label_dir):
        key = os.path.splitext(f)[0]
        if f.lower().endswith('.txt') and key in image_dict:
            img = image_dict[key]
            shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, img))
            shutil.copy(os.path.join(label_dir, f), os.path.join(output_dir, f))
            count += 1
    print(f"配对成功并复制 {count} 对 -> {output_dir}")
```

> 用 copy 原件不动，确认无误后再替换；想直接剪切改 `shutil.move`。

### 1.4 批量统一类别 id（合并多数据集第一步）

不同来源数据集的类别编号体系不同，合并前把标注第一列统一成目标索引：

```python
import os

def unify_class_ids(label_dirs, class_id='0'):
    for folder in label_dirs:
        if not os.path.exists(folder):
            continue
        for name in os.listdir(folder):
            if not name.endswith('.txt'):
                continue
            path = os.path.join(folder, name)
            with open(path, 'r') as f:
                lines = f.readlines()
            with open(path, 'w') as f:
                for line in lines:
                    parts = line.split()
                    if parts:
                        parts[0] = class_id          # 强行统一类别索引
                        f.write(' '.join(parts) + '\n')
```

### 1.5 视频抽帧造数据

每秒抽 1 帧（`frame_count % fps == 0`）即可——相邻帧高度冗余，抽更密意义不大：

```python
import cv2, os

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开: {video_path}"); return
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"总帧数 {total}，帧率 {fps} FPS，约抽 {total // fps} 张")
    i = saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % fps == 0:
            saved += 1
            cv2.imwrite(os.path.join(output_dir, f"{saved}.png"), frame)
        i += 1
    cap.release()
    print(f"完成，共 {saved} 张 -> {output_dir}")
```

### 1.6 ffmpeg 批量转码

```bat
:: Windows CMD，当前目录所有 .mov 无损封装为 .mp4（-c copy 不重编码）
for %i in (*.mov) do ffmpeg -i "%i" -c copy "%~ni.mp4"
```

### 1.7 数据配比与划分经验

1. **负样本占 5%~10%**：混入"什么都不是"的背景图，明显压误检。实例：火灾项目靠 5% 负样本防止把太阳、灯光识别成火源；打架项目的 normal 类同理。
2. **验证集 10%~20% 就够**：实例教训——火灾项目把 33% 数据分去验证了，浪费训练数据。
3. **类别索引与 yaml 严格对齐**：不同项目索引顺序可以不同（火灾 `0:normal,1:fire`，打架 `0:fight,1:normal`），但**同一项目内一经训练就固定**，中途调换/增删 = 作废重来。
4. **空 txt 是合法标注**：代表纯背景负样本图，别当垃圾清掉；用 1.3 的配对逻辑保持图-标一一对应。
5. **合并后用计数核对**：实际图片数应等于标注数；与计划量对不上且等比缩减，说明合并时删过样本——要能说清删了什么。实例：打架项目计划 31680/4746/2326，实际入库 25878/3697/1849。

## 2. 半自动标注闭环（数据飞轮）

流程：训练好的模型 → 批量预标注 → 人工只做复核修正 → 新数据回流再训练。人工从零标注 → 只复核，效率差一个量级。

### 2.1 本地批量预标注

```python
import os
from pathlib import Path
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
img_folder, label_folder = "images/val/2", "labels/val/2"
os.makedirs(label_folder, exist_ok=True)

for img_path in list(Path(img_folder).glob("*.jpg")) + list(Path(img_folder).glob("*.png")):
    result = model(img_path, conf=0.25)[0]        # conf 0.25 起步，宁多勿漏
    with open(os.path.join(label_folder, img_path.stem + ".txt"), "w") as f:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()     # 已归一化 cx cy w h
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        # 无检测也保留空 txt，维持图-标一一对应
    print(f"已处理: {img_path.name}")
```

### 2.2 接入 CVAT 云端自动标注

思路：本地模型包装成 `DetectionFunctionSpec` → cvat-sdk 对指定 Job 批量预标注：

```python
from cvat_sdk.core.client import Client
from cvat_sdk.auto_annotation.functions import annotate_job
from cvat_sdk.auto_annotation.interface import (DetectionFunctionSpec,
                                                label_spec, rectangle)
from ultralytics import YOLO
import PIL.Image

model = YOLO("./runs/detect/train/weights/best.pt")
spec = DetectionFunctionSpec(labels=[label_spec(n, i) for i, n in model.names.items()])

def detect(context, image: PIL.Image.Image):
    results = model.predict(source=image, verbose=False)
    for r in results:
        if r.boxes is not None:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                yield rectangle(int(cls.item()), [p.item() for p in box])

with Client(url="https://app.cvat.ai") as client:
    client.api_client.configuration.api_key["TokenAuth"] = API_KEY     # 从环境变量取
    client.api_client.configuration.api_key_prefix["TokenAuth"] = "Token"
    annotate_job(client, JOB_ID, func=detect, spec=spec, allow_unmatched_labels=True)
```

或 CLI：`cvat-cli --server-host app.cvat.ai --auth <账号>:<密码> auto-annotate <JOB_ID> --function-file ./yolo.py --model-path ./best.pt --allow-unmatched-labels`

> 🔐 **密钥纪律**：API Key / 账号密码不要明文写进代码——实战中曾在源码里留下过 CVAT 的 Key 和明文密码，文档扩散后就收不回来了。用环境变量；发现泄露立即到平台 Profile → Security 撤销换新。

## 3. 训练

### 3.1 全参数实战模板

```python
from ultralytics import YOLO

if __name__ == '__main__':   # Windows 必加！否则 DataLoader 多进程递归创建进程直接报错
    model = YOLO(r"yolo26n.pt")            # 永远从官方预训练权重起步，别从零训
    model.train(
        data=r"./my_dataset/my_dataset.yaml",
        epochs=10000,        # 轮数给足，配合 patience 早停兜底
        imgsz=640,
        patience=100,        # 验证指标 100 轮无改善 → 早停（默认即 100）
        batch=78,            # 按显存往上试到 OOM 回退一格
        save_period=100,     # 每 100 轮存检查点（-1 = 只存 last.pt）
        plots=True,          # 生成 results.png / train_batch*.jpg，复盘必备
        val=True,
        cache=False,         # True=缓存进内存，"disk"=缓存到磁盘；省 IO 换内存
        device=0,
        workers=2,
        name="my_dataset",   # 结果目录名，多组实验靠它区分
    )
```

### 3.2 断点续训（中断后只改两处）

```python
model = YOLO(r"runs/detect/my_dataset/weights/last.pt")   # ① 改为加载 last.pt
model.train(resume=True, data=r"./my_dataset/my_dataset.yaml",
            epochs=10000, imgsz=640)                       # ② resume=True，其余参数照旧
```

> 实测：训练中途 Ctrl+C 强退（Windows 退出码 `0xC000013A` 属正常）后，resume 从中断 epoch 无缝续上（火灾项目从 epoch 125 续训）。所以 `save=True` 别关，它是续训的命根子。

### 3.3 导出

```python
model.export(format="engine")   # TensorRT；也支持 onnx / openvino / tflite / onnx
```

### 3.4 性能与显存经验（5080 16G 实测基线）

- **batch 从小往上试**：imgsz=640、yolo26n 时 5080 能吃 batch=78（显存 ~14G）。爆 OOM 就减半回退。
- **速度参考**：n 模型 640 分辨率、batch=78 时约 966 batch/epoch ≈ 4 分钟，验证 3915 张 ≈ 29 秒。换算自己的数据量估算时长。
- **cache**：内存富余 `cache=True` 最快；不够用 `cache="disk"` 也比每次读盘强；数据在机械盘就别开。
- **workers**：Windows 下 2~6 都行，太大会卡在进程创建。
- **优化器**：YOLO26 默认 `MuSGD(lr=0.01, momentum=0.9)`，一般不用动。
- **增强**：默认开 `auto_augment=randaugment`，小数据集保持默认即可提升泛化。

## 4. 推理与复盘

### 4.1 视频推理模板

```python
import cv2
from ultralytics import YOLO

yolo = YOLO(model="runs/detect/train/weights/best.pt", task="detect")
cv2.namedWindow("video", cv2.WINDOW_NORMAL)   # WINDOW_NORMAL 才允许手动/代码缩放窗口
cv2.resizeWindow("video", 1280, 720)

result = yolo.predict(source=r"test_video/1.mp4", save=True, device="cuda",
                      imgsz=640, conf=0.3,
                      stream=True)            # 迭代器模式，长视频省内存
for r in result:
    cv2.imshow("video", r.plot())             # 已画好框的帧
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
```

`source` 还支持：`0`（摄像头）、`"screen"`（录屏）、图片/文件夹路径、YouTube 链接；跟踪用 `yolo.track(...)`。

### 4.2 训练结果复盘

```python
import pandas as pd
from ultralytics import YOLO

df = pd.read_csv(r"runs/detect/train/results.csv")
df.columns = df.columns.str.strip()           # 去掉列名空格
# best.pt 默认按综合 fitness 选，不一定 mAP50-95 最高；手动找真正的最优轮次
best = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
print(f"最佳 epoch: {int(best['epoch'])}  mAP50={best['metrics/mAP50(B)']:.4f}  "
      f"mAP50-95={best['metrics/mAP50-95(B)']:.4f}")
print(f"P={best['metrics/precision(B)']:.4f}  R={best['metrics/recall(B)']:.4f}")

model = YOLO(r"runs/detect/train/weights/best.pt")
model.val(data=r"my_dataset/my_dataset.yaml", plots=True)   # 重新生成 PR / F1 / 混淆矩阵
```

## 5. 踩坑速查表

| 坑 | 现象 / 原因 | 解法 |
|---|---|---|
| Windows 多进程报错 | DataLoader 递归创建进程 | 训练代码包进 `if __name__ == '__main__':` |
| device 编号对不上 | 任务管理器里核显是 0，torch 里不是 | torch 里独立显卡优先；写 `device=0` 前先确认 |
| 训练中断 | Ctrl+C，退出码 0xC000013A | 正常现象；加载 last.pt + `resume=True` 续训 |
| 路径跨平台出错 | `\` 与 `/` 混用 | 统一正斜杠；yaml 相对路径以 yaml 所在目录为基准 |
| labels.cache 报错/过期 | 修改数据集后缓存没更新 | 直接删掉，下次训练自动重建 |
| 类别顺序错乱 | 训练后调换 names 顺序 | 一经训练索引即固定；要改就重标或做映射脚本 |
| 图标不配对 | 缺标注或缺图片 | 训练前跑 1.3 配对清洗 + 计数核对 |
| 密钥泄露 | API Key/密码明文写在源码 | 环境变量管理；泄露立即平台撤销 |
