# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

2026 第八届 AIC 全球校园 AI 算法精英大赛参赛项目：**面向城市场景的视觉多模态目标检测**（算法挑战赛道）。

任务：对空间对齐的三模态图像（RGB 可见光 / 红外 / 深度）做 12 类目标检测，输出每张测试图的检测框 TXT。排行榜指标为 `mAP@50-95`。赛题权威参考：`比赛资料/2026 AIC 视觉多模态目标检测参赛手册.md`。

**当前进度**：仅完成 RGB-only 基线（`main.py` 用 YOLO26m 只训练 visible 图像）。多模态融合、本地验证集划分、推理生成提交 TXT、打包脚本均未实现。

## 常用命令

```bash
# 训练（入口脚本，RGB-only 基线）
python main.py

# 断点续训：把 main.py 中 YOLO(...) 改为加载 runs/detect/"AI COMP"/weights/last.pt，
# 并加 resume=True，其余参数照旧

# 验证（需先解决下方数据布局问题 + 训练产出权重）
yolo detect val data=datasets/data.yaml model=runs/detect/"AI COMP"/weights/best.pt
```

环境：Windows 11 + RTX 5080 16G，ultralytics 8.4.x（YOLO26），torch 2.9.1+cu130，`device=0`。

## 数据布局（重要：yaml 与实际数据不一致）

```text
训练集/AIC2026_Train_2000/    # 2000 组训练数据（有标注）
├── visible/                  # RGB 8-bit，00000004.jpg …
├── infrared/                 # 红外（存为三通道，本质同一热度灰度图复制三份）
├── depth/                    # 深度 16-bit，单位毫米；0 或极小值 = 无效深度
├── labels/                   # YOLO txt：cls cx cy w h（0-1 归一化）
└── new_labels_2000/          # 目前与 labels/ 文件名、内容相同，疑似副本

测试集/AIC2026_PHASE_1_1000/  # 1000 组初赛测试数据（无标注），visible/infrared/depth
datasets/data.yaml            # 12 类类别表（类别定义的权威来源）
```

**坑：`datasets/data.yaml` 指向 `train/images`、`val/images`，但这些目录不存在**——实际数据在 `训练集/`。跑训练前必须先把数据整理成 yaml 描述的结构（从训练集划分 train/val 放入 `datasets/`），或改写 yaml 路径。验证集只能从训练集内部划分（相邻/相同场景的样本不要分跨两侧，避免分数虚高）。

- 12 类：0=person, 1=boat, 2=animal, 3=seat, 4=sign, 5=bicycle, 6=car, 7=ball, 8=light, 9=garbage can, 10=uav, 11=tricycle。**索引一经训练即固定，禁止调换或增删**。
- 深度图必须按原始 16-bit 位深读取，不能当 8-bit 灰度图。

## 提交格式与比赛硬性约束

- 预测结果：每张测试图一个**同名** TXT（`00000004.jpg` → `00000004.txt`），每行 `class_id cx cy w h confidence`（比训练标签多一个置信度字段）。无目标也要交空 TXT；每图最多 100 框；打包 zip 上传。
- 禁止：使用官方 2000 组以外的训练数据；用测试集训练/标注；调用在线服务或 API；多模型投票/平均式集成。
- 允许：ImageNet / COCO / Objects365 等公开预训练权重。
- 三模态几何增强必须同步（同一变换应用于三张图与标签，保持空间对齐）。
- 成绩低于官方基线会被判无效。

## 代码与仓库约定

- `.gitignore` 已显式排除数据集目录（`训练集/`、`测试集/`）、训练输出（`runs/`）、`*.cache` 及图片/txt/权重等——仓库只跟踪代码和文档，**严禁 git add 数据集或大文件**（GitHub 单文件上限 100MB；历史上曾因暂存数据集使 .git 膨胀到 17GB，2026-09-04 已清理重写）。
- `比赛资料/视频讲解.mp4`、`比赛资料/参赛选手承诺书.pdf` 仅存于磁盘，不入库。
- 训练输出在 `runs/detect/AI COMP/`（`main.py` 中 `name="AI COMP"`）。
- 远程仓库：`origin = github.com/crazyzzpdt/AICOMP`（主分支 `main`，普通 push 即可，勿用 force）。
- 团队文档与提交信息使用中文，新增文档、注释请保持中文。

## Windows 训练注意事项（实战沉淀，详见 docs/YOLO训练通用经验.md）

- 训练代码必须包在 `if __name__ == '__main__':` 里，否则 DataLoader 多进程递归建进程直接报错。
- Ctrl+C 中断的退出码 `0xC000013A` 属正常现象；加载 last.pt + `resume=True` 可无缝续训。
- `workers` 取 0~6；爆 OOM 就把 batch 减半回退。
- 修改数据集后删掉 `labels.cache`，下次训练自动重建。

## 参考文档

- `docs/YOLO训练通用经验.md` —— 训练/标注/复盘全流程实战经验与脚本模板（配对清洗、预标注、断点续训、results.csv 复盘等）
- `docs/YOLO数据集目录结构与配置规范.md` —— YOLO 数据集目录与 yaml 规范
- `比赛资料/2026 AIC 视觉多模态目标检测参赛手册.md` —— 赛题手册（任务定义、类别表、评分规则、赛程、材料要求）
- `比赛资料/视频讲解-完整内容.md` —— 官方赛题视频逐字稿（数据集设计、三阶段难度递增说明）
