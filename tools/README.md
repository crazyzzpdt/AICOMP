# 工具目录

- `split_official_dataset.py`：v27官方1900/100整组与稀有类覆盖划分；已执行，审计及旧清单见runs/dataset_cleaning/official_labels_split_1900_100_v27，禁止重复落位，不修改标签。

- `prepare_dataset.py`：历史数据准备/审计工具。包含旧清洗逻辑；当前官方原标签1900/100已准备好，不要执行旧清洗重建。
- `IntegrateAndPackage.py`：复赛提交内容的唯一整合工具；predict1/predict2预测完成后直接调用，生成固定团队目录，并纳入技术方案PDF、实际权重、submission.zip、源码、依赖和模型评估。
- `dfine/`：由 `src/D-FINE/tools`、`reference` 移入的上游导出、推理、数据处理与可视化示例。它们不是赛事五通道入口，数据示例也不应用来改写官方标签。
- `archive/run_snapshots/`：本地历史代码快照，按原 `runs` 相对路径保存，禁止当作当前配方直接执行，不上传Git。

正式训练/预测只从根目录的 `train1.py`、`train2.py`、`predict1.py`、`predict2.py` 启动。工具只在明确需要时手动运行，不能隐式挂在训练前。

## runs 中仍保留什么

| 目录 | 用途 |
|---|---|
| `detect/` | 权重、逐轮指标、可视化及对应源码快照 |
| `dataset_cleaning/` | 官方1900/100清单、标签哈希和旧标签审计，约27.24 MiB；两个训练入口仍在引用 |
| `diagnostics/launch_history/` | 已集中归档6组旧启动/错误日志，保留首轮/第二轮内存故障证据 |
| `diagnostics/v27_cpu_bn_20260926/` | BN异常诊断的3份小型JSON，仍有追溯价值 |
| `checks_20260921/`、`smoke_main/`、`train1_checks/` | 历史短流程产物；2026-10-05只读检查已不存在，不属于当前保留目录 |
| `evaluations/`、`local_validation/` | 历史独立复评结果，不能与新划分直接横比 |
| `modality_audit/` | 模态数值统计与审阅结果 |

2026-09-24盘点时，`runs/detect`之外没有待迁移的 `.py/.ps1/.bat/.cmd/.sh` 文件；工具源码已在归档。日志、审计、权重均未删除，源码目录不接收这些输出。

2026-10-03按用户清理要求，已删除5个项目测试源码及`check_train1.py`、`validate_local.py`两个检查工具。它们无正式入口引用，后者还依赖旧aic路径和291张划分；需要追溯时可从Git提交`e43d1c2`恢复。历史文档中的测试通过记录不因此抹去，但其中旧运行命令已不是当前工具入口。

10月3日删除临时测试权重/目录和缓存的PowerShell操作被自动审批以`blocked by policy`拒绝，包括缩小到单个明确目录的尝试；当时没有改用其他方式绕过限制。10月5日只读检查确认上述三个runs测试目录已不存在，`tests`和`.pytest_cache`仍在；无法据此声称本轮执行了删除或核实了3.26 GiB实际空间释放。用户现已暂停清理，新项目位于`A:\AIC`，后续安排见[README](../README.md)。

6组有用启动日志已经迁入`runs/diagnostics/launch_history/`，旧文档中的路径按同名目录查找。正式`runs/detect`、数据审计、旧修正标签备份、模态审计和已有独立复评全部保留。本次仅整理文档及Git归档，未运行测试或模型。
