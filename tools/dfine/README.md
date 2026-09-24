# D-FINE 上游工具

原 `src/D-FINE/tools`、`src/D-FINE/reference` 已迁入本目录，保留原许可证及版权头。核心模型和配置仍在 `src/D-FINE/`。涉及源码导入的Python示例已改为通过项目 `src.dfine.runtime.check_source` 挂载官方包，默认配置使用绝对项目路径；手动 `--config` 应指向 `src/D-FINE/configs/...`。

这些是上游RGB模型的导出、部署、基准和数据示例，并非本项目五通道赛事接口。正式任务只用根 `train2.py/predict2.py`。`reference/safe_training.sh` 是上游Linux四卡示例，移位后先切换上游源码目录；不要用它代替本机单卡三模态训练。

迁移只做文本与引用检查，没有运行依赖安装、导出、训练、测试或预测，不保证所有上游可选依赖已安装。`src/D-FINE/README*.md` 保留上游原文；其中 `tools/...` 命令现在对应本目录，需同步使用新路径。
