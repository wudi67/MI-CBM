# 工程验证记录

日期：2026-09-12。以下验证代码与运行流程；完整100/200轮、三学习率的开发对照尚未启动。

## 静态检查和自动化测试

运行 `scripts/check.sh`：Ruff、格式检查、ty、Pylint全部通过，10项测试通过。
日志：[quality_checks.log](../../outputs/grouped_mlp_controls/quality_checks.log)。

测试覆盖两个任务的监督隔离、251/253参数量、32码概率分布、CUDA更新、
中途恢复与连续训练的模型/Adam/RNG逐位一致性，以及最后训练批结束但尚未验证时的恢复。
还检查固定末轮选择、已完成单元跳过、配置/输入/历史被修改时拒绝恢复和状态显示。

## 完整缓存的短程 CUDA 运行

目录：[cuda_smoke_20260912](../../outputs/grouped_mlp_controls/cuda_smoke_20260912)。

- 使用RTX 4090、CUDA，完整训练25,593张、验证5,479张，batch 1024。
- 概念模型2轮、Label模型3轮，每项学习率0.001/0.003/0.01，共6个单元完成。
- 首个单元第1步保存并暂停；恢复后完成全部单元。暂停和完成后worker均退出，tmux会话自动关闭。
- 两个输入缓存文件与VQC原文件字节一致，每轮实际样本排列与VQC历史匹配。
- 六个单元的初始化、排列、checkpoint和历史检查通过；各任务选中模型重新载入CUDA，
  完整验证集指标与已保存结果精确一致。
- 源码检查覆盖208个文件。测试集未处理、未评价。
- 由于本次仅2/3轮，汇总按设计不生成相对于100/200轮VQC的性能差值。

机器可读核验：[verification.json](../../outputs/grouped_mlp_controls/cuda_smoke_20260912/verification.json)。
短程汇总：[summary.md](../../outputs/grouped_mlp_controls/cuda_smoke_20260912/summary.md)。

## 匹配预算的汇总集成检查

另使用已有量子工程运行 `outputs/grouped_vqc_training_modes/tmux_auto_exit_check` 为参照：
训练/验证各18张，概念1轮、Label2轮，学习率0.001。两个MLP单元完成；
相同输入与顺序、初始化和预算检查全部通过，生成预期的4项VQC端点差值。
测试集未处理、未评价。该18张工程样例不用于性能判断。

证据：[summary.json](../../outputs/grouped_mlp_controls/reference_pairing_check_20260912/summary.json)。

## 下一步

按 [README.md](README.md) 的默认命令启动完整预算。
正式开发对照默认输出为 `outputs/grouped_mlp_controls/dsprites_seed0`，与以上工程检查分开保存。
