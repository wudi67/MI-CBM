# 工程验证记录

日期：2026-09-16。设备：NVIDIA GeForce RTX 4090；Python：`/root/miniforge3/envs/VQC/bin/python`。

## 自动检查

执行 `bash experiments/grouped_robot_independent/scripts/check.sh`：

- Ruff 检查与格式检查通过。
- ty 通过。
- Pylint 项目质量检查通过。
- 实际 CUDA 集成测试：`3 passed`，约 35 秒。
- 四个 shell 脚本 `bash -n` 通过。

测试覆盖两个完整种子；前后段实际 CUDA 梯度、与原训练循环相同的模型／Adam／RNG 结果、概念和 label 训练中途恢复、两个 label 模型配对、纠正保持 Born 分支、已完成结果不重写、源数据只读、配置／文件／样本顺序变更拒绝。合成测试数据故意没有 test 文件。

## 真实历史结果预检

输出：`outputs/grouped_robot_independent/preflight_20260916/`。

默认正式配置的预检成功接纳 `0/concept` 和 `0/independent` 两个历史终点，核对了完整预算、Adam、初始化、RNG、样本顺序、全部样本身份以及量子态复振幅。这一目录状态为 `paused`，没有执行训练。

## 真实 Robot 数据与 tmux 工程运行

输出：`outputs/grouped_robot_independent/engineering_cuda_20260916/`。

配置为 `development=true`、`fresh_all=true`、seed 0、train 2,048／validation 1,024、前后段各 1 epoch、训练 batch 1,024、评价 batch 2,048。

先用 `--max-steps 1` 在概念训练中途暂停，再通过 tmux 启动 `--resume` 恢复，最终完成 3 个训练任务、6 个 train/validation 评价条件，共 6 次 Adam 更新。暂停和完成后对应 tmux 会话均自动退出，日志保留。

| 训练任务 | CUDA 活跃参数 | 更新次数 | 峰值已分配显存 GiB |
|---|---:|---:|---:|
| concept | 240 | 2 | 0.722 |
| independent | 112 | 2 | 0.137 |
| no_feedback | 112 | 2 | 0.137 |

结果记录 `status=complete`、`test_read=false`、`test_evaluated=false`。已生成模型、逐样本概率、CSV、Markdown 及 PNG/PDF 汇总。

本次短跑仅验证工程链路，不构成模型性能结论。正式五种子目录尚未启动；正式预算仍为每种子概念 300 epoch、两条 label 路线各 300 epoch。
