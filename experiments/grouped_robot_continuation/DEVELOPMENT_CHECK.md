# Robot 续训实验工程检查

检查日期：2026-09-15。检查通过；默认 100→300 实验尚未启动。

## 代码检查和恢复测试

执行 `bash experiments/grouped_robot_continuation/scripts/check.sh`：

- Ruff 检查和格式检查通过。
- ty 通过。
- Pylint 通过。
- pytest：3 passed，27.10 秒；仅有环境原有的 pynvml 弃用警告。

测试在 CUDA 上调用原训练循环，对照连续训练与恢复完整 Adam 后的续训，
核对模型、优化器状态、历史和样本顺序；覆盖半个 epoch 中断恢复、
两份新 label 电路复用原初始参数和样本顺序、冻结参数不产生梯度、
已完成结果复用，以及错误配置、错误样本顺序和预测文件篡改拒绝。
测试使用没有测试集文件的合成数据 fixture。

## 默认实验预检

`scripts/run.sh --plan-only` 对默认来源
`outputs/grouped_robot_pilot/robot_v5_l4_seed0/` 成功完成配置、原结果文件和 CUDA 检查。
计划为概念 300 + label 100，以及概念 100 + label 300；
batch=1024、lr=0.01、RTX 4090，新增 14400 次 Adam 更新。
此命令没有创建默认结果目录，也没有训练。

## 真实数据、CUDA 和 tmux 检查

运行命令：

```bash
bash experiments/grouped_robot_continuation/scripts/launch.sh \
  --session robot_continue_dev \
  --reference outputs/grouped_robot_pilot/development_check_final \
  --out outputs/grouped_robot_continuation/development_check \
  --development --concept-epochs 3 --head-epochs 3
```

复用真实 Robot 数据开发基准：训练 1024 张、验证 2048 张，
原概念 2 轮、原 label 2 轮，batch=1024、eval batch=2048。

| 训练任务 | 起止轮数 | 新增 Adam 更新 |
|---|---:|---:|
| long_concept / concept | 2→3 | 1 |
| long_concept / independent | 0→2 | 2 |
| long_concept / no_feedback | 0→2 | 2 |
| long_label / independent | 2→3 | 1 |
| long_label / no_feedback | 2→3 | 1 |

五项训练任务完成；24 个验证条件和 9 个训练集诊断完成，合计 33 条结果。
五项训练的输入及梯度检查均为 `cuda:0`，当前训练部分梯度非零，
冻结部分无梯度。概念训练峰值分配显存约 0.74 GiB；监控确认实际 CUDA 进程。
`summary.json` 标记 `engineering_subset=true`、`test_read=false`、`test_evaluated=false`。
完整核对结果锁中 160 个文件的哈希，查看两张汇总图，均通过。
结束后 `robot_continue_dev` tmux 会话自动退出。

随后使用新 tmux 会话执行 `--resume`，完整校验成功，没有新增训练或推理。
除运行日志、心跳和进程锁之外，161 个输出文件的哈希与修改时间保持一致。
恢复检查结束后 tmux 会话同样自动退出。

原默认实验、原开发实验及两套复用代码的 167 个文件，检查前后的哈希与修改时间一致
（不计 Python 字节码缓存）。原数据和模型还由程序的来源锁进行校验。

以上仅验证工程流程和隔离性；小规模、少轮数结果不能用于判断延长训练的性能收益。
默认输出 `outputs/grouped_robot_continuation/robot_v5_l4_seed0/` 尚未创建。
