# 工程验证记录

日期：2026-09-17。环境：`/root/miniforge3/envs/VQC`，NVIDIA GeForce RTX 4090。

## 自动检查

`bash experiments/grouped_robot_four_modes/scripts/check.sh` 全部通过：

- Ruff 检查及格式检查。
- ty。
- 项目 Pylint 质量检查。
- 实际 CUDA 集成测试：`3 passed, 1 warning in 113.06s`。warning 来自既有 PyTorch/pynvml 环境。
- 四个 shell 脚本另行通过 `bash -n` 检查。

CUDA 测试覆盖：

1. 两个种子、六次新训练、36 个评价条件；原始前后段初始化复用、总优化预算配对、两部分均得到 CUDA 梯度、主表 Standard 概念列留空、均值／样本标准差与逐种子收益核对。测试数据没有 test 文件；历史模型、原始预测和心跳文件的内容及修改时间均不改变。
2. 三条路线与手工构造的 loss、模型和 Adam 更新完全一致；直接逐分支演化与优化实现的 label 概率一致。翻转真实概念不影响 Standard 更新，但会改变 Joint 更新。Joint 正常与全部纠正保留相同原概念概率。
3. 在 Standard、Joint 及下一种子训练中断后恢复，最终模型、Adam、RNG、进度和损失累计与连续训练完全一致；已有评价不重写。样本顺序、配置、来源和输出篡改被拒绝。

## 正式来源预检

输出：`outputs/grouped_robot_four_modes/preflight_20260917/`。

完成实际 Sequential／Independent 五种子来源核验，沿用原始 train 18,432／validation 6,144、40 维输入和方差路由。确认所有旧模型、训练预算、初始化、样本身份、概率及来源记录符合协议。

状态 `paused`，新增优化更新数为零。401 个源文件哈希与当前代码一致。默认正式目录未创建，未启动正式 15 次训练。

## 实际 Robot 数据、正式 batch 和 tmux

输出：`outputs/grouped_robot_four_modes/engineering_cuda_20260917/`。

引用已完成的 `outputs/grouped_robot_sequential/engineering_cuda_20260917/`：seed 0，train 2,048／validation 1,024。该来源的概念和 label 各训练 1 epoch，因此本轮三条新路线各训练 2 epoch，以匹配总更新预算。

训练 batch 1,024、评价 batch 2,048，均沿用正式设置。先在第 1 次更新后暂停，再通过 tmux `--resume` 完成剩余步骤。

| 路线 | Adam 更新 | CUDA 活跃参数 | 峰值已分配显存 |
|---|---:|---:|---:|
| Standard | 4 | 352 | 0.789 GiB |
| Joint | 4 | 352 | 0.777 GiB |
| Joint 无反馈 | 4 | 352 | 0.777 GiB |

- 前后两部分均获得有限且非零梯度，参数和输入均在 `cuda:0`。
- 共 3 次新训练、12 次 Adam 更新、18 个评价条件，其中 10 个旧评价只读引用。
- 生成 checkpoint、原始分支概率、四模式表、配对统计、训练曲线、Markdown、PNG/PDF 和结果锁。
- 72 个输出产物哈希及 401 个源文件哈希检查通过。
- `test_read=false`、`test_evaluated=false`。
- 暂停、恢复完成后对应 tmux 会话均自动退出，日志保留。
- 启动器、状态命令及默认 `--plan-only` 均验证可用。

这只是工程通路验证，不是正式性能结论；显存数字对应此短流程。正式运行使用五个种子，每路线 600 epoch，共 15 次训练和 162,000 次新优化更新。
