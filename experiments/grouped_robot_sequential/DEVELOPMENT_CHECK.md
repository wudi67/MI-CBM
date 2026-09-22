# 工程验证记录

日期：2026-09-17。环境：`/root/miniforge3/envs/VQC`，NVIDIA GeForce RTX 4090。

## 自动检查

`bash experiments/grouped_robot_sequential/scripts/check.sh` 已通过：

- Ruff 检查与格式检查。
- ty。
- 项目 Pylint 质量检查。
- 实际 CUDA 集成测试：`3 passed, 1 warning in 41.34s`；warning 来自现有 PyTorch/pynvml 环境。
- 四个 shell 脚本的 `bash -n` 检查另行通过。

CUDA 测试覆盖：

1. 两个完整配对种子的训练与 20 个评价条件；已有模型、概率文件和心跳文件均不改写；无需 test 文件。
2. 手工构造逐分支测量控制，与继承训练循环产生完全相同的 loss、模型及 Adam 更新；翻转训练样本的真实概念，不影响 Sequential 更新。
3. minibatch 中断恢复与连续训练的模型、Adam、历史和 RNG 一致；已完成评价不重写；样本顺序、预算、配置及源／目标文件篡改被拒绝。

## 实际五种子来源预检

输出：`outputs/grouped_robot_sequential/preflight_20260917/`。

使用默认正式配置完成实际 Independent 五种子来源核查：数据、原始初始化、历史复用来源、模型、训练预算、样本顺序、冻结前半段及 30 个已有评价条件。包括核对旧 seed 0 的完整量子态振幅。

预检完成后状态为 `paused`，没有新增参数更新。

## 真实 Robot 小型运行与 tmux

输出：`outputs/grouped_robot_sequential/engineering_cuda_20260917/`。

引用已完成的 `outputs/grouped_robot_independent/engineering_cuda_20260916/`：seed 0、train 2,048／validation 1,024、label 1 epoch。该来源与新 Sequential 的训练预算一致。

训练 batch 1,024、评价 batch 2,048；先在第 1 次更新后暂停，再用 tmux `--resume` 恢复，最终完成：

- 新增 Sequential 训练 1 个，Adam 更新 2 次。
- 评价条件 10 个：4 个新增、6 个只读复用。
- `training_control=measured`；CUDA 活跃参数 112；前半段梯度为 `None`，后半段梯度有限且非零。
- 后半段训练峰值已分配显存约 0.119 GiB。
- `test_read=false`、`test_evaluated=false`。
- 生成完整 checkpoint、原始概率、CSV、Markdown、PNG/PDF 和结果锁。

暂停和完成后，两个工程 tmux 会话均自动退出，日志保留。

小型运行只验证实现和执行链路，不是正式性能结果。正式五种子目录尚未启动；正式运行新增五个 300-epoch Sequential 后半段，共 27,000 次 Adam 更新。
