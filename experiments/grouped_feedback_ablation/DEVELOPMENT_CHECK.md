# 实现检查记录（2026-09-14）

环境：`/root/miniforge3/envs/VQC/bin/python`，NVIDIA GeForce RTX 4090。
本记录是工程检查，不是新 Joint 或多种子完整预算结果。

- `scripts/check.sh`：Ruff 检查及格式、ty、Pylint E/F/W 均通过；
  16 项 pytest 通过，最终一次耗时 29.27 秒。
- 物理检查：有／无反馈的分支计算与直接量子电路演化及梯度一致；
  无反馈有限采样使用零控制，保留包括非法概念码在内的全部测量记录。
- 训练检查：新 Joint 有反馈的一个 Adam 步与旧训练器的损失、参数、Adam 状态完全相同。
  Joint 两个方案、概念阶段、Sequential 两个后段共五条路线均检查轮内恢复，
  恢复与连续训练的参数、Adam、训练进度及历史完全相同。
- 实际完整缓存的 batch=1024 CUDA 反向传播通过；前段 240、后段 24 参数，
  Joint 两部分梯度均为有限非零值；Sequential 冻结前段无梯度。
- 默认配置核验了四个可复用历史端点及完整 train 25,593 / val 5,479 缓存。
  初始化、样本顺序、前段冻结和来源锁检查通过；篡改端点／初始化／预测文件会被拒绝。
- 两种训练种子的短流程检查通过：种子间初始化不同，种子内数据、初始化和顺序相同，
  Sequential 对内概念分布完全相同，Joint 与 Sequential 分别汇总。

另通过真实 tmux 启动器执行了三次衔接：

1. Joint 第一步后暂停，确认保存 `global_step=1`、`offset=18`，tmux 自动退出。
2. 从该检查点恢复，仅完成 Joint；全套报告为 partial，tmux 自动退出。
3. 再以 `--resume --stage all` 继续，校验并跳过 Joint，完成两个 Sequential 种子，
   八个任务全部完成，tmux 自动退出。

此短跑仅使用 train=36 / val=18，各阶段 1 轮，32 shots；
`summary.json.engineering_subset=true`，不能作为论文数据。
产物：`outputs/grouped_feedback_ablation/development_tmux_20260914/`。
质量检查日志：`outputs/grouped_feedback_ablation/development_quality_20260914.log`。

完整协议输出目录 `outputs/grouped_feedback_ablation/dsprites_l4_paired/` 未启动训练。
旧模型、训练器、历史输出及来源锁均未改写。
