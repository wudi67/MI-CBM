# 工程验证：2026-09-14

环境：`/root/miniforge3/envs/VQC/bin/python`，NVIDIA GeForce RTX 4090。

- Ruff check、Ruff format、ty、Pylint E/F/W 全部通过。
- 测试：**6 passed**，22.38 秒。
- 新训练路径的真实概念控制、Label BCE 和一步参数更新与旧真实概念训练实现一致，
  参数数值按 `atol=rtol=2e-6` 核验。
- 冻结前段无梯度，24 个量子分类参数通过 CUDA 更新；实际 batch 1024 训练步通过。
- 轮内暂停后恢复，模型参数、Adam 状态和训练进度与连续执行完全一致。
- 两种子、两种模式、四条件的完整开发流程及评价恢复通过；逐图片原始概念概率一致。
- 已完成预测文件保持不变，源实验文件未改写；预测/权重损坏及不同恢复配置均被拒绝。
- 默认旧 seed 0 的数据、初始化、冻结前段、100 轮/2,500 步预算、图片顺序、
  Adam 更新次数及权重哈希通过核验，可复用。
- 完整验证集 seed 0 复现：不纠正 70.7063%，全部纠正 93.9040%。
  四条件各自进行 256 次联合采样；使用概率误差与二项采样方差检查采样语义，
  不假定有限 shots 的阈值准确率必须非常接近精确准确率。

默认配置预检：
`outputs/grouped_independent_intervention/development_preflight_20260914.json`。

- 核验全部五个种子的来源，训练 25,593、验证 5,479 张。
- 训练预算与 Sequential 配对。
- 复用 Independent seed 0，新增四个后半段训练。
- 复用已完成的 Sequential 四条件评价。

真实 tmux 检查目录：
`outputs/grouped_independent_intervention/development_tmux_20260914/`。

使用现有概念前段、seeds 0/1、训练子集 36、验证子集 18、后半段 1 轮、
batch 18、32 shots，依次验证：

1. `--max-steps 1`：第一个种子训练一步、轮内 offset 18 后暂停，tmux 自动退出。
2. `--resume --max-seeds 1`：继续完成 seed 0 及两模式八条件评价，状态 partial/paused，
   tmux 自动退出。
3. `--resume`：完成两个训练任务和全部 16 个评价条件，状态 complete，tmux 自动退出。

开发结果明确标记为工程检查和非配对训练预算，不用作论文实验结论。
质量日志：`outputs/grouped_independent_intervention/development_quality_20260914.log`。

仅完成工程验证及历史 seed 0 复现，**未启动默认五种子完整实验**。
