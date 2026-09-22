# 工程检查记录：2026-09-14

环境：`/root/miniforge3/envs/VQC/bin/python`，NVIDIA GeForce RTX 4090。

- Ruff check、Ruff format、ty、Pylint E/F/W 全部通过。
- CUDA/协议测试：**10 passed**，12.69 秒。
- 显式投影、分支归一化、实际条件 X 门与量子分类电路，和优化后的分支实现一致。
- 四条件分别从其联合 `(m,y)` 分布采样，与各自精确概率吻合，非法实际分支保留。
- 两种子、四条件的中断恢复与连续运行的逐图片输出完全一致；完成条件未重写。
- 参数冻结、负效应保留、配对计数和训练种子样本标准差检查通过。
- 源实验文件未改写；不同恢复配置和被篡改的预测文件均拒绝。
- 完整验证集 seed 0、5,479 张图片的原始概念概率和 Label 概率最大差值为 0；
  默认 256-shots 逐图片结果与源实验完全一致。
- 默认预检核验了 seeds 0–4 的已完成 Sequential 概念前段与反馈分类模型。

真实 tmux 生命周期检查目录：
`outputs/grouped_sequential_intervention/development_tmux_20260914/`。

使用现有已训练的 seed 0、1 模型，验证子集 18 张、batch 18、32 shots：

1. 首次 `--max-conditions 2`：完成 2/8 条件后状态为 `paused`，tmux 自动退出。
2. `--resume`：跳过原有两条件，完成全部 8/8 条件，状态为 `complete`，tmux 自动退出。
3. 结果含 CSV、JSON、逐图片 PT 和 PNG/PDF/SVG 两条纠正路径图。
4. 小子集报告和图均标记为工程验证，不用作论文实验结论。

质量检查日志：
`outputs/grouped_sequential_intervention/development_quality_20260914.log`。

仅做工程验证和完整 seed 0 基线复现，**未启动默认五种子完整纠正评价**。
正式结果目录 `outputs/grouped_sequential_intervention/dsprites_l4_five_seeds/`
将在用户运行默认启动命令时创建。
