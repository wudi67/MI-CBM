# 开发验证记录：2026-09-14

新代码全部位于 `experiments/grouped_independent_feedback_ablation/`。
没有修改已有实验代码、模型、日志或心跳。

- Ruff 检查及格式检查、ty、Pylint E/F/W 均通过。
- pytest：5 passed，29.71 秒；一个来自原环境的 pynvml FutureWarning。
- 实际 CUDA 全验证集 seed 0：有反馈及无反馈的精确概率、256 shots 逐图片预测，
  重算与核验复用完全相同；正常 Independent 准确率为 70.7063%，没有误用真实概念纠正结果。
- CUDA 两种子 18 张开发子集：暂停在第一种条件时，统计中没有完整配对；
  恢复后四个条件全部完成，已完成预测文件没有改写，结果与不中断运行一致。
- 错配初始权重、变化的恢复配置、被篡改的预测文件均被拒绝；输出不得覆盖源实验。
- 实际 tmux 开发运行：1/4 条件暂停后自动退出；恢复完成 4/4 后再次自动退出。
  目录：`outputs/grouped_independent_feedback_ablation/development_tmux_20260914/`。
- 缺失来源的失败测试：记录明确的 failed 心跳与异常日志后，tmux 自动退出。
  目录：`outputs/grouped_independent_feedback_ablation/development_failure_20260914/`。

默认五种子报告已经通过 tmux 完整运行：
`outputs/grouped_independent_feedback_ablation/dsprites_l4_paired/`。
10/10 条件、5/5 配对完成，每模型 5,479 张验证图片，`test_evaluated=false`，
`engineering_subset=false`。这里只复用已完成训练的模型和预测，重新核验并汇总。

| 指标 | 均值 ± 样本标准差 |
|---|---:|
| 有反馈准确率 | 73.01% ± 3.06% |
| 无反馈准确率 | 66.99% ± 1.96% |
| 配对增益 | +6.03 ± 2.27 个百分点 |
| 256 shots 有反馈准确率 | 71.78% ± 1.70% |
| 256 shots 无反馈准确率 | 66.26% ± 1.48% |
| 256 shots 配对增益 | +5.53 ± 1.48 个百分点 |

精确概率、有限测量都是五个种子全部正收益。训练和评价来源、全部逐图片预测及报告锁
已核验；额外保存的 228 个源文件快照（含已有心跳和日志）也均未变化。
已检查最终 PNG 图；PDF/SVG 同时生成。完整运行后 tmux 已退出，GPU 空闲。
