# Sequential：测量后概念记录纠正实验

目的：检查已经按 Sequential 训练好的模型，在预测时把概念记录纠正后，
最终 Label 是否变得更准确。所有模型参数冻结，不新增训练，也不改变训练模式。

代码、日志及结果与历史实验隔离。直接调用 `GroupedDynamicVQC`、`controls_for`
和 `sample_shots`；前段仍是 `FusionModel.TQLayer` 四层数据重上传及 U3/CU3 电路，
后段仍使用已训练的量子分类电路，无新增经典可训练参数。

## 固定协议

- 来源：`outputs/grouped_feedback_ablation/dsprites_l4_paired/` 中
  `sequential/seed{0,1,2,3,4}/feedback/endpoint.pt`，固定最终轮模型。
- 每种条件评价同一验证划分中的 5,479 张图片；不访问测试集。
- 5 个训练种子 × 4 种条件 = 20 个评价条件。
- 全程 CUDA，默认 eval batch 2048，float32/complex64，关闭 AMP/TF32。
- 每张图片同时计算精确分支混合概率和每条件 256 次联合 `(m,y)` 测量。
  有限 shots 是评价过程，不是带测量噪声训练。
- 保留全部 32 个实际测量分支，包括 14 个非法概念码；不后选择、不过滤非法码，
  不将概率重新归一化到 18 个合法组合，不使用 MAP 记录替代真实测量分支。

| 条件 | 传给条件 X 门的经典记录 |
|---|---|
| measured：不纠正 | 原始测量的 5 位记录 |
| shape：纠正 Shape | 前 2 位换成真实 Shape，后 3 位保留实际测量值 |
| scale：纠正 Scale | 前 2 位保留实际测量值，后 3 位换成真实 Scale |
| both：全部纠正 | 5 位全部换成真实 Shape、Scale |

例如测得 `01|100`、真实概念为 `10|011`，四个条件的控制记录分别是
`01|100`、`10|100`、`01|011`、`10|011`。
这里按 Shape、Scale **两个语义概念**纠正，不把五个编码位当作五个概念。

纠正位置在“中间测量完成”与“给后五位执行条件 X 门”之间。
每个实际测量结果 m 的 Born 权重和反馈前的 B 条件量子态均保留。
纠正只改变之后应用的 X 门，不把实际测量结果 m 改写为真实分支，不重新准备 B 状态。
精确评价为对各实际分支的 Label 概率加权求和，不跨分支相加振幅。

这测量的是本模型中**经典概念记录纠正**的效果；它不代表保留的 B 状态中的
全部图像信息都已被语义纠正。纠正后准确率可能下降，所有结果均保留。

## 一次启动

在仓库根目录运行：

```bash
# 只显示计划，不创建结果，不启动评价
experiments/grouped_sequential_intervention/scripts/run.sh --plan-only

# 一次启动五个种子的四条件评价
experiments/grouped_sequential_intervention/scripts/launch.sh

# 状态、连接和日志
experiments/grouped_sequential_intervention/scripts/status.sh
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_sequential_intervention
tail -f outputs/grouped_sequential_intervention/dsprites_l4_five_seeds/evaluation.log

# 中断后恢复
experiments/grouped_sequential_intervention/scripts/launch.sh --resume
```

tmux 在完成、暂停或失败后自动退出，日志和已完成结果保留。
恢复时核验已经完成的条件并跳过；尚未完整保存的条件按相同随机种子重新评价。
每条件的采样种子独立固定为 `训练 seed + 937 + 100003 × 条件序号`，
条件序号按表格为 0–3。改变 batch、shots 或样本子集需要新输出目录。

## 核验和结果

启动时只读核验源实验的数据、训练配置、初始化、样本顺序、训练预算、
冻结前段、最终参数及已保存评价。源实验代码、缓存、权重和结果文件不改写。
输出目录禁止与源实验及其历史来源重叠；并发 worker 由文件锁阻止。

每个种子的“不纠正”结果必须复现源实验的原始概率。
默认完整划分、相同 batch 和 256 shots 下，也核验有限 shots 的逐图片结果完全一致。
四个条件共用同一份前段量子态缓存，且每条件结束核验模型参数没有变化。

默认结果目录：`outputs/grouped_sequential_intervention/dsprites_l4_five_seeds/`。

- `summary.md/json`、`paired_results.csv`：逐种子四条件 Label accuracy、
  balanced accuracy、BCE、相对不纠正的配对差值，以及错→对、对→错数量；
  有限 shots 同样完整报告。
- `intervention_paths.png/pdf/svg`：不纠正→Shape→全部、不纠正→Scale→全部
  两条路径；分别展示精确概率和有限 shots，误差条为训练种子间的样本标准差。
- `seedN/{measured,shape,scale,both}/evaluation.json`、`predictions.pt`：
  每条件指标和逐图片概率、真实概念、Label、原始图片索引。
- `manifest.json`、`reference_lock.json`、各 `evaluation_lock.json`、
  `result_lock.json`：协议、源文件和结果哈希；`heartbeat.json` 为实时进度。

汇总只使用四条件全部完成的相同种子；部分完成状态会明确标注。
统计单位为训练种子，标准差使用样本标准差；单种子不估计标准差。
有限 shots 的配对计数是在同一批图片上比较，包含采样波动，不是同一条量子测量轨迹
的因果比较。一次固定采样不充分估计所有测量波动。

概念预测指标始终来自**原始测量分布**，不会把替换进去的真实概念算成预测准确率提高。
`summary.json.original_concept_metrics` 单独保存每个种子的原始概念指标。

## 工程检查

```bash
experiments/grouped_sequential_intervention/scripts/check.sh

# 独立小子集，验证流程；不能作为论文结果
experiments/grouped_sequential_intervention/scripts/launch.sh \
  --session sequential_intervention_smoke \
  --out outputs/grouped_sequential_intervention/manual_smoke \
  --seeds 0,1 --val-limit 18 --eval-batch-size 18 --shots 32 \
  --max-conditions 2

experiments/grouped_sequential_intervention/scripts/launch.sh \
  --session sequential_intervention_smoke_resume \
  --out outputs/grouped_sequential_intervention/manual_smoke --resume
```

`--max-conditions` 限制本次新完成的条件数量，不改变清单中的完整计划。
检查包括 Ruff、ty、Pylint 和 CUDA 测试：显式投影/归一化/条件 X 门对照，
四条件各自联合采样，非法实际分支保留，冻结参数，配对统计，恢复一致性，
源文件保护、结果篡改拒绝，以及完整验证集 seed 0 原始结果的复现。
