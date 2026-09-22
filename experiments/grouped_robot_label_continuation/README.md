# Robot：固定概念 300 轮，将 label 训练从 100 轮延长到 300 轮

直接检验当前“概念纠正后 label 准确率下降”是否与后半段训练预算有关。
前半段使用已经完成的 `long_concept` 模型，始终冻结；两份后半段分别恢复自己的
模型、完整 Adam、随机数状态和训练历史，新增 200 轮训练。

## 实验配对

| 配置 | 概念训练 | label 训练 | 来源 |
|---|---:|---:|---|
| 本次基准 | 固定 300 轮 | 已完成 100 轮 | 原续训实验的 `long_concept` |
| 本次续训 | 同一个 300 轮模型 | 从 100 轮续训至 300 轮 | 从上述两份后半段检查点接着训练 |

两份后半段分别为 Independent 和关闭五个条件 X 门的对照，顺序执行。
两边都保留中途测量及其对应的后五位量子态。Independent 使用真实概念控制 X 门训练，
正常评价使用测量记录控制 X 门；全部概念纠正只替换控制位，保留物理测量分支。

本实验的前半段不是原来只训练了 100 轮的模型，也不重新训练前半段。
不复用上一项实验 `long_label` 中的两个后半段；它们对应另一份前半段。

模型、前向、反向传播和 Adam 训练循环调用原来的 TorchQuantum/FusionModel 实现。
仍为 11 qubits、前段四层、40 个池化特征、每层 data re-uploading；后半段结构不变。
沿用原预处理缓存、训练集方差排列、数据划分、seed=0、lr=0.01、batch=1024、eval batch=2048。
原训练集 18432 张，每轮 18 次更新；本次两项训练共新增 **7200 次 Adam 更新**。

保留原 label 样本打乱规则：label 第 101 轮使用原先规则的第 201 个排列，
并继续到 label 第 300 轮。这里的排列偏移仍为原来的 100，不能因为概念训练过 300 轮就改成 300。

## 一条命令运行

在仓库根目录执行：

```bash
bash experiments/grouped_robot_label_continuation/scripts/launch.sh
```

程序自动核查来源，先运行 Independent 续训，再运行关闭 X 门的配对续训，最后完成评价和汇总。
使用 CUDA，一个工作进程，Rich 状态显示，tmux 完成后自动退出。

```bash
# 查看状态
bash experiments/grouped_robot_label_continuation/scripts/status.sh

# 进入 tmux
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_robot_label_continuation

# 实时日志
tail -f outputs/grouped_robot_label_continuation/robot_v5_l4_seed0/evaluation.log

# 中断后继续
bash experiments/grouped_robot_label_continuation/scripts/launch.sh --resume

# 只检查配置、原结果文件和 CUDA，展示计划
bash experiments/grouped_robot_label_continuation/scripts/run.sh --plan-only

# 完整核查原数据和模型，但先不训练；之后使用 --resume 继续
bash experiments/grouped_robot_label_continuation/scripts/launch.sh --preflight-only
```

默认来源：`outputs/grouped_robot_continuation/robot_v5_l4_seed0/long_concept/`。
`--reference` 接收其父目录，即完整原续训实验的根目录。
新结果全部保存在 `outputs/grouped_robot_label_continuation/robot_v5_l4_seed0/`。
原源码、原检查点及原结果不改动。

每个 epoch 末保存；轮内沿用原每 25 次更新保存的设置。
SIGINT/SIGTERM 在安全位置保存完整状态；续跑跳过已经完成且校验通过的任务与预测。
来源、配置、数据、运行环境或结果文件变化会拒绝混用。

## 观察指标

主要观察固定第 100 轮与第 300 轮的：

- 全部概念纠正后的 label 准确率，及相对原 100 轮的变化。
- 正常 label 准确率，以及纠正前后的准确率差。
- 相同控制条件下的训练集与验证集 BCE 和准确率。
- 有条件 X 门与关闭条件 X 门的配对差异。

Independent 每轮保留真实控制下的训练 BCE 和正常测量控制下的验证指标；
另外在 label 第 150、200、250、300 轮记录全部真实控制下的验证准确率与 BCE。
第 100 轮的对应指标从原始预测复算。定期诊断只做推理，不参与梯度、早停或模型选择。

最终两种预算各评价八个验证条件：正常、分别纠正五个概念、全部纠正、关闭 X 门；
各有三个匹配控制条件的训练集诊断。合计 16 个验证条件、6 个训练诊断。
概念指标与原基准逐项对照，确认冻结前半段后没有变化。

训练损失使用真实控制，正常验证使用测量控制，不能直接将两者差值解释成过拟合。
BCE 改善与准确率提高分别报告，不把前者当作后者的替代证据。
所有负结果保留，不按本次曲线选择最优 epoch。
本实验仍为 seed 0 的验证集预算诊断，**不读取测试 CSV 或测试图片**。

## 输出文件

| 文件 | 内容 |
|---|---|
| `summary.md` / `summary.json` | 固定 100 / 300 轮的总对照 |
| `comparison.csv` | 正常、全概念纠正、关闭 X 的准确率与 BCE，以及各项变化 |
| `intervention_learning_curve.csv` | 正常逐轮指标与预定轮数的全部纠正指标 |
| `condition_results.csv` | 16 个验证条件 |
| `training_diagnostics.csv` | 6 个匹配控制条件的训练集评价 |
| `concept_results.csv` | 两种预算各五个概念的指标 |
| `training_budgets.csv` | 原始及最终 Adam 步数、新增更新和耗时 |
| `learning_curves.*` / `budget_comparison.*` | PNG、PDF、SVG 图 |
| `long_label/training/<cell>/imported.pt` | 完整续训起点 |
| `long_label/training/<cell>/endpoint.pt` | 固定 300 轮终点及完整 Adam |
| `<arm>/<role>/<cell>/<condition>/predictions.pt` | 最终评价的原始测量概率、label 概率质量及目标 |
| `reference_lock.json` / `result_lock.json` | 来源与结果完整性记录 |

## 工程检查

```bash
bash experiments/grouped_robot_label_continuation/scripts/check.sh
```

包括 Ruff、ty、Pylint 和 CUDA 测试：与连续训练的模型和完整 Adam 对照、
半轮中断恢复、诊断不改变优化轨迹、源前半段正确、冻结概念概率不变、
已完成结果复用、来源文件不变和异常配置拒绝。

小规模真实数据检查命令：

```bash
bash experiments/grouped_robot_label_continuation/scripts/launch.sh \
  --session robot_label_continue_dev \
  --reference outputs/grouped_robot_continuation/development_check \
  --out outputs/grouped_robot_label_continuation/development_check \
  --development --head-epochs 4 --diagnostic-every 1
```

开发模式只验证工程流程，不能用于判断正式训练收益。
