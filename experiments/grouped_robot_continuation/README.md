# Robot：分别延长概念训练与 label 训练

复用 `experiments/grouped_robot_pilot/` 的 TorchQuantum/FusionModel 电路、损失、
训练循环和概念纠正实现，独立保存所有续训结果。原代码、数据和 100+100 实验不改动。

## 固定实验内容

默认来源：`outputs/grouped_robot_pilot/robot_v5_l4_seed0/`，必须是已经完整结束的 P0。

| 配置 | 概念训练 | 两份后半段各自的 label 训练 | 初始化与优化器 |
|---|---:|---:|---|
| baseline | 100 | 100 | 只读取原结果和模型 |
| long_concept | 100→300 | 重新训练 100 | 前段恢复原 Adam；两个后半段复用原始初始参数，各建新 Adam |
| long_label | 保留原第 100 轮 | 100→300 | 两个后半段分别恢复各自的模型和完整 Adam |

两个后半段分别是：Independent（训练时真实概念控制 X 门，正常评价时测量记录控制 X 门），
以及关闭五个条件 X 门的对照。两边都保留中途测量及后五个 qubit 的条件量子态。

结构不变：11 qubits，前十位参与上传和四层 Fusion VQC，每层重新上传 40 个值，
前五位测量，后五位加一个新预测位进行一层 label 演化，共 264 个参数。
复用原池化、居中和训练集方差排列，不重新拟合输入；lr=0.01、batch=1024、eval batch=2048、seed=0。

默认新增五项训练任务：

| 任务 | 新增 epochs | 新增 Adam updates |
|---|---:|---:|
| long_concept / concept | 200 | 3600 |
| long_concept / independent | 100 | 1800 |
| long_concept / no_feedback | 100 | 1800 |
| long_label / independent | 200 | 3600 |
| long_label / no_feedback | 200 | 3600 |
| 合计 | — | 14400 |

重要的配对约束：

- long_concept 的后半段第 1–100 轮图片顺序，与原后半段第 1–100 轮完全对应；
  不因为前半段变成 300 轮而改成另一组打乱顺序。
- long_label 始终使用原来的第 100 轮前半段，不能使用 long_concept 的第 300 轮前半段。
- 真正续训的三个任务保留原模型、完整 Adam、步数、历史及随机数状态。
- label-only 续训后的概念概率应保持不变，评价时自动检查。
- 原实验的来源、数据、检查点和原始预测均校验哈希。任何变动都会拒绝混用。

## 一次启动，两项实验顺序完成

在仓库根目录执行：

```bash
bash experiments/grouped_robot_continuation/scripts/launch.sh
```

自动核验原实验、复用基准验证结果并计算训练集诊断，然后运行 long_concept 三个训练任务及评价，
再运行 long_label 两个续训任务及评价，最后生成三组配置的对照表与曲线。
只有一个训练进程，不需要手动启动第二项实验。tmux 正常结束后自动退出。

```bash
# 查看状态
bash experiments/grouped_robot_continuation/scripts/status.sh

# 进入 tmux
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_robot_continuation

# 实时日志
tail -f outputs/grouped_robot_continuation/robot_v5_l4_seed0/evaluation.log

# 中断后继续两项实验
bash experiments/grouped_robot_continuation/scripts/launch.sh --resume
```

续跑只恢复未完成任务；已完成的训练、预测及汇总只校验，不重复执行。
SIGINT/SIGTERM 在安全的 batch 边界保存模型、Adam 和当前样本位置。
硬终止会从最近保存点恢复（沿用原每 25 次更新及每个 epoch 末保存的设置）。

```bash
# 只展示计划并检查 CUDA/原结果文件，不创建新输出
bash experiments/grouped_robot_continuation/scripts/run.sh --plan-only

# 只完整核查原数据与模型，不进行训练；结束后用 --resume 启动
bash experiments/grouped_robot_continuation/scripts/launch.sh --preflight-only
```

原 P0 的 `--resume` 会拒绝改变 epoch 预算，这是保护历史结果的设计。
本实验必须使用上面的新入口，不直接修改原配置或原检查点。

## 评价和解释

三组配置各八个验证条件：正常 Independent、分别纠正五个概念、全部纠正概念、关闭条件 X 门。
额外各做三个只推理的训练集诊断：Independent 测量控制、全部真实控制、对照零控制。
共 24 个验证条件和 9 个训练诊断；八个原基准验证条件直接复用，共 33 条结果。

主指标是正常 label 准确率、条件 X 门贡献、全部概念纠正后的准确率及变化。
同时报告五个概念各自的准确率、全部边缘阈值预测正确率、联合 MAP 和真实完整记录单次测量概率。
保留 BCE、概念 NLL、各概念纠正的正负结果，固定预算末尾评价，不挑中间最高准确率。

训练集与验证集的泛化差距要在相同控制条件下比较：measured 对 measured、
correct_all_five 对 correct_all_five、zero 对 zero。
训练日志的 Independent 训练 BCE 使用真实控制，正常验证 BCE 使用测量控制，不能直接把差值当作过拟合。

本实验仍是 seed 0 的验证集训练预算诊断。**不读取测试 CSV 或测试图片，不产生正式测试结论。**
long_concept 还会改变保留的 B 状态，因此 label 提升不自动证明全部来自概念准确率提升。
原设置延长到 300 轮仍无改善，也不能单独证明模型已训练到最优。

## 输出

默认输出：`outputs/grouped_robot_continuation/robot_v5_l4_seed0/`。

| 文件 | 内容 |
|---|---|
| `summary.md` / `summary.json` | 三组固定预算的总对照 |
| `comparison.csv` | 正常／关闭 X／全部纠正的 label 指标与相对基准变化 |
| `condition_results.csv` | 三组配置的 24 个验证条件 |
| `training_diagnostics.csv` | 九个训练集条件，与验证条件直接对应 |
| `concept_results.csv` | 三组配置的五个概念指标 |
| `training_budgets.csv` | 新增更新次数、起止 epoch、是否恢复 Adam、新增优化耗时 |
| `learning_curves.*` / `budget_comparison.*` | PNG、PDF、SVG 图 |
| `<arm>/training/<cell>/imported.pt` | 开始本任务时的模型与优化器快照 |
| `<arm>/training/<cell>/initialization.json` | 起点、优化器、随机数和样本顺序来源 |
| `<arm>/training/<cell>/endpoint.pt` | 固定终点模型与完整 Adam |
| `<arm>/<role>/<cell>/<condition>/predictions.pt` | 原始测量分支、label 概率质量和目标 |
| `reference_lock.json` / `result_lock.json` | 原实验引用及新结果的完整性记录 |

## 工程检查

```bash
bash experiments/grouped_robot_continuation/scripts/check.sh
```

包括 Ruff、ty、Pylint，以及 CUDA 上与原训练循环连续训练逐项对照的测试：
模型、完整 Adam、历史一致；半个 epoch 中断后精确恢复；后半段图片顺序一致；
错误样本排列、来源变更和结果篡改拒绝；已完成结果不重写；全过程不需要测试集文件。

小规模检查必须明确标记 `--development`，并引用已经完成的开发基准，例如：

```bash
bash experiments/grouped_robot_continuation/scripts/launch.sh \
  --session robot_continue_dev \
  --reference outputs/grouped_robot_pilot/development_check_final \
  --out outputs/grouped_robot_continuation/development_check \
  --development --concept-epochs 3 --head-epochs 3
```

开发结果不能替代默认 100→300 实验。
