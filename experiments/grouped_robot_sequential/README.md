# Robot Sequential 五种子对照

本实验复用已完成的 Robot Independent 五种子结果，只新增五个 Sequential 后半段模型。比较正常分类、有／无测量记录反馈，以及全部概念纠正。旧代码、模型、数据和结果均只读。

## 实验内容

| 部分 | 设置 |
|---|---|
| 概念电路 | 复用每个种子已训练 300 epoch 的四层电路，冻结全部 240 个前半段参数 |
| 数据上传 | 沿用 FusionModel/TorchQuantum，每层 data re-uploading、U3 和环形 CU3 |
| 数据 | 原 Robot v5、五个二元概念、原居中／40 维 Pool／训练集方差路由 |
| label 电路 | 五层、112 个参数；后五位保留量子态，加一个新读出位 |
| 后半段初值 | 与对应 Independent／无反馈模型相同，`σ=0.1/(5×6)` 高斯，读出 RY 初值加 `π/2` |
| Sequential 训练 | 前五位实际测量记录决定后五位 X 门；保留各分支对应的条件量子态 |
| 训练预算 | 每个种子后半段 300 epoch，重新创建 Adam，不继承已经训练的 label 参数或 Adam |
| 优化与批次 | Adam `lr=0.01`、梯度裁剪 `5`、训练 batch `1024`、评价 batch `2048` |
| 种子 | `0,1,2,3,4`；与对应已有模型配对 |
| 模型选择 | 固定第 300 epoch，所有五个种子均纳入汇总 |
| 当前数据划分 | train 18,432／validation 6,144；不读取 test |

同一种子的前半段、后半段初值、RNG、训练数据、样本顺序、优化器和更新次数与 Independent／无反馈模型一致。区别在后半段训练时的 X 控制：Independent 使用真实概念，Sequential 使用测量记录，无反馈使用全零控制。

正式运行新增 **5 次 label 训练、27,000 次 Adam 更新**。概念电路和已有对照模型无需重训。

每个种子评价五个条件：

1. Sequential 正常预测。
2. 同一个 Sequential 模型全部概念纠正后预测。
3. Independent 正常预测，复用已有结果。
4. Independent 全部概念纠正后预测，复用已有结果。
5. 匹配的无反馈模型，复用已有结果。

在 train、validation 分别评价，共 **50 个条件**：20 个新 Sequential 评价、30 个已有评价引用。复用结果保存明确的文件路径和哈希，不伪装成本轮重新训练的模型。

## 测量、损失与纠正

- 继续使用实际物理测量的全部 32 个 Born 分支；后五位的条件量子态保留。
- Sequential 在每个分支上使用该分支自己的五位测量记录控制 X 门。不会使用联合 MAP 替换测量分布，也不会用真实概念替换训练控制。
- 对所有分支得到的 label 概率作 Born 加权，然后计算 label BCE，与已有框架保持一致。
- 全部概念纠正只将五个经典 X 控制记录改为真实概念，不改变 Born 权重，不重置或重新编码后五位量子态。
- 无反馈模型也保留测量；它用于考察**测量记录反馈的贡献**。
- 前半段始终冻结，因此五个条件下的概念预测应一致，代码会自动检查。

`training.py` 复用原 TorchQuantum 模型、原 `step/run` 更新循环、CUDA 梯度检查及断点恢复校验；增加 Sequential 的入口和明确的 `training_control=measured` 身份记录。没有修改旧训练器支持的模式列表。

## 运行

在仓库根目录执行：

```bash
bash experiments/grouped_robot_sequential/scripts/launch.sh
```

默认 tmux 会话为 `grouped_robot_sequential`，上方显示日志，下方显示 Rich 进度。任务完成、暂停或失败后自动退出 tmux；日志和所有结果保留。

```bash
# 查看进度
bash experiments/grouped_robot_sequential/scripts/status.sh

# 进入 tmux；Ctrl-b 然后 d 可退出查看，训练继续
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_robot_sequential

# 跟随日志
tail -f outputs/grouped_robot_sequential/robot_v5_l4_head5_five_seeds/train.log

# 原任务停止后恢复，不要重复启动仍在运行的任务
bash experiments/grouped_robot_sequential/scripts/launch.sh --resume

# 只查看计划，不创建结果或训练
bash experiments/grouped_robot_sequential/scripts/run.sh --plan-only

# 完整核查已有模型和输入后暂停，随后可以 --resume 训练
bash experiments/grouped_robot_sequential/scripts/run.sh --preflight-only
```

默认读取：`outputs/grouped_robot_independent/robot_v5_l4_head5_five_seeds/`。

默认输出：`outputs/grouped_robot_sequential/robot_v5_l4_head5_five_seeds/`。

训练必须使用 CUDA；没有 CUDA 时直接报错。环境由脚本自动设置为 `/root/miniforge3/envs/VQC`。程序每 25 次更新及每个 epoch 末保存断点；`--resume` 恢复模型、完整 Adam、RNG、样本排列和 minibatch 位置，已完成任务和评价不会重写。

`--max-steps N` 表示本次调用新增最多 N 次更新，到达预算保存并暂停。SIGINT／SIGTERM 同样请求安全暂停。`--development` 可使用已完成的小型 Independent 工程参考或种子子集，但后半段 epoch 预算仍必须与所引用对照一致，避免产生不匹配的比较。

## 输出

- `summary.md`：五种子验证集主表和配对收益。
- `summary.json`、`summary.csv`：每个种子、每个条件的 label 和概念指标，以及新／复用结果的来源。
- `aggregates.csv`：每个条件的均值、样本标准差（`n-1`）。
- `paired_gains.csv`、`paired_summary.csv`：逐种子差值及汇总。
- `learning_curves.csv`：Sequential、Independent、无反馈三种后半段的学习曲线，标明复用来源。
- `validation_results.png/.pdf`：准确率及逐种子配对收益图。
- `seed_N/training/sequential/`：模型、完整 Adam/RNG 断点、CUDA 梯度检查和训练记录。
- `seed_N/{train,validation}/sequential/.../predictions.pt`：样本身份、概念真值、label、32 分支概率及分支 label 概率质量。
- 其余条件的 `evaluation_reference.json`：已有模型评价的绝对路径和哈希。
- `manifest.json`、`reference_lock.json`、`result_lock.json`：配置、源文件、引用与输出完整性记录。

主要比较：

```text
训练模式差异 = Sequential 正常准确率 − Independent 正常准确率
Sequential 反馈收益 = Sequential 正常准确率 − 无反馈准确率
Sequential 纠正收益 = Sequential 全部纠正后准确率 − Sequential 正常准确率
```

同时保留已有 Independent 的反馈收益、纠正收益，报告 accuracy、balanced accuracy、BCE。先在每个种子内求差，再汇总，负收益同样保留。`bce_change` 为左侧条件减右侧条件，负数表示 BCE 改善。

Independent 训练使用真实控制，而 Sequential 训练使用测量记录，所以其训练 BCE 对应不同控制条件；模式主比较应使用相同评价条件下的结果。所有本轮数字均为 train／validation，后续再统一做有限 shots 和测试集评价。

## 验证

```bash
bash experiments/grouped_robot_sequential/scripts/check.sh
```

包括 Ruff、ty、Pylint，以及实际 CUDA 测试：显式测量分支公式与训练更新相同、改变真实概念不影响 Sequential 训练更新、前半段冻结、配对一致、断点恢复与连续训练一致、源结果只读、配置／文件／样本顺序变更拒绝。
