# Robot Independent 五种子主实验（P0）

本实验补齐固定架构下的五个完整训练种子，比较正常 label 预测、有／无测量记录反馈，以及全部概念纠正的收益。代码和输出单独存放，不修改已有实验或模型。

## 固定协议

| 项目 | 设置 |
|---|---|
| 数据 | Robot v5，五个二元概念，32 种概念组合 |
| 数据处理 | 复用原 pilot 的居中、40 维 pool 和仅用训练集拟合的方差路由 |
| 概念电路 | 10 个输入 qubit，4 层；每层复用 FusionModel/TorchQuantum 的 data re-uploading、U3 和环形 CU3 |
| 概念初始化 | 原来的 `U[0, π]`；240 个参数 |
| 中间测量 | 测量前五位；保留后五位的条件量子态与全部 32 个 Born 分支 |
| label 电路 | 后五位加入一个新读出位，沿用现有结构，5 层、112 个参数 |
| label 初始化 | 高斯 `σ = 0.1/(5×6) = 0.0033333333`；最后读出 RY 初值加 `π/2`，仍参与训练 |
| 总规模 | 11 个物理 qubit，352 个参数；没有新增经典读出参数 |
| 种子 | `0,1,2,3,4`；每个种子改变前半段、后半段初始化以及训练样本排列 |
| 训练预算 | 概念 300 epoch；随后冻结概念电路，两条 label 路线分别训练 300 epoch |
| 优化器 | 每个训练任务重新创建 Adam，学习率 `0.01`，梯度裁剪 `5` |
| batch | 训练 `1024`，评价 `2048`，必须使用 CUDA，无 CPU 回退 |
| 模型选择 | 固定最后一个 epoch；五个种子全部保留 |
| 当前划分 | train 18,432、validation 6,144；本轮不读取测试集 |

前处理、损失、量子电路、测量反馈和训练更新直接复用已有代码；新包负责种子管理、公平配对、历史复用核查、恢复和汇总。

## 每个种子执行的任务

1. 从均匀初始化训练概念电路，监督目标为真实五位概念记录的联合 NLL。
2. 冻结同一个概念电路，训练两条 label 路线：
   - **Independent 有反馈**：训练时真实概念决定五个 X 门。
   - **无反馈**：仍保留中间测量和后五位量子态，但五个 X 控制全部置零。
3. 在相同的训练集、验证集分别评价三个条件：
   - `independent/measured`：正常预测，由实际测量记录控制 X 门。
   - `independent/correct_all_five`：同一个已训练模型，将五个控制记录全部换成真实概念。
   - `no_feedback/zero`：单独训练的匹配无反馈模型，X 控制全部为零。

同一种子的两条 label 路线使用完全相同的前半段、后半段初值、随机数状态、样本顺序和更新次数。它们各自重新训练后半段。无反馈对照不是仅在已训练有反馈模型上关闭 X 门。

纠正只替换经典控制记录，不改变原始 Born 权重或保留量子态，不重新编码真实概念，不对非法结果做筛选。此消融考察**测量结果反馈的贡献**，两条路线都包含测量。

正常 label 推断使用全部测量分支的精确加权结果，不把概念联合 MAP 记录直接代入整个分类电路。有限 shots 实验后续再做。

## 旧结果如何复用

默认复用两个符合协议的 seed 0 终点：

- 概念：`outputs/grouped_robot_concept_init/robot_v5_l4/uniform/init_0/training/concept/endpoint.pt`。
- Independent：`outputs/grouped_robot_label_init/robot_v5_l4_head5/eft_readout/init_0/training/independent/endpoint.pt`。

旧概念模型的未训练后半段是一层，本实验**只读取它已经训练的前半段**；正式分类使用五层 label 电路。原 checkpoint 保持原样，以 `training_reference.json` 明确记录复用范围。

复用前检查源文件和结果哈希、初始化、RNG、训练预算、Adam、每个 epoch 的样本排列、样本身份和完整量子态振幅。出现不一致会报错，不会跳过核查或悄悄混用。旧的其他初始化结果共享同一个前半段或同一个样本排列，不充作新的完整种子。

为了匹配旧 seed 0 的 label 顺序，label 的排列编号仍使用 `101…400`。`head_order_offset=100` **只影响排列生成编号**，不减少概念训练的 300 个 epoch。

因此默认总计 15 个逻辑训练任务，其中 2 个复用、13 个新训练；新训练 70,200 次 Adam 更新。可使用新的输出目录和 `--fresh-all` 从头训练全部 15 个任务。

## 执行

在仓库根目录运行：

```bash
bash experiments/grouped_robot_independent/scripts/launch.sh
```

默认 tmux 会话 `grouped_robot_independent`，自动显示训练日志和 Rich 状态。完成或暂停后两个窗口自动关闭；日志、模型与结果保留。

```bash
# 查看状态
bash experiments/grouped_robot_independent/scripts/status.sh

# 进入 tmux；Ctrl-b 然后 d 可退出查看，训练继续
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_robot_independent

# 跟随日志
tail -f outputs/grouped_robot_independent/robot_v5_l4_head5_five_seeds/train.log

# 中断后恢复；活跃任务无需重启
bash experiments/grouped_robot_independent/scripts/launch.sh --resume

# 仅打印计划，不创建实验、不训练
bash experiments/grouped_robot_independent/scripts/run.sh --plan-only

# 核查实际数据与旧模型后暂停；之后用 --resume 继续
bash experiments/grouped_robot_independent/scripts/run.sh --preflight-only
```

默认输出目录：`outputs/grouped_robot_independent/robot_v5_l4_head5_five_seeds/`。

`--resume` 恢复模型、完整 Adam、随机数状态、当前 epoch 的样本排列和 minibatch 位置，并复用已完成任务与评价。配置、源文件、引用模型发生变化会拒绝混入原实验。程序每 25 次更新及每个 epoch 末保存断点。训练时 SIGINT/SIGTERM 会请求在更新边界保存并暂停。

`--max-steps N` 表示本次调用新增最多 N 次更新，用于工程验证；到达预算会保存后暂停。减小 epoch、改变种子集合或限制数据必须同时使用 `--development --fresh-all`，开发结果不计入正式五种子汇总。

## 输出及解释

- `summary.md`：可直接阅读的验证集主结果。
- `summary.json`、`summary.csv`：每个完整种子的概念与 label 指标，train/validation 分开。
- `aggregates.csv`：各条件五种子的均值、样本标准差（`n-1`）。
- `paired_gains.csv`：每个种子的反馈收益及全部纠正收益，单位百分点。
- `per_concept.csv`：概念预测指标，不追加单概念干预实验。
- `learning_curves.csv`：固定终点前的完整学习曲线，标明历史复用任务。
- `validation_results.png/.pdf`：三条件 label 准确率与逐种子配对收益图。
- `seed_N/training/...`：新训练模型、Adam/RNG 断点、CUDA 梯度检查、历史记录和复用来源。
- `seed_N/{train,validation}/.../predictions.pt`：样本身份、真值、32 个概念概率及各分支的 label 概率质量。
- `manifest.json`、`reference_lock.json`、`result_lock.json`：配置、源代码、引用输入与输出的完整性记录。

主要收益定义：

```text
反馈收益 = Independent 正常 label 准确率 − 无反馈 label 准确率
全部纠正收益 = Independent 全部概念纠正后准确率 − 同模型正常准确率
```

先对每个种子求差，再汇总五个配对差值的均值与标准差。收益为负时如实保留。

概念同时报告平均逐位准确率、五位全部正确率、联合 MAP 正确率、真实记录的单次测量概率和 NLL，名称明确区分。三个评价条件共享概念电路，其概念预测指标应相同。

Independent 的训练损失使用真实控制；每个 epoch 的正常验证使用预测控制。检查拟合情况时应比较 `summary.csv` 中训练集和验证集的同一条件，不能直接把真实控制的训练 BCE 与预测控制的验证 BCE 当成同一输入条件。

这是此前选定架构的五种子验证实验。架构选择已经使用 seed 0 验证结果，最终论文测试数据仍须在后续统一评估，不能将本轮 validation 写成 test。

## 检查

```bash
bash experiments/grouped_robot_independent/scripts/check.sh
```

运行 Ruff、ty、Pylint 和实际 CUDA 集成测试，包括与原训练循环一致、完整种子配对、纠正保留分支、断点恢复、历史来源不变和哈希篡改拒绝。
