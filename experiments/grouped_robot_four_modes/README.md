# Robot Standard / Joint 五种子实验

补齐 Robot 四种训练模式。新增 Standard、Joint 有反馈、Joint 无反馈，复用已完成的 Independent、Sequential 及其匹配的无反馈结果。所有旧代码、数据、模型和概率文件只读；本轮仅训练集／验证集，不读取测试集。

## 固定设置

| 项目 | 设置 |
|---|---|
| 数据 | 原 Robot v5，五个二元概念，train 18,432／validation 6,144 |
| 输入 | 原居中、Pool40、训练集拟合的方差路由；每个上传位四个数值 |
| 前半段 | 四层，每层复用 FusionModel/TorchQuantum 的 data re-uploading、U3、环形 CU3；240 个参数 |
| 后半段 | 五层原量子 label 电路，加读出位；112 个参数 |
| 初始化 | 前半段原 U[0,π]；后半段 Gaussian σ=0.1/(5×6)，最后读出 RY 初值加 π/2 |
| 测量 | 前五位测量，保留全部 32 个 Born 分支、后五位条件量子态和分支权重 |
| 种子 | 0、1、2、3、4；所有声明种子均报告 |
| 优化 | Adam lr=0.01，梯度裁剪 5；batch=1024，评价 batch=2048；必须 CUDA |
| 预算 | 每条新路线 600 epoch；固定最终模型，无最优 epoch／种子选择 |

三个新模型在同一种子下，使用相同的**原始**前后段参数、RNG、数据、逐 epoch 样本顺序及 Adam 预算。不会用已经训练的概念电路作为起点。

600 epoch 对应 Independent／Sequential 的概念 300 + label 300，总优化更新数一致：每模型 10,800 次，15 个新模型共 **162,000 次 Adam 更新**。这不表示每部分参数参与更新的次数或运行时间相同。Joint 的后半段梯度会通过精确分支概率传回前半段，训练时不缓存或 detach 前半段量子态。

## 三条新训练路线

| 路线 | 训练目标 | X 门控制 | 更新范围 |
|---|---|---|---|
| Standard | label BCE | 实际测量记录 | 前后两部分 |
| Joint | 真实五位联合概念码 NLL + label BCE，权重 1:1 | 实际测量记录 | 前后两部分 |
| Joint 无反馈 | 同样的 NLL + BCE，权重 1:1 | 全零，不触发 X | 前后两部分 |

概念 NLL 是 `-log p(c*)`；label BCE 使用全部分支加权得到的 label 概率。不会以联合 MAP 概念替代物理测量分布。无反馈仍保留中间测量，因此它消融的是**测量记录反馈**。

Standard 的中间位没有概念监督。其概念指标仅保存为诊断，主表概念列留空；不执行 Standard 的语义概念纠正。训练时记录的 Standard 概念 NLL 已 detach，绝不参与损失。

## 评价和比较

每个种子在 train、validation 各评价九个条件，共 **90 个条件**：40 个新增、50 个只读引用。

- 新增：Standard 正常；Joint 正常；Joint 全部五个概念纠正；Joint 无反馈。
- 引用：Independent 正常／全部纠正；Sequential 正常／全部纠正；已有冻结前半段的无反馈模型。

概念纠正只替换经典 X 控制记录；Born 权重、测量分支和反馈前条件量子态保留。不重编码、不重置后五位。Joint 正常和纠正条件的概念概率应一致；不同训练路线的前半段已经分别训练，不能要求概念概率跨模型一致。

报告四种模式的正常 label accuracy、balanced accuracy、BCE；监督模式的 mean-bit accuracy、五概念全对率、joint MAP accuracy、正确联合单次测量概率和 NLL。报告五种子均值及样本标准差（n−1）。Standard 诊断另存。

主要配对比较：

1. Joint 与 Standard，以及 Independent／Sequential 与 Standard 的正常分类差值。
2. Joint 有反馈 − **本轮重新联合训练的** Joint 无反馈。
3. Joint 全部纠正后 − Joint 正常预测。

同时汇总 Independent／Sequential 的既有纠正、反馈结果。先在每个种子内求差，再汇总；负收益原样保留。已有冻结前半段的无反馈模型不作为 Joint 的匹配消融对照。

## 运行

在仓库根目录执行一次即可顺序完成 15 个训练任务及全部汇总：

```bash
bash experiments/grouped_robot_four_modes/scripts/launch.sh
```

默认 tmux 会话 `grouped_robot_four_modes`。上方日志，下方 Rich 状态。完成、暂停或失败后自动退出 tmux，保留文件和日志。

```bash
# 查看计划（不训练、不创建输出）
bash experiments/grouped_robot_four_modes/scripts/run.sh --plan-only

# 查看进度
bash experiments/grouped_robot_four_modes/scripts/status.sh

# 进入 tmux；Ctrl-b 然后 d 离开查看，训练继续
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_robot_four_modes

# 跟随日志
tail -f outputs/grouped_robot_four_modes/robot_v5_l4_head5_ep600_five_seeds/train.log

# 任务已经停止时恢复；不要重复启动健康运行中的任务
bash experiments/grouped_robot_four_modes/scripts/launch.sh --resume

# 完整检查历史来源、数据、初始化后暂停；随后 --resume 才训练
bash experiments/grouped_robot_four_modes/scripts/run.sh --preflight-only
```

默认引用 `outputs/grouped_robot_sequential/robot_v5_l4_head5_five_seeds/`，它连接到已经完成的 Independent 五种子来源。默认新输出 `outputs/grouped_robot_four_modes/robot_v5_l4_head5_ep600_five_seeds/`。

每 25 次更新以及每个 epoch 结束保存完整模型、Adam、RNG、损失累计、样本顺序和 minibatch 位置。SIGINT／SIGTERM 请求安全暂停。`--max-steps N` 限制本次调用的**新增**优化更新数量。`--resume` 校验源代码、配置、环境、参考模型和数据；已完成模型及评价不重写。

工程测试可用 `--development --reference <completed-small-sequential-run> --seeds 0 --epochs N --out <new-output>`；N 必须等于该来源的概念 epoch 与 label epoch 总和，避免生成预算不一致的比较。工程结果明确标记，不用于正式性能结论。

## 结果文件

- `summary.md`、`summary.json`：完整结果和训练来源。
- `four_modes.csv`：四种模式的逐种子正常分类结果；Standard 概念列为空。
- `summary.csv`、`aggregates.csv`：九个条件的指标与均值／样本标准差。
- `paired_gains.csv`、`paired_summary.csv`：逐种子及汇总的配对收益。
- `standard_concept_diagnostics.csv`：未监督中间位的诊断，不作概念解释性证据。
- `learning_curves.csv`：训练过程；新路线同时记录概念 NLL 与 label BCE。
- `validation_results.png/.pdf`：准确率和逐种子收益图。
- `seed_N/training/{standard,joint,joint_no_feedback}/`：完整 checkpoint、历史和 CUDA 梯度证据。
- `seed_N/{train,validation}/.../predictions.pt`：新模型的样本身份、32 分支概念概率与 label 概率质量。
- `evaluation_reference.json`：旧结果的真实文件位置与哈希，不伪装成重新推理。
- `manifest.json`、`reference_lock.json`、`result_lock.json`：源代码、来源和产物完整性记录。

四种模式补齐后再进行有限 shots 和统一测试集评价。本轮验证集结果不表述为最终测试结果。

## 工程检查

```bash
bash experiments/grouped_robot_four_modes/scripts/check.sh
```

包含 Ruff、ty、Pylint，以及实际 CUDA 电路测试：三条路线损失和控制语义、前后段梯度、直接分支演化等价性、预算与初始化配对、概念纠正不改变原概念概率、原始结果只读、断点恢复与连续训练完全一致、来源／输出篡改拒绝及汇总统计核对。
