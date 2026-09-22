# Independent 五种子训练与概念干预

本实验检查：后半段分类电路在训练时使用真实概念，是否比使用预测概念的
Sequential 更能从概念纠正中受益，以及不纠正时的分类准确率是否受到影响。

这里的 Independent 按本项目约定定义：**真实概念控制记录＋保留的后五位量子态**。
训练仍对前五位实施测量，保留实际分支及其 Born 权重；只把条件 X 门使用的记录
换成真实 Shape/Scale。前段冻结，只训练后半段量子分类电路。

## 默认协议

| 项目 | 设置 |
|---|---|
| 种子 | 0、1、2、3、4，与已有 Sequential 配对 |
| 概念前段 | 已训练 100 轮的各自 L4 前段，240 参数全部冻结 |
| 后半段 | 原 B5＋读出位，L1，24 个可训练量子参数 |
| 初始化 | 对应种子的概念训练终点内，尚未训练的后半段参数 |
| 优化 | 全新 Adam，lr 0.01，batch 1024，梯度裁剪 5 |
| 预算 | 后半段 100 轮、每轮 25 次更新，共 2,500 次 |
| 训练数据 | 同一缓存的 25,593 张图片 |
| 验证数据 | 同一缓存的 5,479 张图片 |
| 顺序 | 对应 Sequential 分类阶段第 101–200 轮的相同图片顺序 |
| 端点 | 固定最终轮，不使用验证最高分挑选模型 |
| 数值 | 强制 CUDA，float32/complex64，关闭 AMP/TF32 |
| 评价 | 精确概率＋每图片每条件 256 次联合 `(m,y)` 采样 |

前段继续复用 `FusionModel.TQLayer`、TorchQuantum 和 `GroupedDynamicVQC`，
保持每层每位四值上传、U3＋相邻环形 CU3；不改变居中、Pool(10,4) 和训练集方差路由。
代码和输出独立，不改写旧源码、预处理、权重或报告。

所有 32 个实际测量分支保留，包括非法概念码。训练的 Label BCE 作用于
分支概率混合后的输出；不跨分支相加振幅，不使用 MAP 概念替代测量分支。

## 已有结果复用

训练模型来自 `outputs/grouped_feedback_ablation/dsprites_l4_paired/`。
程序核验完成端点、冻结前段、初始化、数据、样本顺序、预算及 CUDA 环境。

- 默认核验并复用 `outputs/grouped_control_diagnostics/dsprites_l4_seed0/head_true/`
  的 seed 0 Independent 最终模型。复用的是完整训练端点，不继续训练旧模型。
- 其他四个种子从各自尚未训练的后半段初始化出发训练；不从已训练的 Sequential
  后半段微调，不继承概念训练阶段的 Adam。
- 默认核验并复制 `outputs/grouped_sequential_intervention/dsprites_l4_five_seeds/`
  中已经完成的 Sequential 四条件评价。
- Independent 的四条件评价全部使用本次统一接口重新计算；不会误用旧诊断中只对
  measured 条件采样的有限 shots 指标。
- `--retrain-reference` 可强制重新训练 Independent seed 0。
- Sequential 评价的 batch、shots 或子集不同，或者没有提供可用的已有评价目录时，
  在本次输出内重新评价相同冻结模型。源码或来源哈希损坏时明确拒绝复用。

默认新增 **4 个后半段训练＋20 个 Independent 评价条件**；另复用
20 个 Sequential 评价条件。汇总覆盖两种训练模式、五个种子、四个纠正条件。

## 四种评价条件

| 条件 | 控制 X 门的经典记录 |
|---|---|
| 不纠正 measured | 原始测量记录 |
| 纠正 Shape | 前两位替换为真实 Shape，后三位保留原测量值 |
| 纠正 Scale | 前两位保留原测量值，后三位替换为真实 Scale |
| 全部纠正 both | 五位全部替换为真实 Shape/Scale |

Independent 仅在**训练时**使用真实概念；正常“不纠正”评价仍用预测概念。
全部纠正后的分数需要真实概念帮助，不能当作无人帮助时的分类准确率。

纠正单位为 Shape、Scale 两个概念，不是五个编码位。纠正保留原来的测量分支、
Born 权重和反馈前 B 状态，不对 B 重新编码。两模式的原始概念概率应完全一致。

采样使用 `训练 seed + 937 + 100003 × 条件序号`，与已有 Sequential 干预一致。
每种条件从自身的联合 `(原测量 m, Label y)` 分布采样；固定一次采样并不穷尽
所有测量波动，不代表带采样噪声训练。

## 一次启动、查看、恢复

在仓库根目录执行：

```bash
experiments/grouped_independent_intervention/scripts/run.sh --plan-only

experiments/grouped_independent_intervention/scripts/launch.sh

experiments/grouped_independent_intervention/scripts/status.sh
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_independent_intervention
tail -f outputs/grouped_independent_intervention/dsprites_l4_five_seeds/train.log

# 中断后恢复
experiments/grouped_independent_intervention/scripts/launch.sh --resume
```

完成、暂停或失败后 tmux 自动退出。训练可在轮内恢复 Adam、随机状态、样本排列、
轮内位置和损失累计；已完成训练及评价核验后跳过。未保存完整的评价条件重新计算。
单个输出目录使用排他文件锁，恢复要求配置、源码、运行环境和锁定的来源一致。

`--max-steps 1` 在首个尚未完成的训练任务累计达到一步后暂停。
`--max-seeds 1` 在本次新完成一个种子的两模式四条件对照后暂停。
两项仅控制执行暂停，不改变已锁定的完整实验预算。

## 结果

默认目录：`outputs/grouped_independent_intervention/dsprites_l4_five_seeds/`。

- `summary.md/json`：两种训练模式、四种纠正条件的均值和样本标准差；
  包括绝对准确率、各自干预增益和两种模式之间的配对差值。
- `condition_results.csv`：逐种子、逐条件 accuracy、balanced accuracy、BCE，
  错→对、对→错、保持正确、保持错误数量，以及全部有限 shots 指标。
- `paired_results.csv`：同种子 Independent−Sequential 的绝对准确率差，
  以及 `Independent(纠正后−纠正前) − Sequential(纠正后−纠正前)`。
- `mode_interventions.png/pdf/svg`：两种模式各自的 Shape 优先、Scale 优先纠正路径。
- `independent/seedN/`：最终模型、训练历史、初始化来源及新训练的 `resume.pt`、梯度检查。
- `{independent,sequential}/seedN/evaluation/{measured,shape,scale,both}/`：
  每条件指标、逐图片预测和哈希锁。
- `manifest.json`、`reference_lock.json`、各评价锁及 `result_lock.json`：来源和产物证明。

统计单位为配对训练种子；只汇总两种模式四条件均完成的同一组种子。
部分完成状态及所有正负结果均保留；单种子不估计标准差。
比较时同时看正常预测和纠正后的绝对准确率，避免较低起点造成的较大增幅误导结论。
前段概念指标只使用原始测量分布，不把真实概念替换算作预测改善。未评价测试集。

## 工程验证

```bash
experiments/grouped_independent_intervention/scripts/check.sh

# 小子集、短训练，仅检查 CUDA、tmux 和恢复，不作为科学对照结果
experiments/grouped_independent_intervention/scripts/launch.sh \
  --session independent_intervention_smoke \
  --out outputs/grouped_independent_intervention/manual_smoke \
  --development --seeds 0,1 --head-epochs 1 \
  --train-limit 36 --val-limit 18 --batch-size 18 --eval-batch-size 18 \
  --shots 32 --max-steps 1

experiments/grouped_independent_intervention/scripts/launch.sh \
  --session independent_intervention_smoke_resume \
  --out outputs/grouped_independent_intervention/manual_smoke --resume
```

正式模式强制与源 Sequential 匹配训练预算和超参；子集或不匹配预算需显式
`--development`，报告会标注工程检查及是否匹配训练预算。
短训练开发模式的 Sequential 仍来自既有最终模型，不将两者标为公平训练预算对照。
