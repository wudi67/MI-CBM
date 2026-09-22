# Robot：固定 VQC 的概念→label MLP 诊断

目的：同一批 VQC 概念预测结果，交给直接读取概念的 MLP 后，纠正概念是否仍降低 label 准确率。
独立目录；历史 VQC、数据预处理、量子测量和实验输出均不修改。

默认只读来源：`outputs/grouped_robot_label_continuation/robot_v5_l4_seed0/long_label`。
这对应概念训练 300 轮、label 训练 300 轮的 Independent 模型。
启动时沿来源链核对检查点、Adam、数据、源码、评估原始预测和结果锁。
首次准备时，用实际检查点在 CUDA 重新计算前半段概率，核对保存的概率；随后复用原始概率。

## 固定实验

| 项目 | 配置 |
|---|---|
| 概念顺序 | head_shape, body_shape, has_antennae, ears_shape, foot_shape |
| 输入 | 五个二元概念，未附加图像特征或保留量子态 |
| MLP | Linear(5,16) → Tanh → Linear(16,1)，113 参数 |
| 模式 | Independent：仅训练集真实概念 → label |
| 训练 | seed 0，Adam，lr 0.01，300 epoch，batch 1024，梯度裁剪 5 |
| 样本 | 与原实验相同的 train 18,432 / validation 6,144 |
| 样本顺序 | 复用原量子 label 训练的逐 epoch 排列：原 seed 0，原 offset 100 |
| 更新次数 | 300 × 18 = 5,400；固定末轮，无最佳 epoch/LR 选择 |
| 设备 | CUDA 必须可用；float32，关闭 AMP/TF32；无 CPU 训练回退 |
| 监测 | 每轮训练 BCE；每 25 轮及末轮监测 train/validation 的正常、纠正脚、纠正全部 |
| 最终指标 | accuracy、balanced accuracy、BCE、准确率变化、对→错/错→对样本数 |

本轮只用一个预先固定的配置定位问题，不进行参数量匹配或量子优势比较。
训练和初始化都从头进行，VQC 参数不参与优化。

## 32 个测量分支如何处理

记原 VQC 的概念测量概率为 `p(m|x)`，MLP 对二元概念记录 `m` 输出的 label=1 概率为 `f(m)`。

- 正常预测：`sum_m p(m|x) * f(m)`。
- 纠正脚：对每个 `m`，替换最后一位为真实脚概念，再计算 MLP；原 `p(m|x)` 保持不变。
- 全部纠正：每个分支都输入同一组真实概念，理论上得到 `f(c_true)`；程序检查浮点误差。

脚的 mask 为 1，全部为 31。既不改为联合 MAP，也不先平均概念数值或 logits。
没有对错误的概念测量分支做后选择；没有将 Born 概率改成纠正后的条件概率。
`direct_true` 单列验证直接使用真实概念时的学习能力；它应与全纠正等价。
所有 label 预测都使用概率 >= 0.5 的固定分类规则。

## 启动与恢复

在仓库根目录执行：

```bash
# 查看固定计划，不创建输出、不训练
experiments/grouped_robot_mlp_diagnostic/scripts/run.sh --plan-only

# 正式诊断预算，后台训练，完成后 tmux 自动退出
experiments/grouped_robot_mlp_diagnostic/scripts/launch.sh \
  --session robot_mlp_diag \
  --out outputs/grouped_robot_mlp_diagnostic/robot_v5_l4_seed0

# 状态、连接与日志
experiments/grouped_robot_mlp_diagnostic/scripts/status.sh
/root/miniforge3/envs/VQC/bin/tmux attach -t robot_mlp_diag
tail -f outputs/grouped_robot_mlp_diagnostic/robot_v5_l4_seed0/train.log

# 原 worker 已结束时，恢复；自动核验并跳过完成的训练
experiments/grouped_robot_mlp_diagnostic/scripts/launch.sh \
  --session robot_mlp_diag_resume \
  --out outputs/grouped_robot_mlp_diagnostic/robot_v5_l4_seed0 --resume
```

`--preflight-only` 在来源审计、CUDA 概念概率核对和初始化完成后暂停。
`--max-steps N` 在本次新增 N 次 MLP 更新后暂停；恢复时省略即可完成剩余预算。
支持 SIGINT/SIGTERM 安全保存，恢复包含完整 Adam、RNG、epoch 内样本排列/偏移与损失累计。
独占文件锁防止两个 worker 写同一输出。源码/配置/来源改变后拒绝恢复。

## 输出

- `summary.md/json/csv`：两种后半段在相同数据上的正常/纠正指标及解释边界。
- `diagnostic.png`：验证准确率与直接真实概念输入的 BCE 训练曲线。
- `train/`、`validation/`：原始 MLP 分支概率质量、32-code 概率表、逐样本变化索引、评估锁。
- `training/`：固定末轮检查点、完整恢复状态、训练历史与 CUDA 梯度证据。
- `reference_lock.json`、`concept_cache.pt`、`data_lock.json`、`frontend_cuda_check.json`：来源和冻结前端核查。
- `result_lock.json`：最终结果和所有关键产物的哈希；完成后恢复会重新计算 MLP 预测核验。

训练集上原 VQC 没有保存“仅纠正脚”的结果，因此汇总不制造这个数字。
这轮不读取测试样本、不做测试集评估。全部数据仍是开发诊断证据。

## 解释边界

MLP 直接读取五个概念；原量子后半段读取的是概念控制 X 门作用后的保留量子态。
两者的输入形式、信息通路、参数量（113 vs 24）和模型类型不同。
若 MLP 全纠正有效，说明可以继续定位当前量子概念接口及后半段的问题；不能单独认定 VQC 表达能力有缺陷。
局部纠正不保证单调提升；必须保留全部纠正的主要结果。

## 工程检查

```bash
experiments/grouped_robot_mlp_diagnostic/scripts/check.sh

# 可选：真实来源、缩短训练，独立工程输出
experiments/grouped_robot_mlp_diagnostic/scripts/launch.sh \
  --session robot_mlp_check \
  --out outputs/grouped_robot_mlp_diagnostic/cuda_check \
  --development --epochs 2 --diagnostic-every 1 --max-steps 1

experiments/grouped_robot_mlp_diagnostic/scripts/launch.sh \
  --session robot_mlp_check_resume \
  --out outputs/grouped_robot_mlp_diagnostic/cuda_check --resume
```

检查包含 Ruff、ty、Pylint，以及真实 CUDA 上的训练输入隔离、全部编码/纠正 mask、
连续与中途恢复轨迹一致性、历史量子来源核查、历史文件不变和结果篡改拒绝。
`--development` 输出始终标注工程预算，不作为正式实验结论。
