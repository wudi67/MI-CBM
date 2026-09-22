# Robot：图片到概念的初始化对照

检验 H-EFT-VA 启发的小角度高斯初始化，能否改善现有四层 Fusion VQC 的概念预测。
代码、输出与此前后半段初始化实验隔离，旧代码和已有结果保持原样。

| 方法 | 四层 U3/CU3 的初值 |
|---|---|
| `uniform` | 原 `FusionModel.TQLayer` 的 `π × rand`，即 `[0, π]` 均匀初始化 |
| `eft_gaussian` | `N(0, σ²)`，`σ=0.1/(4×10)=0.0025` |

这是原有 240 个参数的初始化对照。每层的四数值上传、十个 U3、十个环形 CU3 全部复用。
前十个 qubit 都参与图片编码与纠缠，前五个 qubit 的测量分布预测五个二元概念。
`N=10` 使用参与前半段演化的 qubit 总数。没有新增门，没有使用后半段的 π/2 读出调整。

尺度来源见 [H-EFT-VA v2](https://arxiv.org/abs/2601.10479v2) 和
[固定版本作者代码](https://github.com/eyadiesa/H-EFT-VA/blob/893698185ea905ffd3d0c89c0470696dcea89dd5/src/main.py#L185)。
我们的电路包含 data re-uploading；本实验不是原文 ansatz 的复现，也不据此宣称避免贫瘠高原。

## 固定比较

- Robot 原 train 18,432 / validation 6,144；只读复用原居中、40 维池化、训练方差排列和上传角度缓存。
- 每种初始化三个编号 `0,1,2`，共六个单元，全部从随机初值训练 300 轮；不复用训练完成的概念权重。
- 原始 `uniform/init_0` 的初值必须与历史实验完全一致；其他编号使用同一个原均匀分布。
- 每个单元均用新的 Adam，lr=0.01，batch=1024，eval batch=2048，grad clip=5。
- 六个单元的训练图片顺序相同：shuffle seed=0、epoch=1…300。各编号仅改变前半段参数初值。
- 直接调用原概念训练循环，目标仍为真实五位记录的负对数 Born 概率。
- 后半段参数冻结且不参与损失；不训练 label，不新增概念纠正实验，不读取测试集。
- 全部六个单元各 5,400 次 Adam 更新，共 32,400 次；单个 CUDA 进程串行执行。

主指标是五个边缘阈值概念同时正确的比例。并列保留平均概念准确率、联合 MAP、单次测量真实记录概率、概念 NLL、逐概念准确率。
报告固定第 300 轮的均值、样本标准差和按编号配对差值，保留全部初始化；不挑最佳轮次或种子。
每轮保存验证曲线，末轮评价 train/validation，合计十二份原始预测。
三个初始化共享数据划分和训练顺序；这是初始化敏感性比较，不是三个独立数据划分的结论。

## 启动及恢复

```bash
cd /root/autodl-tmp/Quantum_CBM
bash experiments/grouped_robot_concept_init/scripts/launch.sh --session robot_concept_init
```

自动按编号执行 `uniform → eft_gaussian`。Rich 状态窗口、训练日志与断点自动保存；完成后 tmux 自动退出。

```bash
bash experiments/grouped_robot_concept_init/scripts/status.sh
/root/miniforge3/envs/VQC/bin/tmux attach -t robot_concept_init
tail -f outputs/grouped_robot_concept_init/robot_v5_l4/train.log
bash experiments/grouped_robot_concept_init/scripts/launch.sh --session robot_concept_init --resume
```

每 25 步及每轮保存完整 Adam、RNG 和当前 batch 位置；SIGINT/SIGTERM 在 batch 边界保存。
恢复核验来源、代码、数据、配置与已完成结果，跳过已结束的单元。
不需要改动之前的实验入口。

```bash
# 只核验计划和 CUDA，不训练
bash experiments/grouped_robot_concept_init/scripts/run.sh --plan-only

# Ruff、ty、Pylint，以及 CUDA 中断恢复/原训练循环一致性检查
bash experiments/grouped_robot_concept_init/scripts/check.sh

# 小规模真实数据工程验证；不能当成正式训练结果
bash experiments/grouped_robot_concept_init/scripts/launch.sh \
  --session robot_concept_init_check \
  --out outputs/grouped_robot_concept_init/development_check \
  --development --concept-epochs 2 --train-limit 64 --val-limit 64
```

默认结果目录：`outputs/grouped_robot_concept_init/robot_v5_l4/`。

- `summary.md/json`、`summary.csv`、`aggregates.csv`：固定末轮结果与配对差值。
- `per_concept.csv`、`learning_curves.csv/png`：逐概念指标和训练过程。
- `initial_diagnostics.json`：未训练时概念损失、逐层梯度；不代替最终指标。
- `<method>/init_<i>/training/concept/`：完整断点、固定末轮模型、历史和 CUDA 梯度检查。
- `<method>/init_<i>/<train|validation>/`：原始概念概率、指标和完整性记录。
- `manifest.json`、`reference_lock.json`、`data_reference.json`、`initialization_lock.json`、`result_lock.json`：可追溯的代码/输入/来源/结果。
