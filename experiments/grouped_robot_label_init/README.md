# Robot：五层 VQC 后半段参数初始化对照

目的：在已确定的五层后半段电路上，检验借鉴 H-EFT-VA 的小角度初始化能否改善最终预测、全部概念纠正收益或训练稳定性。

所有组使用相同的 measurement-induced VQC、112 个后半段参数和冻结前半段。只改变后半段可训练门的初始参数，不增加 qubit 或门，不重新制备概念量子态。

| 方法 | 代码名称 | 初始参数 |
|---|---|---|
| 现有均匀初始化 | `uniform` | 原 A/B 深度实验中 L5 的初始参数，均匀分布在 `[-π, π]` |
| 小角度高斯 | `eft_gaussian` | 所有后半段参数服从 `N(0, σ²)` |
| 小角度高斯＋读出调整 | `eft_readout` | 与上一组使用完全相同的随机值，仅最后的读出 `RY` 增加 `π/2` |

默认 `σ = κ/(L×N) = 0.1/(5×6) = 0.00333333`。这里 `N=6` 是后半段五个保留位加一个预测位的数量；`L=5` 是后半段总层数。

这个尺度借鉴 [H-EFT-VA v2](https://arxiv.org/abs/2601.10479v2) 和 [作者代码](https://github.com/eyadiesa/H-EFT-VA/blob/893698185ea905ffd3d0c89c0470696dcea89dd5/src/main.py#L185)。读出角度的调整是针对本模型的适配。我们保持自己的电路结构，因此实验是初始化对照，不是复现论文电路，也不据此宣称本模型已被证明避免贫瘠高原。

## 固定条件与历史复用

- 数据为 Robot v5，使用原 train 18,432 张和 validation 6,144 张；不读取 test。
- 前半段是同一个已训练 300 epoch 的四层 FusionModel/TorchQuantum VQC，保持冻结。
- 后半段均为原有五层 RY/RZZ/RY/RXX 星形电路及最终读出门，112 参数。
- 三种方法各三个后半段初始化编号 `0,1,2`，共九个单元；它们不是九次完整模型训练，也不是三个独立前半段种子。
- Independent 模式：训练时用真实概念控制 X，保留原来的 32 个物理测量分支及其 B 量子态。
- 每单元训练 300 epoch，Adam lr=0.01，batch=1024，eval batch=2048，grad clip=5。
- 所有单元的样本排列一致，复用原有 shuffle seed=0 和 epoch offset=100。
- 两个高斯组使用相同的标准正态样本，CPU 随机生成器种子为初始化编号。三个方法在同一编号下恢复相同的训练 RNG 状态。
- 复用 `outputs/grouped_robot_label_depth/robot_v5_l4_seed0/L5/` 下已完成的三个均匀初始化模型。启动会核验完整引用链、代码指纹、初始权重、训练预算及保存的预测。原文件只读。
- 六个新增单元均从相应随机初值开始，使用新的 Adam；总计 32,400 次新更新。不存在从已训练权重继续优化的暖启动。
- 直接复用已核验的冻结前半段振幅缓存，在 CUDA 上计算后半段并反向传播。

## 检查与评估

训练前，对固定的第一个训练 batch 记录九个初始化的平均/min/max label 概率、初始 BCE、裁剪前梯度范数，以及是否会触发原有的梯度裁剪。这一步没有优化器更新，用于检查初始预测偏置，不能当作训练效果或贫瘠高原诊断。

每个训练单元在 train 和 validation 上只评估两个条件：正常预测、纠正全部五个概念，共 36 个条件。纠正时仅替换经典概念控制记录，保留原物理测量分支。新的比较不增加逐 concept 干预任务。

主要报告：

1. 正常 label 准确率、全部概念纠正后的 label 准确率。
2. 全部纠正收益，即上述两者之差，单位为百分点。
3. BCE、balanced accuracy、逐样本由错变对/由对变错数量。
4. 每个初始化和三个初始化的均值、样本标准差（ddof=1），以及按编号配对的“新方法减均匀初始化”差值。
5. 每轮训练损失与正常验证指标，每 50 轮和末轮的真实概念控制指标，以及训练时间。

使用固定第 300 轮，不根据验证结果挑选种子或最佳 epoch。初始 BCE 更低、梯度更大都不自动意味着训练后更好；主要结论应根据固定末轮的比较得出。曲线和损失用于辅助解释。

## 一次启动

```bash
cd /root/autodl-tmp/Quantum_CBM
bash experiments/grouped_robot_label_init/scripts/launch.sh --session robot_label_init
```

程序按初始化编号依次执行 `uniform → eft_gaussian → eft_readout`。三个 uniform 单元核验复用，其余六个自动串行训练。tmux 中包含日志与 Rich 状态界面，完成后自动退出。

查看进度：

```bash
bash experiments/grouped_robot_label_init/scripts/status.sh
```

接回界面，或在会话退出后恢复中断任务：

```bash
/root/miniforge3/envs/VQC/bin/tmux attach -t robot_label_init
bash experiments/grouped_robot_label_init/scripts/launch.sh --session robot_label_init --resume
```

恢复时核验配置、源代码和上游指纹，恢复完整 Adam、随机状态和 batch 位置；跳过已核验的完成单元。每 25 步及每轮保存一次，正常 SIGINT/SIGTERM 会保存后退出。

只看计划、不训练也不创建输出：

```bash
bash experiments/grouped_robot_label_init/scripts/run.sh --plan-only
```

## 输出

默认目录：`outputs/grouped_robot_label_init/robot_v5_l4_head5/`。

- `summary.md`：三个方法的结果和相对基线变化。
- `summary.json`、`summary.csv`、`aggregates.csv`：完整数据、逐样本变化和配对差值。
- `learning_curves.csv`、`comparison.png`：学习曲线和性能比较。
- `initial_diagnostics.json`：训练前输出/梯度检查，单独标记为未训练结果。
- `<method>/init_<index>/training/independent/`：新增训练的 Adam 断点、末轮权重、每轮历史。
- `uniform/init_<index>/training_reference.json`：历史基线权重的引用；没有伪造新的训练过程。
- `<method>/init_<index>/<role>/<condition>/predictions.pt`：每个样本的概念分布和分支 label 概率。
- `manifest.json`、各 `*_lock.json`：代码、数据、初始化、引用和结果的指纹。
- `train.log`、`heartbeat.json`：日志和当前进度。

所有新增代码与旧实验隔离；不要修改运行中 manifest 固定的源代码。已有输出必须使用 `--resume`。

## 工程验收

```bash
bash experiments/grouped_robot_label_init/scripts/check.sh
```

包含 Ruff、ty、Pylint 和真实 CUDA 测试，检查初始化只改变预期参数、历史基线复用、冻结前半段、完整评估、中断恢复与连续训练一致，以及旧输出保护。

如需单独跑两轮工程检查：

```bash
bash experiments/grouped_robot_label_init/scripts/launch.sh \
  --session robot_init_check \
  --out outputs/grouped_robot_label_init/manual_check \
  --development --head-epochs 2 --diagnostic-every 1
```

当工程预算与历史基线不同，程序会将三个 uniform 也从头训练，保证九个单元预算一致。工程结果会标记为 `engineering_run=true`，不能与正式 300 轮结果混作结论。
