# Robot：measurement-induced 后半段深度 A/B 实验

目的是检验：原有 24 参数的后半段 VQC 是否限制了真实概念控制下的 label 学习，以及增加后半段深度是否改善概念纠正后的预测。两组都使用中途测量和测量记录控制 X 门，保留测量后的 B 寄存器。只实现已确定的 A、B 两组。

| 条件 | A | B |
|---|---|---|
| 前半段 | 同一个已训练 300 epoch 的四层 VQC，冻结 | 与 A 完全相同 |
| 后半段层数 | 1 | 5 |
| 后半段参数 | 24 | 112 |
| 后半段初始化编号 | 0、1、2 | 0、1、2 |
| 训练模式 | Independent | Independent |
| 后半段训练预算 | 300 epoch | 300 epoch |
| optimizer | Adam，lr=0.01，grad clip=5 | 相同 |
| batch / eval batch | 1024 / 2048 | 相同 |

实际 Robot v5 数据为五个二元概念，依次是 head_shape、body_shape、has_antennae、ears_shape、foot_shape。前半段使用 A5+B5 共十个上传位，后半段添加一个初态为零的预测位。概念数、数据拆分和预处理均从已有实验读取。

## 对照条件

- 前半段复用 `outputs/grouped_robot_label_continuation/robot_v5_l4_seed0` 中已完成 concept 300 + label 300 的实验。启动时审计整个上游链、模型权重、配置及结果文件。
- 前半段、data re-uploading、后半段门结构均调用原有 `GroupedDynamicVQC` / FusionModel / TorchQuantum 实现。后半段每层是原来的 RY、RZZ、RY、RXX 星形电路，最后接 RZ/RY 读出。参数量为 `22 × 层数 + 2`。
- A/init_0 直接引用已经完成的 1 层后半段 300 epoch 端点，并重新核验预测；不重训、不复制成一个“新种子”、不改变旧输出。其余五个单元从随机初始化训练。
- 同一初始化编号的 A、B 共用第一层和最后两个读出参数的初值。B 新增的四层按原实现的均匀分布随机初始化，固定额外层种子 100000、100001、100002。B 不从已经训练过的 A 继续训练。
- 所有单元使用相同训练样本顺序，保持原有 shuffle seed=0、epoch offset=100。编号只改变后半段初始化；不会改变前半段、数据或样本顺序。
- 默认 train 18,432 张、validation 6,144 张。前半段振幅计算一次并缓存到 CUDA，之后冻结；后半段仍以实际量子门计算并反向传播。
- Independent 训练时使用真实概念控制 X，原有测量分支及各自的 B 量子态不变。精确枚举全部 32 个物理测量分支，并对分支输出概率求和，不进行振幅跨分支求和、重新制备或后选择。
- 固定末轮评估所有初始化。三个编号反映同一个前半段下的后半段初始化波动，不能写成三次完整模型独立训练的结果。

## 评估内容

每个单元分别在 train 和 validation 评估：正常预测、仅纠正 foot_shape、纠正全部五个概念，共 36 个条件。纠正仅替换控制 X 门的经典概念记录，保留物理测量分支。

主要查看 label 准确率、BCE、balanced accuracy，以及“纠正后准确率 − 正常准确率”的百分点变化。同时保留概念指标、逐样本预测、纠正后由错变对和由对变错的数量。

每轮保存正常验证指标；每 50 轮及末轮评估训练集和验证集的真实概念控制指标。这样可以区分训练集是否仍拟合不足，以及改善是否延伸到验证集。A/init_0 保留原有历史，缺失的历史指标不会补造。

汇总包含每个初始化、均值与样本标准差（ddof=1），以及同编号 B−A 的差值。加深改善支持后半段深度的作用；加深无改善不能直接证明接口有问题。该对照同时改变层数、参数量和门的组合，不能单独归因于参数数量，也不是测量有无的消融。

本轮使用验证集做结构诊断，不读取或评估 test。正式测试应在架构及评估协议确定后进行。

## 一次启动六个单元

在项目根目录执行：

```bash
cd /root/autodl-tmp/Quantum_CBM
bash experiments/grouped_robot_label_depth/scripts/launch.sh --session robot_label_depth
```

默认按 A0、B0、A1、B1、A2、B2 顺序执行。A0 核验复用，其余五个训练，无须手动逐个启动。每个新单元 5,400 次 Adam 更新，总计 27,000 次新更新。程序及状态窗口结束后自动退出 tmux，不保留空会话。

查看进度或接回界面：

```bash
bash experiments/grouped_robot_label_depth/scripts/status.sh
/root/miniforge3/envs/VQC/bin/tmux attach -t robot_label_depth
```

意外中断后，在原会话已退出的情况下恢复：

```bash
bash experiments/grouped_robot_label_depth/scripts/launch.sh --session robot_label_depth --resume
```

恢复检查代码、数据、上游结果和配置指纹，加载完整 Adam、随机状态、epoch 与批次位置。每 25 步及每个 epoch 保存，正常 SIGINT/SIGTERM 会安全保存并退出；已完成单元和评估直接核验复用。不要在运行中修改被 manifest 固定的源代码。

只看计划，不创建实验输出：

```bash
bash experiments/grouped_robot_label_depth/scripts/run.sh --plan-only
```

## 输出与隔离

默认输出：`outputs/grouped_robot_label_depth/robot_v5_l4_seed0/`。

- `summary.md`、`summary.csv`：全部单元结果。
- `summary.json`：配对差值、逐样本变化、模型来源等。
- `aggregates.csv`、`comparison.png`：三次后半段初始化的均值和标准差。
- `L*/init_*/training/independent/`：新训练单元的历史、梯度检查、Adam 断点及末轮权重。
- `L1/init_0/training_reference.json`：A0 的历史权重引用。
- `L*/init_*/*/*/predictions.pt`：各评估条件的逐样本概率。
- `manifest.json`、各 `*_lock.json`：实验代码、输入、初始权重及结果指纹。
- `heartbeat.json`、`train.log`：进度与日志。

代码和输出均在独立目录，不改写前面的实验。可用 `--out` 指定新目录；已有输出必须用 `--resume`。

## 工程检查

```bash
bash experiments/grouped_robot_label_depth/scripts/check.sh
```

运行 Ruff、ty、Pylint 及 CUDA 测试，覆盖五层电路的概率/梯度一致性、冻结前半段、初始化配对、A0 复用、真实控制训练、批次中断恢复和旧结果保护。

完整 36 条件的短预算检查可以单独执行：

```bash
bash experiments/grouped_robot_label_depth/scripts/launch.sh \
  --session robot_depth_check \
  --out outputs/grouped_robot_label_depth/manual_check \
  --development --head-epochs 2 --diagnostic-every 1
```

工程预算与历史 A0 的 300 epoch 不相同时，六个单元全部从头训练，避免把 300 epoch 与 2 epoch 混在一张表里。这些结果会明确标记为工程检查，不能当作正式实验结果。
