# 四层 Standard 与概念控制诊断

本目录独立实现新训练入口、检查点、评估和启动脚本。复用
`grouped_dynamic_vqc` 的 `GroupedDynamicVQC`、控制 X、CUDA 和数据缓存读取，
实际调用 `FusionModel.TQLayer` 与 TorchQuantum 的 uploading、U3、CU3。
不修改原来的模型、训练代码、实验输出或缓存。历史 provenance 检查复用已有工具；
不会启动任何 MLP 实验。

## 默认一次运行的内容

| 路线 | 初始化 | 训练目标 | 更新参数 | 预算 |
|---|---|---|---|---|
| Standard | 与旧 Joint 相同的初始权重 | Label BCE | 全部前端和分类头 | 200 epoch / 5,000 step |
| 真实控制头 | Sequential100 的前端＋原始未训练头 | 真实五位控制下的 Label BCE | 只更新分类头 | 100 epoch / 2,500 step |
| 零控制头 | 同上 | 全零控制下的 Label BCE | 只更新分类头 | 100 epoch / 2,500 step |
| 测量控制头 | 原 Sequential200 | 已训练完成 | 默认复用，无新增训练 | 原分类阶段 100 epoch / 2,500 step |

已有 Joint200、Sequential200 同时接受新的统一干预评价。Standard 训练代码直接对
Label BCE 反向传播；概念 NLL 在 `no_grad` 中计算，仅用于监控。

前端固定 L4、10 qubit、每层每位 uploading 四个值，每层 U3＋环形 CU3。
前五位为 Shape 2 bit / Scale 3 bit；后五位条件态保留，加一个读出位和原分类头 L1。
参数总数 264（前端 240＋头 24）。保留所有 32 个测量码，包括 14 个非法码。

默认数据为原缓存中的训练 25,593 / 验证 5,479 张，居中＋Pool(10,4)，方差路由仅由
原完整训练集拟合。输入缓存直接复制并校验哈希，不重新拟合、不更改样本。
不预处理或评价测试集。CUDA 必需，无 CPU 回退；Adam 0.01、训练 batch 1024、
验证 batch 2048、梯度裁剪 5、seed 0、256 shots 端点评价。

Standard 使用旧 Joint 第 1–200 轮的样本排列。两个冻结头使用原 Sequential
分类阶段第 101–200 轮的排列，从全新的 Adam 开始，不继承概念阶段 Adam 状态。
历史头仅在前端、初始化、训练数据、样本顺序、预算、学习率、裁剪和运行环境配对时复用。
如果使用子集、缩短头预算或更改训练超参，程序会额外训练 `head_measured`，保证新的
3×3 头对照内部一致；这类运行不能当作与旧 Joint 完整配对的实验。
`pairing.json` 和所有结果明确记录配对状态。

## 六种控制评价

所有条件保持当前图片的真实测量分支、Born 权重和后五位条件态，只替换经典 X 记录：

| 名称 | 控制记录 |
|---|---|
| measured | 原测量结果 |
| shape | 前两位改成真实 Shape |
| scale | 后三位改成真实 Scale |
| both | 五位改成真实概念 |
| zero | 全零，条件 X 不触发 |
| random | 从其他验证图片的完整测量码分布取控制，精确求期望 |

随机条件对图片 i 使用 `q_i(r) = sum_{j != i} p(r|x_j)/(N-1)`。
这是均匀选择另一张图片，再从该图片的 Born 分布取得完整控制记录的精确平均。
全体验证图片的平均控制码频率保持不变，包括非法码和五位之间的相关性。
没有逐位随机、均匀 32 码替换或 Monte Carlo 抽样误差；该分布只用于冻结模型的
验证诊断，不用于优化、选择模型或读取测试集。

加速计算对保留寄存器形成非选择性混合态，并与原分类头的同一线性映射收缩。
不会相加不同测量分支的振幅。CUDA 测试与全部 32 种恒定控制的原电路直接演化核对。

Label 始终按 0.5 阈值评价。每种控制报告 accuracy、balanced accuracy、BCE、
错误→正确/正确→错误的样本数和比例、预测翻转率、概率绝对变化、真实 Label 概率变化。
另按真实 18 个概念组合、真实 Label 和干预前概率距 0.5 的固定区间分组，报告样本数。
边距区间为 `[0,.05), [.05,.15), [.15,.3), [.3,.50001)`，空组标为空值。
不同模型的边距组可包含不同图片，不能把这些组直接当作配对样本。

概念指标包括 Shape、Scale、分组同时正确、联合 MAP、正确概念组合的单次 Born 概率、
NLL 和非法码概率。单次测量不等于 MAP。原始控制另提供 256 shots 的联合物理采样。
3×3 表分别把 measured / both / zero 用于头训练和评价。

## 一次启动、监控和恢复

在仓库根目录执行。默认输出为
`outputs/grouped_control_diagnostics/dsprites_l4_seed0`。

```bash
# 检查预算，不创建实验、不启动训练
experiments/grouped_control_diagnostics/scripts/run.sh --plan-only

# 一次性后台执行 Standard、两个分类头以及全部干预诊断
experiments/grouped_control_diagnostics/scripts/launch.sh

# 连接：上方日志，下方 Rich 状态
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_control_diagnostics

# 离开界面但保持运行：Ctrl+b，然后 d
experiments/grouped_control_diagnostics/scripts/status.sh
tail -f outputs/grouped_control_diagnostics/dsprites_l4_seed0/train.log

# 原 worker 已退出后，使用相同目录恢复
experiments/grouped_control_diagnostics/scripts/launch.sh \
  --session grouped_controls_resume --resume
```

训练正常结束、暂停或失败时，工作窗格和监控窗格自动退出，不保留 tmux 会话。
日志和结果文件保留。一个 worker 顺序执行所有路线；目录锁防止重复训练。
SIGINT/SIGTERM 在安全边界保存模型、Adam、RNG、样本排列、轮内偏移和损失累计。
恢复检查源码、配置、环境、缓存、历史来源、完成的检查点和诊断输出哈希。
已完成路线核验后跳过；诊断中断时仅重做当前尚未写完的模型诊断。
历史运行唯一允许的源码差异是已有记录及原源码快照证明的 tmux 自动退出修改。

训练窗格可短暂安静，请结合心跳中的 step、epoch、PID 和 GPU 活动判断运行状态。
最后所有训练路线完成后仍需进行诊断，只有父级 `status=complete` 才表示全部完成。

## 输出

| 文件 | 内容 |
|---|---|
| `manifest.json`, `reference_lock.json`, `pairing.json` | 来源、配置、环境、源码、配对证明 |
| `*/resume.pt`, `*/history.json`, `*/gradient_checks.json` | 可恢复训练状态、逐轮指标、实际 CUDA 梯度 |
| `*/endpoint.pt`, `*/result.json` | 固定终点权重、预算和资源记录 |
| `diagnostics/*.json` | 每个模型的六种控制、细分组、shots 结果 |
| `diagnostics/*.pt` | 逐图片 source index、概念分布、各控制 Label 概率 |
| `mode_comparison.csv` | Standard / Joint / Sequential 概念和分类对照 |
| `control_comparison.csv` | 每个模型的控制干预及样本转移 |
| `head_matrix.csv` | 同一冻结前端的 3×3 分类头矩阵 |
| `subgroups.csv` | 18 概念组合、Label、原始边距的分组结果 |
| `summary.md`, `summary.json` | 汇总及完成状态 |

## 工程检查

```bash
# Ruff、ty、Pylint 和实际 CUDA 测试
experiments/grouped_control_diagnostics/scripts/check.sh

# 小规模全流程，独立输出，不能作为科学实验结果
experiments/grouped_control_diagnostics/scripts/launch.sh \
  --session grouped_controls_smoke \
  --out outputs/grouped_control_diagnostics/manual_smoke \
  --epochs 2 --head-epochs 2 --train-limit 36 --val-limit 18 \
  --batch-size 18 --eval-batch-size 18 --shots 16
```

使用 `--max-steps 1` 可在当前未完成路线的累计第一个优化步暂停；随后对同一目录
使用 `--resume`。该选项仅控制暂停，不改变已锁定的科学预算。

## 解释范围

本轮是固定终点的单 seed 开发验证。概念监督是否改善码表对齐、控制对应关系是否
影响分类、分类头是否适应真实控制，是不同问题。
真实控制头仍读取图像相关的后五位条件态，不是原论文 Independent CBM。
零控制头表现不能换算为泄漏比例；真实控制头也不构成信息论上限。
Standard 的码没有概念监督，给它真实控制只是同结构操作对照。
随机控制同时破坏记录与条件态的对应关系，其下降不能单独证明概念语义的因果作用。
