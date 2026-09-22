# 固定电路的 Joint / Sequential 训练对照

已完成的 CUDA、冻结及恢复检查见 [DEVELOPMENT_CHECK.md](DEVELOPMENT_CHECK.md)。
完整 seed 0 的 200 轮配对结果见 [RESULTS_SEED0.md](RESULTS_SEED0.md)。

本实验使用新的并列目录，复用 `experiments/grouped_dynamic_vqc` 的模型、
损失、数据预处理、评估和 CUDA 工具。旧模型、旧训练入口及其输出不修改。
量子前端仍实际调用 `FusionModel.TQLayer` 的上传、U3 和 CU3 实现。

## 实验协议

| 路线 | 第 1–100 epoch | 第 101–200 epoch |
|---|---|---|
| Joint | 概念 NLL + Label BCE；更新前端和分类头 | 同左 |
| Sequential | 仅概念 NLL；只更新前端，分类头保持初始权重 | 冻结整个前端；仅 Label BCE 更新分类头 |

Sequential 第 100 轮的固定结果就是概念单训结果。其分类头尚未训练，
该时点的 Label 指标只作为诊断，在 Markdown 主表中显示为“未训练分类头”。
分类阶段继续使用预测测量分支、相应的后五位条件量子态和测量控制记录，
不注入真实概念，不重置后五位，不改变电路结构。

冻结对象是测量前全部 10 个 qubit 的前端。第 101 轮开始创建新的、只含分类头
参数的 Adam，不继承前端的优化器状态。分类头从两条路线共享的初始权重开始。
前端冻结后仍执行实际量子前向计算；量子训练的输入张量、参数和计算均位于 CUDA。

默认固定：dSprites compact_c、居中 + Pool(10,4)、完整训练集方差分配、
前端 4 层、分类头 1 层、seed 0、Adam 0.01、batch 1024、验证 batch 2048。
损失权重均为 1，梯度裁剪 5。保留完整训练 25,593 / 验证 5,479 的划分。
不新增 Pool 种类统计，不删除池化后相同的样本，不预处理或评价测试图片。

两条路线读取同一份缓存和初始权重文件；每轮使用独立、由 seed/epoch 决定的
样本排列，记录完整顺序的哈希。默认各有 5,000 次 batch 更新：Joint 前端和
分类头分别更新 5,000 次，Sequential 二者分别更新 2,500 次。
总 batch 更新数相同；计算时间和每个模块的更新次数有意不同，不能称作相同算力预算。

## 启动与恢复

在仓库根目录执行。脚本自动设置 VQC Python、CUDA 库和确定性计算选项；
没有 CUDA 时直接失败，不会静默回退 CPU。一个后台 worker 顺序执行两条路线。

```bash
# 查看配置，不启动训练
experiments/grouped_vqc_training_modes/scripts/run.sh \
  --out outputs/grouped_vqc_training_modes/dsprites_l4_seed0 --plan-only

# 一次启动完整配对实验：Joint 200；Sequential 100 + 100
experiments/grouped_vqc_training_modes/scripts/launch.sh \
  --session grouped_modes_l4_seed0 \
  --out outputs/grouped_vqc_training_modes/dsprites_l4_seed0

# 查看当前路线、阶段、epoch、优化步、心跳、PID 和设备
experiments/grouped_vqc_training_modes/scripts/status.sh \
  --out outputs/grouped_vqc_training_modes/dsprites_l4_seed0

# 上方日志，下方动态状态；Ctrl+b 然后 d 离开
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_modes_l4_seed0

tail -f outputs/grouped_vqc_training_modes/dsprites_l4_seed0/train.log

# 原 worker 已退出后恢复；保留原配置和进度
experiments/grouped_vqc_training_modes/scripts/launch.sh \
  --session grouped_modes_l4_seed0_resume \
  --out outputs/grouped_vqc_training_modes/dsprites_l4_seed0 --resume
```

恢复检查配置、源码、共享权重、数据缓存和 CUDA/软件环境。原配置禁止在恢复时
修改。已完成的路线核验结果后跳过，未完成路线恢复模型、当前阶段、Adam、RNG、
样本排列、epoch 内偏移、损失累计与历史。`.worker.lock` 防止同一实验被重复运行。
SIGTERM/SIGINT 在安全的训练边界保存并暂停；硬中断最多丢失一个检查点间隔。
程序结束后，训练窗格和监控窗格自动关闭，tmux 会话随之退出。
日志、检查点和结果保留在输出目录；暂停后的恢复仍使用新 session 和 `--resume`。

仅做工程检查时可以缩短预算，必须使用独立输出目录：

```bash
experiments/grouped_vqc_training_modes/scripts/launch.sh \
  --session grouped_modes_cuda_check \
  --out outputs/grouped_vqc_training_modes/cuda_check \
  --epochs 4 --concept-epochs 2 --shots 32 --max-steps 1

# 第一条路线第 1 步暂停后，使用新的 session 恢复至两条路线完成
experiments/grouped_vqc_training_modes/scripts/launch.sh \
  --session grouped_modes_cuda_check_resume \
  --out outputs/grouped_vqc_training_modes/cuda_check --resume

experiments/grouped_vqc_training_modes/scripts/check.sh
```

`--max-steps` 表示每条尚未完成路线的累计优化步上限，不是本次新增步数。
它只用于检查暂停/恢复，不保存到科学实验配置中。

## 输出与判断

| 文件 | 用途 |
|---|---|
| `config.json`, `manifest.json` | 固定预算、环境、源码与配对协议 |
| `initialization.pt` | 两条路线共享的初始权重与随机数状态 |
| `data.pt`, `preprocessing.json` | 共同输入及仅训练集拟合的方差路由 |
| `heartbeat.json`, `train.log` | 当前路线/阶段和运行状态 |
| `{joint,sequential}/resume.pt` | 最近可恢复的完整训练状态 |
| `{joint,sequential}/history.json` | 每轮损失、概念/Label 指标与顺序哈希 |
| `{joint,sequential}/gradient_checks.json` | 各阶段活动模块梯度和冻结模块无梯度证据 |
| `{joint,sequential}/endpoints/epoch_0100.{pt,json}` | 第 100 轮固定权重与全部诊断 |
| `{joint,sequential}/endpoints/epoch_0200.{pt,json}` | 第 200 轮固定权重与全部诊断 |
| `summary.json`, `summary.csv`, `summary.md` | 两条路线完成且配对核验通过后生成的汇总 |

每个固定时点同时报告：Shape/Scale 分组准确率、分组同时正确率、联合 MAP、
正确概念组合的单次测量概率、非法编码概率、Label accuracy/BAcc/BCE、
真实 Shape/Scale/两组控制与零控制的读数，以及有限 shots 模拟。
量子训练使用可微的精确 Born 分支求和；默认 256 shots 的采样仅用于端点评估。

先比较第 100 轮的 Joint 与概念单训，再比较第 200 轮的 Joint 与 Sequential。
前者控制前端更新次数，后者控制总 batch 更新次数。
恢复期间的训练秒数会累计，但只计训练步，不含预处理、验证和 I/O；
显存记录是当前进程的 PyTorch 峰值分配值，不能当作设备总显存或跨恢复的全程峰值。

该框架保留后五位的图像信息通道，Sequential 不是原论文的 Independent CBM。
控制记录干预只改条件 X 的经典记录，不纠正测量后的条件量子态；干预下降不能
单独证明控制通道失效。首轮 seed 0 与短程检查均属于开发验证，不是最终测试结论。
