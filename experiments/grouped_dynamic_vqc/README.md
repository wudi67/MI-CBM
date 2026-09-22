# 分组概念动态 VQC：独立实现

本目录实现 `figures/grouped_dynamic_vqc_v2.md` 中的方案。旧模型、旧实验代码、权重和结果不作修改。唯一共享的模型实现是仓库根目录的 `FusionModel.py` 及其 TorchQuantum 依赖。数据使用已经通过准入检查的 dSprites `confirmatory_2027/compact_c`。

已完成的设备验证、10 epoch 结果及其适用边界见 [DEVELOPMENT_CHECK.md](DEVELOPMENT_CHECK.md)。

## 电路与训练

```text
图像 → 图像自身 bbox 居中（仅平移、零填充）
     → AdaptiveAvgPool(10,4) → 40 个值
     → 完整训练集方差排序、轮流分配给 10 位 → 乘 π 一次
     → [四值上传 + 每位 U3 + 顺序 CU3 环] × 4 层
     → 测量 q0..q4，得到 shape(2 bit) | scale(3 bit)
     → 对保留条件量子态的 q5..q9 执行 X^m
     → 加入独立 r=|0⟩ → 星形量子分类头 → 测量 r
```

- **前端直接调用 `FusionModel.TQLayer.forward()`**：继承原来的上传器、U3/CU3 参数和执行代码，只替换末端读出，使其返回量子态。线路编码由原 `single_enta_to_design()` 生成。
- 每位上传顺序为 `Ry(a0) → Rz(a1) → Rx(a2) → Ry(a3)`。每层重复同一组输入角度，参数独立；层间不测量或重置，不增加 beta 门。原实现将不同线的上传/U3 交错执行，与先完成全部上传再执行全部 U3 等价，因为不同线的单比特门可交换。
- CU3 按 `0→1,1→2,…,8→9,9→0` 的顺序执行。输入位为 10 个，完整电路含读出位为 **11 个**。
- 概念整数编码为 `8*shape + scale`，q0 为最高位。shape 合法值 0–2，scale 合法值 0–5。18 个合法码、14 个非法码；**不丢弃或重归一化非法码**，全部分支参与标签预测，概念 NLL 抑制错误码概率。
- 分类头为 `Ry(6) → RZZ(5,r) → Ry(6) → RXX(5,r)`，默认 1 层，再执行 `Rz(r) → Ry(r)`。其连接是星形，保留 B 的量子态，r 不上传图像。名义参数数为 `60*4 + 22*1 + 2 = 264`。
- 默认 Adam，学习率 0.01，batch 1024，验证 batch 2048，梯度裁剪 5，概念联合 NLL 与标签 BCE 权重各 1。默认训练预算 100 epoch；短程验证通过命令指定 10 epoch。固定末轮为最终结果，没有根据最佳验证分数选择权重。

### 中途测量的精确 CUDA 模拟

前端输出 `ψ(x)`，重排为 `[batch,32个测量分支,32个保留态振幅]`。分支振幅仍包含该分支概率的平方根，不单独归一化，因而零概率分支也能安全处理：

```text
p(m|x) = ||ψ_m(x)||²
p(y=1|x) = Σ_m ||Π_(r=1) U_label [X^m ψ_m(x) ⊗ |0⟩]||²
```

不同 m 之间只加概率。B 内部的复振幅、相位和干涉均保留。训练使用可微的精确分支求和，不对离散采样结果求导。

为减少模拟开销，分类头每批用 TorchQuantum 演化 32 个 B 基态，得到同一电路的复线性映射 `K[b,j]=⟨j,1|U_label|b,0⟩`；随后把每个分支的复振幅乘 K。**这不是用概念概率查询标签表**。K 在每次前向重新计算并保留梯度。测试将其输出及梯度与逐分支直接演化对照。

这是在 GPU 上运行的量子电路模拟；本目录尚未实现量子硬件的动态电路提交。有限 shots 验证从 `(m,y)` 联合分布采样，保留概念与标签的相关性。

### 数据与解释性边界

保持现有原始栅格分组划分：train 25,593 / val 5,479 / test 5,477。训练/验证图像分别变换，方差映射只由完整 train 图像拟合，随后固定复用。开启开发子集时也使用完整训练划分拟合方差，实际优化行数另行记录。

`preprocessing.json` 记录数据、准入文件、特征缓存和源样本哈希，以及方差、映射和所用行号。NPZ 成员读取会物化整个数组，但测试行不进入图像变换、方差拟合、训练或评估。原始栅格划分无重叠，并不意味着居中/池化后的特征没有重复或碰撞。

后五位参与输入编码和前段纠缠，保留输入相关的残余通道。本实现及指标不证明严格概念瓶颈，也不证明概念解释覆盖全部标签信息。

末轮报告 shape、scale、两组同时纠正及全零控制四项诊断。纠正仅替换控制 X 门的经典记录，保持原测量分支及其条件量子态，不是对量子态本身的概念纠正，也不保证准确率单调提高。

## 运行

在仓库根目录执行。脚本配置 VQC 环境、CUDA 确定性选项及 `LD_LIBRARY_PATH`，解决当前设备直接导入 TorchQuantum 的 C++ 运行库兼容问题。训练要求 CUDA，缺失时直接报错。

```bash
# 10 epoch，全训练划分，后台运行
experiments/grouped_dynamic_vqc/scripts/launch.sh \
  --session grouped_dsprites_l4 \
  --out outputs/grouped_dynamic_vqc/dsprites_l4_seed0 \
  --epochs 10

# 查看状态 / 进入训练终端（上方日志、下方状态）
experiments/grouped_dynamic_vqc/scripts/status.sh \
  --out outputs/grouped_dynamic_vqc/dsprites_l4_seed0
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_dsprites_l4

# 跟随日志
tail -f outputs/grouped_dynamic_vqc/dsprites_l4_seed0/train.log

# 原训练中断后恢复；读取原 config，不重置学习率、Adam 或样本顺序
experiments/grouped_dynamic_vqc/scripts/launch.sh \
  --session grouped_dsprites_l4_resume \
  --out outputs/grouped_dynamic_vqc/dsprites_l4_seed0 --resume

# 前台运行，或指定更小的开发子集
experiments/grouped_dynamic_vqc/scripts/run.sh \
  --out outputs/grouped_dynamic_vqc/quick_check \
  --epochs 3 --train-limit 2304 --val-limit 576

# 完整检查：Ruff、ty、Pylint、pytest（包含真实 dSprites/CUDA 恢复测试）
experiments/grouped_dynamic_vqc/scripts/check.sh

# 重新测量本设备的完整训练步吞吐与显存占用
source experiments/grouped_dynamic_vqc/scripts/environment.sh
"$VQC_PYTHON" -m experiments.grouped_dynamic_vqc.benchmark \
  --out outputs/grouped_dynamic_vqc/cuda_batch_benchmark.json
```

使用 `run.sh --help` 查看参数。`--front-layers` 默认 4；`--label-layers` 默认 1。更改实验参数请选新输出目录。默认 100 epoch 命令是上述启动命令移除 `--epochs 10`，仍只是开发训练预算。

`--max-steps N` 在第 N 个累计优化步后保存并暂停，用于小规模检查或演练恢复。正常停止可向 `heartbeat.json` 中的训练 PID 发送 SIGTERM，程序在当前训练步结束后保存。异常中断最多回退到最近保存的步，默认每 25 步和每个 epoch 保存一次。恢复严格核对源码、配置、CUDA/软件环境、预处理与数据缓存哈希。

tmux 训练结束后保留终端内容，worker 已退出。离开终端使用 `Ctrl+b` 后按 `d`；不用时可关闭对应的已完成 tmux 会话。

## 输出文件

| 文件 | 内容 |
|---|---|
| `config.json`, `manifest.json` | 配置、实际源码和共享依赖哈希、CUDA 环境、架构边界 |
| `preprocessing.json`, `data.pt` | train-only 映射、划分来源和训练/验证缓存 |
| `initial_validation.json` | 训练前验证分数 |
| `history.json` | 每个 epoch 的训练损失与验证指标 |
| `resume.pt` | 最新权重、Adam、随机数状态、样本排列、偏移和训练记录 |
| `cuda_gradient_check.json` | 实际输入/参数设备与两段电路的梯度检查 |
| `result.json` | 固定末轮验证、控制记录干预和有限 shots 诊断 |
| `heartbeat.json`, `train.log` | 进度、进程与日志 |

`joint_map_accuracy` 是整个 5-bit 测量分布的众数命中率；`group_argmax_exact_accuracy` 是两组边缘分布各取众数后的共同命中率；`joint_single_shot_probability` 是实际单次测量同时得到正确两组概念的平均概率。三者不能互换。

所有结果应注明训练预算、随机种子和开发验证角色，不能据此宣称正式测试性能或量子优势。
