# dSprites CUDA 开发验证（2026-09-12）

当前实现已完成完整训练划分上的 10 epoch 验证。此次结果用于确认代码、梯度、CUDA 和恢复流程可运行，不是正式测试性能，也未证明解释或干预有效。

- 输出目录：`outputs/grouped_dynamic_vqc/dsprites_l4_seed0_cuda_check/`
- 设备：NVIDIA GeForce RTX 4090；PyTorch 2.11.0+cu130；输入、模型参数及梯度实际位于 `cuda:0`。
- 前端 4 层，分类头 1 层；264 个参数；batch 1024，Adam lr=0.01；seed=0。
- 训练 25,593 张、验证 5,479 张，250 个优化步。测试行不参与预处理、拟合或评估。
- 本次训练的 PyTorch 峰值分配显存 **0.777 GiB**；这是分配给张量的峰值，不是进程或设备总显存。独立完整训练步基准约 **3914 张/秒**，不含预处理和验证。

| 验证指标 | 初始化 | 第 10 epoch |
|---|---:|---:|
| 标签准确率 | 49.59% | 71.33% |
| 标签 BCE | 0.6914 | 0.5301 |
| 概念联合 NLL | 4.0688 | 2.1188 |
| 概念联合众数命中率 | 3.25% | 34.06% |
| 两组边缘众数共同命中率 | 0.91% | 24.86% |
| 正确两组概念的单次测量概率 | 2.54% | 14.05% |
| 非法概念编码概率 | 50.04% | 9.01% |

256 shots/image 验证的标签准确率为 71.22%，正确概念组合采样频率为 14.06%。它们是有限采样的模拟结果。

替换为真实 shape、真实 scale、两组真实值的控制记录后，标签准确率分别为 **51.93%、60.25%、60.12%**；全部关闭 X 控制时为 **65.61%**。这里保持原测量分支及其条件量子态，仅改变经典门控制。因此当前结果不能支持“纠正概念即可改善预测”的结论；后续需要专门分析训练方式与残余状态通道。

## 工程验证

- Ruff、ty、Pylint 全部通过；**17 项 pytest 测试通过**。
- 优化分支求和与逐分支 TorchQuantum 演化的输出、梯度一致；CPU/CUDA 数值一致。
- 测量去除跨分支相干性，保留 B 分支内部相位的影响；控制位方向、零概率分支和联合 shots 采样有独立检查。
- 真实 dSprites/CUDA 测试验证中途恢复后的权重、Adam、随机数状态、样本顺序和训练记录与连续执行逐比特一致。
- 此运行实际通过 tmux 启动，在第 3 步保存暂停，再恢复至第 250 步。恢复后的上下两个窗格均正常工作，完成后训练进程退出。
- 核验当前源码哈希、配置/预处理来源、缓存、检查点及末轮结果均匹配。

原始证据：

- [固定末轮结果](../../outputs/grouped_dynamic_vqc/dsprites_l4_seed0_cuda_check/result.json)
- [检查点与源码核验](../../outputs/grouped_dynamic_vqc/dsprites_l4_seed0_cuda_check/verification.json)
- [CUDA 梯度检查](../../outputs/grouped_dynamic_vqc/dsprites_l4_seed0_cuda_check/cuda_gradient_check.json)
- [训练日志](../../outputs/grouped_dynamic_vqc/dsprites_l4_seed0_cuda_check/train.log)
- [静态检查与测试日志](../../outputs/grouped_dynamic_vqc/quality_checks.log)
- [CUDA batch 基准](../../outputs/grouped_dynamic_vqc/cuda_batch_benchmark.json)

启动、恢复和查看状态的命令见 [README](README.md)。默认的 100 epoch 预算尚未运行。
