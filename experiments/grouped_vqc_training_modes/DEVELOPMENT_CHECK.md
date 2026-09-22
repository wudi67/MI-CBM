# 训练方式对照的实现验证（2026-09-12）

以下记录对应早前的短程工程验证。随后完成的 Joint 200 / Sequential 100+100
完整实验见 [RESULTS_SEED0.md](RESULTS_SEED0.md)。

- Ruff、格式检查、ty、Pylint 均通过；新阶段测试与原 grouped VQC 回归测试合计 **23 passed**。
- 真实 dSprites / CUDA 测试覆盖 epoch 中途、概念阶段结束及分类阶段开始后的恢复。
  恢复训练与连续训练的模型、Adam、RNG、样本顺序、损失历史和梯度记录逐比特一致。
- 概念单训更新不受 Label 真值翻转影响；冻结模块不接收梯度。
- 配对检查发现样本顺序/历史改变时会拒绝汇总；恢复拒绝配置、初始权重变化。

后台完整划分检查使用 RTX 4090、前端 L4、分类头 L1、264 个参数、batch 1024：

| 路线 | 短程预算 | 前端更新 | 分类头更新 |
|---|---|---:|---:|
| Joint | 4 epoch | 100 | 100 |
| Sequential | 概念 2 + 分类 2 epoch | 50 | 50 |

两条路线均使用训练 25,593、验证 5,479 张图片，测试图片不预处理、不评价。
通过 tmux 在 Joint 第 1 步暂停，再从同一目录恢复；两个后台窗格正常结束，worker 已退出。
端点使用 32 shots 模拟，这是工程检查配置，不是最终的科学实验预算。

结果核验：同一共享初始化、同一缓存、每轮样本顺序相同；Sequential 分类阶段前端
逐张量不变，概念预测指标也保持完全相同；概念阶段分类头与共享初始权重逐张量相同。
Joint、概念阶段、分类阶段的活动梯度均位于 `cuda:0`。
195 个实验相关源码哈希与运行清单一致；旧实验的 183 个源码文件全部保持不变。

证据文件：

- [静态检查与测试日志](../../outputs/grouped_vqc_training_modes/quality_checks.log)
- [运行与来源核验](../../outputs/grouped_vqc_training_modes/l4_seed0_cuda_smoke_20260912/verification.json)
- [短程汇总](../../outputs/grouped_vqc_training_modes/l4_seed0_cuda_smoke_20260912/summary.md)
- [完整端点与配对结果](../../outputs/grouped_vqc_training_modes/l4_seed0_cuda_smoke_20260912/summary.json)
- [后台日志](../../outputs/grouped_vqc_training_modes/l4_seed0_cuda_smoke_20260912/train.log)

短程分数只说明运行流程可用，不能据此判断训练方式优劣或概念干预有效。
