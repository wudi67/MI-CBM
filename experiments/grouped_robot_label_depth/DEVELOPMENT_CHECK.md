# A/B 后半段深度代码验收

正式的 300 epoch 实验尚未启动。下列结果只验证实现和运行流程。

## 自动检查

执行 `bash experiments/grouped_robot_label_depth/scripts/check.sh`：

- Ruff 检查和格式检查通过。
- ty 类型检查通过。
- 项目 Pylint 检查通过。
- pytest：6 passed，52.67 秒。1 条第三方 pynvml 弃用提醒。

CUDA 测试覆盖五层电路在正常、foot 纠正、全部纠正三种条件下，加速计算与直接逐分支演化的概率和梯度一致；前半段无梯度；A/B 对应初始层配对；A0 引用旧权重；全部 36 条件完整；单个 batch 中断后权重、Adam、进度、随机状态与连续训练完全一致；旧输出内容和时间戳不变；改变源配置、样本顺序或结果文件会被拒绝。

## 真实 Robot 数据短跑

输出目录：`outputs/grouped_robot_label_depth/cuda_check_20260916/`。

配置为 `--development --head-epochs 2 --diagnostic-every 1`，保留真实数据规模、batch=1024、eval batch=2048、lr=0.01、四层冻结前半段及全部六个单元。

第一次通过 tmux 启动时加 `--max-steps 1`，检查程序在 epoch 0、Adam step 1、样本位置 1024 安全暂停，tmux 自动退出。随后通过 `--resume` 启动剩余工作。

- 设备：NVIDIA GeForce RTX 4090，模型、振幅和活动梯度使用 CUDA。
- 六个单元全部完成，各 2 epoch、36 次 Adam 更新，共 216 次更新。
- 36 个 train/validation 评估条件全部完成。
- 三个五层单元的五层均有非零梯度，前半段梯度为 None。
- 所有 A/B 单元的样本顺序一致。
- 结果锁、代码指纹、整个旧实验引用链的文件指纹复核通过。
- `test_read=false`，`test_evaluated=false`。
- 最终 heartbeat 为 complete，训练进程退出，tmux 会话自动退出。
- `comparison.png` 已生成并检查布局。

短跑的预算与原有 A0 的 300 epoch 不相同，因此这次六个单元均从头训练，没有混用不同预算。正式默认设置经 `--plan-only` 确認：复用旧 A0，另训练五个 300 epoch 单元，总计 27,000 次新更新。

默认正式输出 `outputs/grouped_robot_label_depth/robot_v5_l4_seed0/` 在验收结束时尚未创建。工程短跑的准确率不能作为本次 A/B 科学问题的正式结论。
