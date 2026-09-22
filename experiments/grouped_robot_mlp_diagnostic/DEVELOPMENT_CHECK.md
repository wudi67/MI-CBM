# 工程验证记录

已验证当前实现；300 轮的正式诊断未启动。

## 静态检查和自动测试

运行：

```bash
experiments/grouped_robot_mlp_diagnostic/scripts/check.sh
```

- Ruff check、Ruff format、ty、Pylint 均通过。
- pytest：7 passed（约 15.5 秒）；唯一 warning 来自既有环境中的 pynvml 弃用提示。
- 穷举 32 个真实概念组合 × 32 个物理测量分支 × 32 个纠正 mask，核对独立逐位参考实现。
- 检查加权对象是 label 概率，不是联合 MAP 或 logits。
- 检查全纠正与直接真实概念输入一致；对→错/错→对计数守恒。
- 真实 CUDA 测试证明：改变训练时的预测概念概率、验证标签及验证概念，不改变训练得到的 MLP 权重和 Adam。
- epoch 内第 3 步暂停，恢复结果的模型、Adam、完整进度和 RNG，与连续训练完全一致。
- 使用缩小数据的真实 VQC 来源链，验证缓存、结果重算、完成后恢复、输出隔离及结果篡改拒绝。
- 来源实验所有文件的 SHA256 和 mtime 在测试前后均不变。

## 实际 Robot 来源和后台生命周期

工程输出：`outputs/grouped_robot_mlp_diagnostic/cuda_check_20260916`。
这里的后缀是工程目录名称；实际运行时间以产物中的 UTC 时间戳为准。

使用真实来源 `outputs/grouped_robot_label_continuation/robot_v5_l4_seed0`，完整
train 18,432 / validation 6,144 行，缩短为 2 epoch，batch 1024。

1. 用 `launch.sh --development --epochs 2 --diagnostic-every 1 --max-steps 1`
   启动 `robot_mlp_diag_cuda_check`；在第 1 步、样本偏移 1,024 处正常暂停。
2. 用 `launch.sh --resume` 启动 `robot_mlp_diag_cuda_check_resume`；恢复至
   2 epoch、36 次 Adam 更新，完成 6/6 个正常/纠正评估。
3. 两个 tmux 会话均随 worker 和 monitor 结束自动退出；日志持久化在 `train.log`。
4. `summary.md/json/csv`、`diagnostic.png`、原始预测和最终结果锁均生成。
5. 最终锁包含 24 个产物；来源锁包含 326 个文件；源码锁包含 384 个文件。

真实 GPU 为 NVIDIA GeForce RTX 4090。MLP 参数、概念输入和 label 均位于 `cuda:0`，
参数量 113，首次梯度范数约 0.19934，冻结前端不参与训练。

从实际 VQC 检查点重新计算的 Born 概率，与历史保存结果在训练和验证集上的
最大绝对差异均为 0。实际冻结前端 SHA256：

```text
ebcbe480d7f219806a15a9b069e9aa530dfcb3115b2e347ad7d2bb06567f16e1
```

`test_read=false`、`test_evaluated=false`。两轮输出只是工程检查，不能作为固定
300 轮诊断的结论，也不能据此选择训练预算或最佳 epoch。
默认正式输出 `outputs/grouped_robot_mlp_diagnostic/robot_v5_l4_seed0` 未创建。
