# 实现核验：2026-09-13

本文件记录工程检查。完整 200/100/100 epoch 的正式开发实验尚未启动。

- Ruff check、Ruff format、ty、Pylint E/F/W 全部通过。
- pytest：10 passed。覆盖实际 CUDA、无概念监督梯度、四个分支的轮内恢复、
  验证前中断恢复、模型/Adam/RNG/历史一致性、冻结前端、完整管线、配对条件和篡改拒绝。
- 对全部 32 种恒定控制，新密度矩阵收缩与原 TorchQuantum 分类电路的直接演化一致。
  给不同测量分支加入独立全局相位不会改变结果，验证没有错误地相加分支振幅。
- 改变 Standard 的概念标签后，概念 NLL 改变，而 Label loss、更新后模型与 Adam 完全一致。
- 随机对照保留完整联合控制分布，包括非法码；不产生逐位随机或抽样误差。

## 真实 batch 1024 的全流程检查

输出：`outputs/grouped_control_diagnostics/cuda_pipeline_check_20260913`。
使用完整训练 25,593 / 验证 5,479 张，L4，训练 batch 1024、验证 batch 2048、256 shots。
Standard 和三个头各训练 1 epoch / 25 step；缩短预算自动触发额外训练测量控制头。
这只是管线检查，`standard_matches_historical_budget=false`，不得作为最终模式比较。

| 分支 | CUDA 训练步耗时累计 | PyTorch 峰值分配 |
|---|---:|---:|
| Standard | 7.461 s | 0.789 GiB |
| 真实控制头 | 4.067 s | 0.147 GiB |
| 零控制头 | 3.884 s | 0.147 GiB |
| 测量控制头 | 3.884 s | 0.147 GiB |

设备为 NVIDIA GeForce RTX 4090。耗时不含预处理、验证、I/O 或初始化；峰值是该进程
PyTorch 分配量，不是设备全部显存。测量控制头仍用原 Sequential 的第 101 轮样本顺序。

- 新头第 1 轮与旧 Sequential 第 101 轮的训练 BCE 完全相同：0.6868092545273202。
- 上述两者的全部概念、Label 验证指标完全相同，样本排列哈希完全相同。
- 新评估器复算旧 Joint200 时，已有概念/分类/干预指标最大绝对差为 1.1920928955078125e-7。
- 新评估器复算旧 Sequential200 时，对应最大差为 5.960464477539063e-8。
- 完成六个模型的全部诊断，输出模式表、控制表、3×3 头矩阵、分组表、原始预测和总结。
- 195 个历史源码条目已核验，21 个历史文件锁定，新运行固定 221 个源码条目。

## 生命周期与隔离

- 通过实际 tmux launcher 跑完后，会话自动消失，训练 PID 已退出，GPU 归零。
- 对已完成目录重新 `--resume`：四条已完成路线校验后跳过，诊断校验后复用，没有新增优化步；
  完成后 tmux 再次自动退出。
- 单独的无效参数检查输出到 `failure_lifecycle_check_20260913`：写入 failed 心跳，
  记录退出码 2，tmux 自动退出，没有留下等待初始化的监控进程。
- `launch.sh --plan-only` 不启动 tmux、不创建输出目录。
- 原实验源码与结果未修改；原有两个根目录 Markdown 改动仍保持其原状态。
- 默认正式目录 `outputs/grouped_control_diagnostics/dsprites_l4_seed0` 未创建。

使用 README 中不带缩短预算参数的默认 launcher 才会开始完整实验。
