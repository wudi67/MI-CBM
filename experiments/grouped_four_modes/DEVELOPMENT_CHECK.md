# 开发验证记录（2026-09-15）

本记录是工程验证，不是正式模型精度结果。正式五种子长训练尚未启动。

- `scripts/check.sh` 通过：Ruff check/format、ty、Pylint，以及 **5 个 CUDA 测试**（59.02 秒）。
- 设备：NVIDIA GeForce RTX 4090；PyTorch 2.11.0+cu130。
- 默认 batch=1024 已实际执行 Standard、Joint 有反馈、Joint 无反馈各一次梯度更新。所有参数和有效梯度均在 CUDA，两个电路部分均有非零梯度，未发生显存不足。
- 中途恢复测试：Standard 完成后，在 Joint 第一个 epoch 的第一个 batch 后暂停；恢复后的模型权重、完整 Adam 状态、训练历史，与连续运行一致。
- Standard 的一次参数更新在更换概念目标后保持完全一致，验证其只使用 label BCE 训练。
- 原来 Standard、Joint 有反馈、Joint 无反馈的 seed 0 在全量验证集重新推理，均重现原训练终点指标，容差 2e-6。
- Independent 的历史验证概率和有限 shots 采样按文件哈希原样复用；没有重新采样后替换历史值。
- 独立开发数据上完成两种子、三路线、各两 epochs，以及两个评价阶段共 60 个条件；第二阶段为验证数据代理 `test_proxy`，没有读取真实测试图片。
- 主表、全部 CSV、PNG/PDF/SVG 曲线已生成。
- 通过真实 tmux 启动，结束后训练窗格与状态窗格自动退出。`--resume` 再次运行只校验已有结果。
- 原实验被引用的文件哈希保持一致；新代码只位于 `experiments/grouped_four_modes/`。

可检查的开发输出：`outputs/grouped_four_modes/release_check_20260915/`。其中的两 epoch、18 条验证样本结果不能用于论文性能结论。

正式启动命令见 [README.md](README.md)。
