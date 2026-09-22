# 2026-09-16 工程验收

- `scripts/check.sh`：Ruff、格式检查、ty、Pylint 通过；两个 CUDA 集成测试通过（38.81 秒）。
- 对照原概念训练循环，模型、Adam、训练历史、RNG 逐项一致。
- 高斯组在半个 epoch 后恢复，与连续训练的模型、Adam、历史、RNG 一致。
- 旧实验文件哈希和修改时间不变；完成结果恢复不会重写产物；篡改预测或样本排列会被拒绝。
- 真实 Robot 工程运行：`outputs/grouped_robot_concept_init/cuda_check_20260916/`。
- 每个单元取 train/validation 各 64 张、训练 2 轮，六个单元、十二项概念评价全部完成；共 12 次 Adam 更新。
- CUDA 设备为 RTX 4090；前半段 240 个参数参与训练，后半段不接收梯度。
- 原输入和代码引用核验通过；`same_sample_orders_verified=true`，`test_read=false`，`test_evaluated=false`。
- 已验证 Rich/tmux 启动、状态与日志输出，完成后 tmux 自动退出。

以上为工程验证，不是正式 300 轮实验结果。正式结果目录为 `outputs/grouped_robot_concept_init/robot_v5_l4/`。
