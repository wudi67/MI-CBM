# 实现检查记录

检查日期：2026-09-15。正式的 seed 0、每阶段 100 epochs 预实验尚未启动。

## 静态检查和测试

执行 `bash experiments/grouped_robot_pilot/scripts/check.sh`：Ruff、格式检查、ty、
Pylint 全部通过；pytest **6 passed**。

测试覆盖全部五位控制映射、实际 TorchQuantum X 门对照、灰度保持与训练集方差路由、
概念训练不依赖 label、CUDA 梯度与冻结参数，以及中途恢复后的模型和完整 Adam 状态
与连续训练一致。测试数据没有测试集文件，完整流程仍可运行。

## 当前源码的真实数据检查

输出：`outputs/grouped_robot_pilot/development_check_final/`。

```bash
bash experiments/grouped_robot_pilot/scripts/launch.sh \
  --session grouped_robot_pilot_check \
  --out outputs/grouped_robot_pilot/development_check_final \
  --development --concept-epochs 2 --head-epochs 2 \
  --train-limit 1024 --val-limit 2048 \
  --batch-size 1024 --eval-batch-size 2048 --checkpoint-steps 1
```

- 使用当前 Robot v5 的真实图像和原始标签。
- 在完整 18,432 张训练图像上拟合 40 维方差排列；优化使用 1,024 张训练图像，
  评价使用 2,048 张验证图像。没有读取测试 CSV 或测试图片。
- RTX 4090 上三个阶段均完成前向和反向传播；实际训练 batch 1024，评价 batch 2048。
  输入、训练参数和有效梯度均在 `cuda:0`，冻结部分没有梯度。
- 三个阶段各两次 Adam 更新，八个验证条件全部完成，曲线和 CSV/JSON 原始结果已生成。
- 概念阶段峰值 CUDA tensor allocation 约 0.721 GiB；后半段各约 0.142 GiB。
  这是该开发子集的 PyTorch 分配量，不是全量正式训练或设备总显存峰值。
- 正常结束后，`grouped_robot_pilot_check` tmux 会话自动退出。
- 对已完成目录执行 `scripts/run.sh --out ... --resume` 成功：40 个结果及锁文件的
  SHA-256 和修改时间全部保持不变，模型、Adam 和预测文件未被重新生成。

两轮工程结果仅验证程序可运行，不能用于判断模型性能。正式预算、数据范围和启动命令见
[README.md](README.md)。较早的 `development_check/` 目录是修改纠正条件名称之前的开发检查，
不应与当前源码混用或当作正式结果。
