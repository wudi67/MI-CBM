# 初始化对照代码验收

正式的 300 轮初始化比较尚未启动。下列检查用于验证代码和运行流程。

## 自动检查

执行 `bash experiments/grouped_robot_label_init/scripts/check.sh`：

- Ruff 检查与格式检查通过。
- ty 类型检查通过。
- 项目 Pylint 检查通过。
- pytest：3 passed，97.70 秒；1 条第三方 pynvml 弃用提醒。

测试使用真实 CUDA 电路和合成 Robot 图片，覆盖九个单元、36 个评估条件、三个历史基线的复用、固定初始角度分布与成对读出偏移、冻结前半段、源文件内容及时间戳不变、批次中断恢复后权重/Adam/进度/RNG 与连续训练完全一致，以及结果篡改和错误恢复配置的拒绝。

## 真实 Robot 数据短跑

输出：`outputs/grouped_robot_label_init/cuda_check_20260916/`。

使用 `--development --head-epochs 2 --diagnostic-every 1`，保留全部真实训练/验证样本、batch=1024 和 eval batch=2048。

第一次经 tmux 启动时添加 `--max-steps 1`，核验在 epoch 0、Adam step 1、样本位置 1024 保存并退出，tmux 自动结束；随后使用 `--resume` 完成剩余工作。

- RTX 4090 上九个单元全部完成，每单元 2 轮、36 次 Adam 更新，共 324 次更新。
- 36 个 train/validation 评估条件完整，未增加逐 concept 干预条件。
- 所有单元的五层均有有限、非零 CUDA 梯度；前半段梯度为 None。
- 九个单元训练样本顺序一致。
- 初始偏置检查仅用固定训练 batch，没有优化器更新。
- 逐样本概率重新计算的指标与保存结果一致，汇总均值、标准差和配对差值复算通过。
- 当前结果、代码以及整个历史引用链的文件指纹复核通过。
- `test_read=false`、`test_evaluated=false`。
- 最终 heartbeat 为 complete，训练进程退出，tmux 自动退出。
- `comparison.png` 已生成并检查布局。

工程预算与历史 300 轮不同，因此九个单元均从初始参数训练，没有混入不同预算的历史端点。这些两轮准确率不能用于判断新初始化是否优于当前模型。

默认 `--plan-only` 已验证：三组各三个初始化、固定 5 层/112 参数、σ=0.0033333、300 轮；复用三个均匀初始化基线，只新增六次训练，共 32,400 次 Adam 更新。

默认正式输出 `outputs/grouped_robot_label_init/robot_v5_l4_head5/` 在验收结束时尚未创建。
