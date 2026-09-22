# 工程验证记录

日期：2026-09-18。环境：`/root/miniforge3/envs/VQC`，NVIDIA GeForce RTX 4090。

## 自动检查

`bash experiments/grouped_robot_shots_final/scripts/check.sh` 全部通过：

- Ruff 检查与格式检查、ty、项目 Pylint 质量检查。
- 三项实际 CUDA 测试：`3 passed, 1 warning in 87.76s`；warning 来自既有 PyTorch/pynvml 环境。
- 四个 shell 脚本另行通过 `bash -n`。

测试覆盖：

1. 全部 32 种五位记录的位序、概念与 label 联合抽样的相关性、嵌套 shots 前缀、固定随机种子的确定性重放、异常计数拒绝。
2. 两个训练种子、两种 shots、两次抽样重复、两个阶段共 36 个条件；禁止创建 Adam；既有模型与预测只读；验证阶段结束前不开启最终评价入口；暂停恢复与连续执行保存的概率和计数相同；指标先在种子内平均，再跨种子计算样本标准差；配置或结果改变时拒绝恢复。
3. 合成数据的正式测试读取入口：没有验证结果锁时不读取测试文件；原始观测 label 保留；复用训练方差排列、居中、pool 和一次 π 缩放；测试身份与开发身份重叠时拒绝执行。

合成数据测试不访问真实 Robot 测试集。

## 真实五种子来源预检

执行默认 `scripts/run.sh --plan-only`，通过完整四模式五种子来源核验：

`outputs/grouped_robot_four_modes/robot_v5_l4_head5_ep600_five_seeds/`

确认冻结模型、训练来源、原始 train/validation 数据、历史逐样本概率及 CUDA 环境符合协议。预检不创建正式输出，不读取真实测试集，不进行优化更新。

## 实际 Robot 验证样本与 tmux

输出：`outputs/grouped_robot_shots_final/engineering_cuda_20260918/`。

引用已完成的 `outputs/grouped_robot_four_modes/engineering_cuda_20260917/`；这些仅是先前短训练的工程模型。本轮配置为 seed 0、验证子集 64 张、shots 8/16、抽样重复 2 次、`development=true`。第二阶段使用验证数据构成 `test_proxy`。

- 先用 `--max-conditions 1` 完成一个条件并暂停，然后通过 tmux `--resume` 完成全部 18 个条件。
- 前段量子演化、后段读出与联合抽样在实际 RTX 4090 上完成；新程序不创建优化器或修改模型。
- 两个阶段均生成 CSV、Markdown、JSON、PNG/PDF/SVG、逐样本概率和抽样计数，以及结果锁。
- 独立复核 validation 67 个产物、test_proxy 67 个产物、根目录 9 个产物、417 个源文件及 232 个引用产物哈希，全部一致。
- 完成心跳为 `18/18`，`test_read=false`、`test_evaluated=false`，无真实 `test/` 目录。
- 暂停和恢复完成后，tmux 会话均自动退出；日志保留。
- 默认正式输出目录尚未创建。

这份记录仅验证程序和运行流程。正式性能需执行默认五种子、完整验证与测试划分、五种 shots、十次重复的评价后报告；工程模型的准确率不用于性能结论。
