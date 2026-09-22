# 工程验证记录

日期：2026-09-18。Python 环境：`/root/miniforge3/envs/VQC`；GPU：NVIDIA GeForce RTX 4090。

## 自动检查

`bash experiments/grouped_robot_intervention_curve/scripts/check.sh` 全部通过：Ruff、格式检查、ty、项目 Pylint 质量检查、CUDA 测试。

测试结果：`3 passed, 1 warning in 67.60s`。warning 为既有 PyTorch/pynvml 弃用提示。四个 shell 脚本另行通过 `bash -n`。

1. 穷举子集数量与位序、先对每个组合分类评分再平均、组合不充当额外种子、均值／样本标准差、非单调结果保留、缺失和重复组合拒绝。
2. 实际四层前段和五层后段电路在 CUDA 上运行；对多个部分纠正掩码逐分支施加实际 X 门，直接演化后半段，与新评价程序的概率质量一致；原 Born 概率不变，模型参数不变。
3. 两种子、三模式、192 个条件的集成测试：禁止创建 Adam；12 个端点与历史结果逐项相同；180 个部分条件完成；暂停恢复与连续执行的所有概率张量及汇总一致；完成后恢复不重写产物；历史来源内容与时间戳不变；改变配置、输出重叠及篡改文件被拒绝。

集成测试使用临时合成数据和已有测试夹具，不对实际 Robot 模型进行新训练。

## 正式来源预检

默认 `scripts/run.sh --plan-only` 已通过，确认三模式五种子的冻结模型、原正式测试缓存 6,144 张、模型和数据来源、CUDA 运行环境符合既定协议。预检没有产生新的测试预测或创建默认正式输出目录。

本实验补齐已经规划过、此前遗漏的数量曲线；此前正式测试结果已经查看，不把本次工作描述为新的未见测试集确认实验。

## 实际 Robot 验证子集与 tmux

输出：`outputs/grouped_robot_intervention_curve/engineering_cuda_20260918/`。

引用当前完整正式模型来源，使用 `development=true`、seed 0、验证样本 64 张。新评价的角色为 `validation_proxy`，不产生正式测试曲线。

- 先完成三个组合后暂停，再使用 tmux `--resume`。
- 三模式全部完成，共 96 个条件：6 个历史端点、90 个新增部分纠正条件。
- CUDA 推理成功；心跳最终为 `complete`、`96/96`、`test_read=false`、`test_evaluated=false`。
- CSV、JSON、Markdown 与 PNG/PDF/SVG 曲线均已生成，图片排版已检查。
- 独立复核 299 个输出产物、431 个源文件、1,634 个历史引用产物哈希，全部一致。
- 暂停后和完成后 tmux 会话均自动退出；日志保留。
- 原有代码未修改，默认正式输出目录尚未创建。

工程子集结果不用于性能结论。正式实验需执行默认命令，完成五种子、完整测试集的 480 个组合条件，再解释正式曲线。
