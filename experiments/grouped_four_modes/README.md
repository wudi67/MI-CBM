# dSprites 四种训练模式补齐

独立目录，复用原有电路、训练器和评估实现，不改写旧代码或旧实验输出。

## 默认运行内容

| 模式 | 新训练 | 复用 | 正常预测时的概念控制 |
|---|---|---|---|
| Standard | seeds 1–4，各 200 epochs，仅 label BCE | seed 0 | 实际测量记录；未进行概念监督 |
| Joint 有反馈 | seeds 1–4，各 200 epochs，概念 NLL + label BCE | seed 0 | 实际测量记录 |
| Joint 无反馈 | seeds 1–4，各 200 epochs，同样的 Joint 损失 | seed 0 | 全零控制，即不执行反馈 X 门 |
| Independent | 无 | 已完成的五种子模型和评估 | 实际测量记录 |
| Sequential | 无 | 已完成的五种子模型和评估 | 实际测量记录 |

总计 **12 个新训练任务**。每种子三条新路线使用同一份初始权重和相同的逐 epoch 样本顺序；不根据精度选择种子、模式或最佳 epoch。

- 原有中心化、Pool projector、训练集方差分配不变：40 个数值 → 10 个 qubit，每个上传 4 个数值。
- 前半段固定 4 层；每层继续使用原来的 Fusion 数据 re-uploading 和 U3/CU3 电路实现。后半段沿用 B5 + 额外 readout qubit，264 个可训练参数。
- Adam，lr=0.01，batch=1024，eval batch=2048，梯度裁剪 5；默认每任务 5000 次更新。强制 CUDA，不会静默退回 CPU。
- 每任务固定第 200 epoch 的结果。Independent/Sequential 原预算是概念阶段 100 epochs + 分类阶段 100 epochs，因此相同总 epoch 不意味着各部分参数更新次数相同。
- Independent/Sequential 使用本项目已有定义，保留后五个 qubit 的输入相关信息；没有改成严格的经典概念瓶颈。

## 一次启动，顺序执行

在仓库根目录执行：

```bash
bash experiments/grouped_four_modes/scripts/launch.sh
```

程序顺序完成：**补训 → 验证集统一评价 → 固定模型与评价规则 → 测试集评价 → 四模式汇总**。tmux 的训练和状态窗格会在完成后自动退出；日志、模型、结果保留。

```bash
# 查看运行情况
bash experiments/grouped_four_modes/scripts/status.sh

# 进入 tmux
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_four_modes

# 实时日志
tail -f outputs/grouped_four_modes/dsprites_l4_five_seeds/evaluation.log

# 中断后恢复：保留模型、Adam、随机状态和 epoch 内进度
bash experiments/grouped_four_modes/scripts/launch.sh --resume

# 只校验来源并展示计划，不训练、不评价测试集
bash experiments/grouped_four_modes/scripts/run.sh --plan-only
```

源文件、配置、原实验模型或数据发生变化时会拒绝续跑。正常续跑只需 `--resume`，不要改变超参。进程和输出目录有锁，不能同时启动两个写入同一输出的任务。

## 评价与结果文件

每个种子、每个 split 共 15 个条件。Independent/Sequential 及其已有无反馈对照共 9 个条件直接复用历史评估的概率和采样数据；新增 6 个条件为 Standard 正常预测、Joint 的正常预测/纠正 shape/纠正 scale/全部纠正，以及 Joint 无反馈。

- 正常 label accuracy、balanced accuracy、BCE。
- 概念的 shape/scale 分组准确率、两组同时正确率、联合 MAP、真实组合单次测量概率、非法编码率。
- Independent、Sequential、Joint 的概念纠正效果，保留上升和下降结果。
- Joint 有/无反馈使用各自 Joint 训练的模型；不会拿 Joint 与仅训练后半段的旧无反馈模型当作对应消融。
- shots=64、128、256、512、1024，每种设置 10 次采样；按训练种子汇总均值和样本标准差，不把采样重复次数当作训练种子数。
- Standard 没有概念监督，概念数值仅供诊断，主表中为 N/A，不进行有语义的概念纠正。

输出目录：`outputs/grouped_four_modes/dsprites_l4_five_seeds/`。

| 文件 | 内容 |
|---|---|
| `summary.md` | 总入口 |
| `training_lock.json` | 三路线五种子的模型、初始化、训练步数及来源 |
| `test/summary.md` | 四模式结果与消融、干预效果 |
| `test/four_modes_main.csv` | 四模式主表 |
| `test/condition_results.csv` | 各种子、模式、控制方式和 shots 的全部指标 |
| `test/paired_results.csv` | 按同一种子配对的反馈及概念纠正变化 |
| `test/sampling_repeats.csv` | 每次采样的结果 |
| `test/correction_count_results.csv` | 纠正 0/1/2 个概念；1 个为分别纠正 shape、scale 的等权平均 |
| `test/four_modes_shots.*` | 四模式 shots 曲线，PNG/PDF/SVG |
| `test/four_modes_interventions.*` | 概念纠正曲线，PNG/PDF/SVG |

这里补充的是**已经使用过的同一个测试划分**，不能声称这是一个从未查看过的新测试集。新增路线采用既定预算，先完成验证再自动测试，不依据结果决定是否纳入 Joint。

有反馈和无反馈两边都保留中途测量。因此该消融检验的是“测量记录控制 X 门”的贡献，不能单独归因于测量坍缩。

## 工程检查

```bash
bash experiments/grouped_four_modes/scripts/check.sh
```

包括 Ruff、ty、Pylint 和 CUDA 测试：Joint 在 epoch 内中断后恢复与连续训练一致；完整 Adam 状态一致；恢复不重写已完成条件；历史结果按原采样数据复用；Standard 的更新不依赖概念目标；Joint 消融使用正确的对应模型。

开发检查只使用训练/验证子集，第二阶段是 `test_proxy`，不会读取真实测试图片。显式开发命令如下，结果不可当作正式精度：

```bash
bash experiments/grouped_four_modes/scripts/launch.sh \
  --session grouped_four_modes_dev \
  --out outputs/grouped_four_modes/development_check \
  --development --seeds 0,1 --epochs 2 \
  --train-limit 36 --val-limit 18 --batch-size 18 --eval-batch-size 18 \
  --shots 8,16 --repeats 2 --checkpoint-steps 1
```
