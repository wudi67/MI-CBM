# dSprites：验证集 shots 曲线 → 固定协议 → 正式测试

这是独立的评价程序，复用已经完成训练的 Fusion L4 模型。默认一次启动，顺序执行两个阶段，不重新训练，不修改历史实验代码、权重或结果。

## 默认实验

- 训练种子：0、1、2、3、4。
- 测量预算：64、128、256、512、1024 shots／图片／条件，另计算精确概率。
- 每个模型进行 10 次独立抽样重复；同一次重复的五个预算使用同一条采样序列的不同长度前缀。
- 每个种子有 9 个条件：Independent 的不纠正／Shape／Scale／全部纠正，Sequential 的四个对应条件，以及单独训练的无反馈模型。
- 两阶段共 90 个条件。两种训练方式共用已经单独训练过的无反馈模型。
- 评价 batch 默认 2048；强制 CUDA，复用 TorchQuantum、FusionModel.TQLayer 和既有量子分类电路。
- 同一种子、同一划分的前段状态只计算一次，后续条件复用。先计算分支联合概率，再抽样，不为每个 shot 重做量子态演化。

第一阶段使用全部 5,479 张验证图片，检查新推理重现历史逐图片精确概率，并评价多个 shots。所有条件完成后，保存验证结果锁和 `protocol_lock.json`，自动进入第二阶段。第二阶段使用原划分中的全部 5,477 张测试图片，执行完全相同的矩阵。模型、种子、预算、指标和抽样规则在启动时固定；转入测试不以验证准确率高低为条件。

## 一条命令顺序执行

在仓库根目录运行：

```bash
experiments/grouped_shots_final/scripts/launch.sh
```

该命令启动后台 tmux：验证 → 协议固定 → 测试 → 汇总。正常完成、暂停或失败后，worker 和监控窗口都会退出，结果及日志保留。

```bash
# 查看进度
experiments/grouped_shots_final/scripts/status.sh

# 连接界面；Ctrl+b 然后 d 可以离开，任务继续运行
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_shots_final

# 跟踪日志
tail -f outputs/grouped_shots_final/dsprites_l4_five_seeds/evaluation.log

# 中断以后续跑：自动读原配置，核验并跳过已完成条件
experiments/grouped_shots_final/scripts/launch.sh --resume

# 只读核查模型、配对和 CUDA；不创建正式结果、不评价测试集
experiments/grouped_shots_final/scripts/run.sh --plan-only

# 不使用 tmux，在当前终端顺序执行
experiments/grouped_shots_final/scripts/run.sh
```

`--max-conditions N` 表示完成 N 个新条件后暂停；恢复时不把已有条件算入 N。在验证阶段恰好达到上限时，先保存验证报告，再暂停，测试入口保持未开启。已完成条件按数据、配置、代码、模型和观测文件核验。信号中断可能重做当前未完成条件；已缓存的联合概率可以复用。一个输出目录同时只允许一个 worker。

## 指标和统计

- label：准确率、平衡准确率、BCE；决策阈值固定为 `>= 0.5`，平票判为 1。
- concept：Shape／Scale 边缘概率 argmax、两个边缘预测同时正确、联合 MAP、真实完整记录单次出现概率、非法码概率等分别命名。
- 概念读出保留全部 32 个码，不过滤非法码；argmax 平票采用最低索引，非法预测记错。
- 反馈收益：相同训练种子下，有反馈减无反馈。概念纠正收益：纠正后减不纠正。
- 每个种子先对 10 次抽样评价取平均，再报告五个训练种子的均值 ± 样本标准差。CSV 单独保留抽样波动；50 次抽样评价不冒充 50 个训练种子。
- 两个训练模式及无反馈的前段相同。正式概念表和曲线只报告一次共享前段，有限 shots 使用 Independent／不纠正的测量样本；不同条件的抽样编号独立，不应把抽样差异解释成前段改变。
- 纠正概念数曲线中，纠正 1 个概念的准确率取 Shape、Scale 两个条件的平均，四个具体条件同时保留。

每一次 shot 抽取完整的 `(原始测量记录 m, label y)` 联合结果。纠正仅替换控制 X 门的经典记录，保留原始分支权重和该分支的 B5 状态。正常推理始终使用预测记录；真实概念只在明确标注的干预条件中使用，不计作概念预测成功。反馈消融的两边都保留中间测量。

实验范围是理想模拟器中、冻结模型的有限测量评价；它不包含有限 shots 训练或硬件门噪声。

## 输出

默认目录：`outputs/grouped_shots_final/dsprites_l4_five_seeds/`。

- 根目录：`summary.md/json`、`manifest.json`、`reference_lock.json`、`protocol_lock.json`、`test_access.json`、`result_lock.json`、`heartbeat.json`、`evaluation.log`。
- `validation/` 和 `test/`：各自的汇总、CSV、图形、数据缓存及结果锁。
- CSV：`condition_results.csv`、`sampling_repeats.csv`、`paired_results.csv`、`concept_results.csv`、`correction_count_results.csv`。
- 图形：`classification_shots`、`feedback_shots`、`intervention_shots`、`concept_shots`，均输出 PNG／PDF／SVG；阴影为训练种子间标准差，虚线为精确概率参考。
- 每个条件：`joint.pt` 保存全部原始分支联合概率；`samples.pt` 保存每次重复、每个预算、每张图片的计数和预测；`evaluation.json` 保存由它们重算的指标。锁文件绑定对应模型和数据。

## 开发检查

```bash
experiments/grouped_shots_final/scripts/check.sh

# 小规模检查完整衔接、作图和 tmux；第二阶段使用验证图片作为软件测试替身
experiments/grouped_shots_final/scripts/launch.sh \
  --session grouped_shots_smoke \
  --out outputs/grouped_shots_final/development_smoke \
  --development --seeds 0,1 --val-limit 18 \
  --shots 8,16 --repeats 2 --eval-batch-size 18
```

开发模式的第二阶段目录是 `test_proxy/`，始终标记 `test_evaluated=false`，不会预处理或评价真实测试图片。实际测试读取路径由合成数据单元测试覆盖。正式评价禁止删减训练种子或使用验证子集。开发结果不能填入论文主表。
