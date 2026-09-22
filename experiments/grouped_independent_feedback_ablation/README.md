# Independent 有反馈／无反馈五种子消融

本实验评价正常、无人纠正时的 label 预测：Independent 的测量记录反馈，
是否优于从相同初始化单独训练的无反馈电路。只读复用已有模型，独立保存结果，
没有训练阶段，不访问测试集。

## 实验定义

| 项目 | Independent 有反馈 | 无反馈对照 |
|---|---|---|
| 前段 | 各种子已训练的 Fusion L4，240 参数冻结 | 同一种子的相同前段 |
| 概念测量 | 前五位，Shape 两位、Scale 三位 | 保留相同测量 |
| 后段训练 | 真实概念控制 X，保留实际测量分支及 B5 状态 | 不执行条件 X，保留相同分支及 B5 状态 |
| 正常评价 | 实际测量记录控制 X | 不执行条件 X |
| 后段电路 | B5＋读出位，L1，24 参数 | 相同结构及参数量 |

两组都保留中间测量。本消融衡量测量记录驱动的条件反馈模块，不单独归因于测量坍缩。
无反馈对照在关闭反馈条件下完整训练过，不是临时关闭 Independent 模型中的门。
无反馈不读取概念记录，因而预测／真实记录的训练区别对它不起作用，可以作为共用基线。

前段继续复用 `FusionModel.TQLayer` 和 TorchQuantum；居中、Pool(10,4)、
训练集方差路由、每层四值上传、U3 与环形 CU3 都来自已有锁定模型。
评价保留全部 32 个 Born 分支，包括非法概念码；不取 MAP 控制，不重编码。

## 默认数据与来源核验

- 种子 0、1、2、3、4；25,593 张训练图片，5,479 张验证图片。
- Independent：`outputs/grouped_independent_intervention/dsprites_l4_five_seeds/`。
- 无反馈：上述实验关联的 `grouped_feedback_ablation/dsprites_l4_paired/`
  下 `sequential/seedN/no_feedback/`。
- 逐种子核验初始权重、冻结前段、数据索引、样本顺序、训练预算、模型权重及 Adam 更新数。
  后段都是 100 轮、2,500 次更新，Adam lr=.01、batch=1024、梯度裁剪 5。
- 只使用固定最终轮模型。评价条件必须是 `measured`／`zero`，不会导入全部概念纠正后的成绩。
- 默认对兼容预测文件核验哈希并复制，再从逐图片概率重新计算所有指标。
  `--recompute`、不同 shots、不同评价 batch 或开发子集会触发冻结模型 CUDA 重算。
- 强制使用当前 VQC CUDA 环境；重算继续调用已有量子模型与联合 `(m,y)` 采样实现。
  前段状态按种子缓存，同一种子的两组评价共用；不改变源模型、日志、心跳或结果。

## 一次启动

在仓库根目录运行：

```bash
# 只读核验来源并显示计划，不创建本实验输出
experiments/grouped_independent_feedback_ablation/scripts/run.sh --plan-only

# 默认五种子，复用已核验预测并生成独立报告
experiments/grouped_independent_feedback_ablation/scripts/launch.sh

# 查看状态、连接或跟踪日志
experiments/grouped_independent_feedback_ablation/scripts/status.sh
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_independent_feedback_ablation
tail -f outputs/grouped_independent_feedback_ablation/dsprites_l4_paired/evaluation.log

# 恢复原配置；完成条件核验后跳过
experiments/grouped_independent_feedback_ablation/scripts/launch.sh --resume

# 在独立目录强制 CUDA 重算全部十个模型条件
experiments/grouped_independent_feedback_ablation/scripts/launch.sh \
  --session independent_feedback_replay \
  --out outputs/grouped_independent_feedback_ablation/dsprites_l4_cuda_replay \
  --recompute
```

程序正常完成、暂停或失败后 tmux 自动退出；结果及日志保留。
`--max-conditions N` 在新增 N 个评价条件后暂停；不完整种子配对不参与均值或标准差。
每个条件有独立锁，恢复要求配置、源码、运行环境及来源文件一致。
同一输出目录有排他进程锁。开发子集必须显式指定 `--development --val-limit N`。

## 输出与解释

默认输出：`outputs/grouped_independent_feedback_ablation/dsprites_l4_paired/`。

- `summary.md/json`：正常 label 准确率、有限 shots 准确率及配对增益，均值±样本标准差。
- `condition_results.csv`：逐种子、逐方案的准确率、平衡准确率、BCE、概念指标和有限 shots 指标。
- `paired_results.csv`：有反馈减无反馈的配对差值、错→对／对→错计数；保留所有正负结果。
- `feedback_comparison.png/pdf/svg`：精确概率、有限 shots 的配对准确率与逐种子增益。
- `pairing_audit.json`：各对模型的初始化、前段、训练更新、顺序及评价配对证据。
- `seedN/{feedback,no_feedback}/`：核验复用或重新计算的逐图片预测、指标与锁。
- `manifest.json`、`reference_lock.json`、`result_lock.json`：代码、配置、来源与报告哈希。

五个训练种子是重复单位，不把验证图片数当作独立训练重复。
256 shots 是每张图片按模型自身联合分布采样 256 次；不代表带测量噪声训练或真实硬件实验。
两种方案的原始概念分布必须完全一致；准确率变化用于评价后续反馈方案。

## 检查

```bash
experiments/grouped_independent_feedback_ablation/scripts/check.sh
```

执行 Ruff、ty、Pylint E/F/W 和实际 CUDA 测试，覆盖复用／重算一致性、半对结果恢复、
错配拒绝、源文件保留和预测文件篡改检测。需要已有完整训练来源。
