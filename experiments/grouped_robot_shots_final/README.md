# Robot：验证集有限 shots → 固定评价记录 → 正式测试集

独立的评价程序，复用已完成的 Robot 五种子四模式结果及匹配对照，不创建优化器，不重新训练。旧代码、数据、模型、概率文件及报告全部只读。

## 默认任务

| 设置 | 内容 |
|---|---|
| 模型 | 固定四层概念电路 + 五层 label 电路，原始初始化对应的已训练最终 checkpoint |
| 种子 | 0、1、2、3、4 |
| 测量次数 | 64、128、256、512、1024 shots／图片／条件 |
| 参考值 | 精确概率；文件中的 `shots=0` 仅是精确结果标记 |
| 抽样重复 | 每种设置 10 次；一次重复的各预算使用同一抽样序列的不同长度前缀 |
| 推理与采样 | CUDA，评价 batch 默认 2048；量子演化继续调用原 TorchQuantum/FusionModel 电路 |
| 划分 | 完整验证集 6,144 张 → 完整测试集 6,144 张 |
| 训练 | 无；所有模型冻结，保留全部既定种子和正负收益 |

每个种子、每个划分有九个条件：

1. Standard 正常预测。
2. Independent 正常预测。
3. Independent 全部五个概念纠正。
4. Sequential 正常预测。
5. Sequential 全部五个概念纠正。
6. Independent／Sequential 对应的已训练无反馈模型。
7. Joint 正常预测。
8. Joint 全部五个概念纠正。
9. Joint 对应的**重新联合训练过的**无反馈模型。

两阶段共 **90 个条件**。每个条件包含精确概率以及五种 shots、各十次重复的结果。Standard 没有概念监督，不做语义概念纠正；其概念读出仅作诊断并另存，主表概念列为空。

## 电路与采样规则

- 每个模型使用自己的前半段和后半段参数。仅当前半段参数哈希相同时共享缓存量子态；Independent／Sequential／原无反馈的相同前半段可以复用，Standard、Joint、Joint 无反馈分别计算。
- 保留全部 32 个合法的五位二元概念记录及对应后五位条件量子态。概念纠正只替换控制 X 门的经典记录，保留原始分支、权重及反馈前条件量子态，不重新编码。
- 无反馈仍保留中间测量，消融的是测量记录反馈。
- 对每张图先计算完整的 `(原始测量记录 m, 最后读出 y)` 联合概率，再调用原 dSprites 使用的 `sample_shots` 在 CUDA 上抽样。每次抽样同时得到概念记录和 label，保持二者关联；无需为每个 shot 重复量子态演化。
- label 使用读出 `1` 的频率，阈值 `>=0.5`，平票判为 `1`。五个概念分别按位频率 `>=0.5` 预测，联合 MAP 取最多次出现的完整记录，平票取最低编号。
- 正常预测不以联合 MAP 替换每次测量的反馈控制。有限 shots 只改变观测的统计估计。
- 真实概念只在明确标记的纠正条件中替换控制记录，不计作模型自行预测概念成功。

本实验考察冻结的理想电路在有限测量次数下的表现，不包含有限 shots 训练或硬件门噪声。

## 两阶段衔接

启动时固定模型、种子、shots、重复次数、指标、阈值和抽样规则。

第一阶段重新推理验证集，核对每张图片的概念分支概率和分支 label 概率质量与历史结果一致，再进行 shots 采样。完成所有条件和报告后保存验证结果锁及 `protocol_lock.json`。

第二阶段自动进入测试集，入口不依赖验证准确率或收益的高低。测试图片继续使用原居中方法、`AdaptiveAvgPool(10,4)`、训练集拟合的方差排列及一次 π 缩放；不重新拟合。读取原始观测 label，不依据概念规则重新生成 label。检查原 Robot 身份划分不重叠、每个身份四个渲染、完整样本数及图像路径唯一性。

测试 CSV 和测试图像仅在验证评价完整锁定后读取；读取后绑定 CSV、图像和预处理哈希，断点恢复会核验。`--plan-only`、`--preflight-only` 和开发模式不会读取真实测试集。

## 运行命令

在仓库根目录启动一次，自动顺序完成两个阶段：

```bash
bash experiments/grouped_robot_shots_final/scripts/launch.sh
```

默认 tmux 会话 `grouped_robot_shots_final`，上方日志、下方 Rich 状态。完成、暂停或失败后自动退出 tmux；文件和日志保留。

```bash
# 查看进度
bash experiments/grouped_robot_shots_final/scripts/status.sh

# 进入 tmux；Ctrl-b 然后 d 可以离开查看，程序继续执行
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_robot_shots_final

# 跟随日志
tail -f outputs/grouped_robot_shots_final/robot_v5_l4_head5_five_seeds/evaluation.log

# 程序已经停止时恢复；不要重复启动健康运行中的任务
bash experiments/grouped_robot_shots_final/scripts/launch.sh --resume

# 只读核验实际来源与 CUDA，不创建输出、不读取测试集
bash experiments/grouped_robot_shots_final/scripts/run.sh --plan-only

# 保存来源预检后暂停；之后 --resume 才开始评价
bash experiments/grouped_robot_shots_final/scripts/run.sh --preflight-only
```

默认来源：`outputs/grouped_robot_four_modes/robot_v5_l4_head5_ep600_five_seeds/`。默认输出：`outputs/grouped_robot_shots_final/robot_v5_l4_head5_five_seeds/`。

`--max-conditions N` 在本次调用新增完成 N 个条件后暂停；恢复时验证并跳过已有条件。若恰好完成验证阶段，先保存验证报告再暂停，测试入口保持关闭。SIGINT／SIGTERM 请求安全暂停；已经完成的条件不会重写，未完成条件可复用精确联合概率，重新执行确定性的采样。源代码、配置、模型、数据或结果被修改时拒绝恢复。

开发模式使用验证集子集作为第二阶段 `test_proxy`，始终标记 `test_read=false`、`test_evaluated=false`：

```bash
bash experiments/grouped_robot_shots_final/scripts/launch.sh \
  --session robot_shots_smoke \
  --out outputs/grouped_robot_shots_final/development_smoke \
  --development --seeds 0 --val-limit 64 --shots 8,16 --repeats 2
```

开发结果只验证程序通路，不用于正式性能结论。正式评价禁止缩减种子、划分或预设 shots／重复次数。

## 指标、统计和结果文件

label 指标：准确率、平衡准确率、BCE。概念指标：平均单概念准确率、五个概念全部正确率、联合 MAP、真实完整概念记录的单次出现概率和联合 NLL。有限 shots 的 BCE／NLL 采用既有 `1e-7` 概率下限；它们包含有限采样估计误差。

每个训练种子先对十次抽样结果取平均，再对五个训练种子报告均值与样本标准差（n−1）。单种子的采样波动单独保存，五十次抽样评价不作为五十个训练种子。配对收益在同一种子内计算，再跨种子汇总。

Independent／Sequential 的前半段相同，概念主表和曲线只报告一次共享前半段，以 Independent 正常预测的样本为准；四模式主表的 Sequential 概念列明确标记该来源。各条件原始采样指标均保留，避免把独立抽样的差异解释成前半段改变。

根目录保存入口报告、运行配置、来源锁、验证协议锁、测试访问记录、心跳和日志。`validation/`、`test/` 各自保存：

- `summary.md/json`：九个条件及各 shots 的汇总。
- `four_modes_main.csv`：四种模式正常预测的逐种子结果。
- `condition_results.csv`：精确值和每个种子内平均后的有限 shots 指标。
- `sampling_repeats.csv`：全部独立抽样重复，保留采样种子。
- `paired_results.csv`：匹配反馈、全部纠正及模式比较的逐种子差值。
- `concept_results.csv`、`standard_concept_diagnostics.csv`：监督概念主结果和单独的 Standard 诊断。
- `classification_shots`、`feedback_shots`、`correction_shots`、`concept_shots`：PNG／PDF／SVG；虚线表示精确值，阴影为训练种子间标准差。
- `seedN/.../joint.pt`：32 分支概率、label 概率质量及样本身份。
- `seedN/.../samples.pt`：各预算和重复的五位计数、label 计数、联合预测和真实码次数，可重算报告指标。
- 数据缓存、测试来源哈希、条件锁和完整结果锁。

## 检查

```bash
bash experiments/grouped_robot_shots_final/scripts/check.sh
```

包含 Ruff、ty、Pylint 和真实 CUDA 测试：联合抽样的关联和位顺序、嵌套 shots 前缀、确定性重放、冻结模型推理、历史结果只读、暂停和恢复、重复次数与训练种子的统计区分、测试读取前置条件、身份隔离、原始 label 和固定预处理。

已完成的检查及工程运行记录见 [DEVELOPMENT_CHECK.md](DEVELOPMENT_CHECK.md)。
