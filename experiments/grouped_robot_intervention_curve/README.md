# Robot：纠正 0～5 个概念的数量曲线

补齐既定实验计划中的概念纠正数量曲线。复用已完成的 Robot 四层前段、五层后段模型及正式测试数据缓存，评价 Independent、Sequential、Joint 三种模式和五个训练种子。新代码和输出独立，历史模型、数据及报告只读，不重新训练。

## 固定实验内容

| 设置 | 内容 |
|---|---|
| 三种模式 | Independent、Sequential、Joint |
| 训练种子 | 0、1、2、3、4 |
| 正式数据 | Robot v5 原正式测试集，6,144 张图片 |
| 模型与预处理 | 使用既定 checkpoint、居中、pool、训练方差路由及角度缩放 |
| 纠正数量 | 0、1、2、3、4、5 个概念位置 |
| 各数量的组合数 | 1、5、10、10、5、1，共 32 个子集 |
| 预测规则 | 精确 Born 加权 label 概率，阈值 `>=0.5` |
| 推理 | CUDA；默认 batch 2,048；调用原 FusionModel/TorchQuantum 电路 |

每种模式、每个种子都覆盖全部 32 个组合，共 **480 个条件**。其中纠正 0 个、5 个的 **30 个端点**复用上一轮保存的精确概率，并核对模型、数据和指标；新增推理 **450 个部分纠正条件**。

这里的“纠正 k 个”指向模型提供 k 个概念位置的真实值，包括原本已经预测正确的位置；不是保证恰好找到并修复 k 个错误。未被选中的位置继续使用原来的物理测量记录。

干预只替换用于控制后五位 X 门的经典记录；原测量分支、Born 权重、反馈前条件量子态保留。前半段概念预测不因纠正而变成真实概念。Joint 使用自己的前后段 checkpoint，缓存量子态仅在前半段参数哈希相同时复用。

## 平均与报告规则

以纠正两个概念为例：分别评价全部 `C(5,2)=10` 个组合，每个组合先得到 label 准确率，再对十个准确率等权平均，得到该训练种子的一个点。最后对五个种子的五个点报告均值和样本标准差（n−1）。

这相当于对均匀随机选择 k 个概念位置的平均效果做穷举计算。**不先平均不同组合的 label 概率再分类**，那会得到另一个集成模型。组合间波动单独记录，不把组合当作额外训练种子。

主图包含正常／部分／全部纠正后的 label 准确率，以及相对不纠正的提升。保留下降和非单调曲线，不选择最优纠正顺序，不比较哪个具体概念最重要。额外保存 balanced accuracy、BCE 和相邻纠正数量之间的变化。

本次使用精确概率，不额外扩展为“所有子集 × 所有 shots”的实验。既有 shots 报告保持原样。

## 启动

在仓库根目录执行：

```bash
bash experiments/grouped_robot_intervention_curve/scripts/launch.sh
```

默认 tmux 会话为 `grouped_robot_intervention_curve`。日志和 Rich 状态分窗显示；完成、暂停或失败后自动退出，文件保留。

```bash
# 查看状态
bash experiments/grouped_robot_intervention_curve/scripts/status.sh

# 进入 tmux；Ctrl-b 然后 d 离开查看，后台继续
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_robot_intervention_curve

# 程序已经停止后恢复
bash experiments/grouped_robot_intervention_curve/scripts/launch.sh --resume

# 跟随日志
tail -f outputs/grouped_robot_intervention_curve/robot_v5_l4_head5_five_seeds/evaluation.log
```

默认来源：`outputs/grouped_robot_shots_final/robot_v5_l4_head5_five_seeds/`。

默认输出：`outputs/grouped_robot_intervention_curve/robot_v5_l4_head5_five_seeds/`。

## 预检和开发模式

```bash
# 校验历史结果、checkpoint、缓存及 CUDA；不产生新的预测或输出
bash experiments/grouped_robot_intervention_curve/scripts/run.sh --plan-only

# 保存预检状态后暂停；之后 --resume 开始评价
bash experiments/grouped_robot_intervention_curve/scripts/run.sh --preflight-only

# 本次调用新增完成三个组合条件后暂停
bash experiments/grouped_robot_intervention_curve/scripts/launch.sh --max-conditions 3
```

默认预检会读取**已经完成过正式评价的**测试集缓存并核验其来源；本实验补齐既定计划，没有再次拟合预处理、挑选模型或重新训练。完成后保留与先前测试报告的关联，不将本次曲线说成尚未查看过测试结果时完成的预注册实验。

开发模式只使用来源中的 validation 数据作为 `validation_proxy`，不读取真实测试数据或预测：

```bash
bash experiments/grouped_robot_intervention_curve/scripts/launch.sh \
  --session robot_curve_smoke \
  --out outputs/grouped_robot_intervention_curve/development_smoke \
  --development --seeds 0 --limit 64
```

开发结果明确标记 `test_read=false`、`test_evaluated=false`，不计入正式性能结果。正式模式要求五个种子和完整测试集。

每个组合单独保存结果。恢复时核验源码、配置、运行环境、历史模型、数据和已有产物，跳过已完成条件；当前未完成条件重新计算。SIGINT／SIGTERM 请求安全暂停。完整报告需要所有条件完成，不把部分运行误报为完成。

## 输出

- `summary.md/json`：六个纠正数量的三模式结果、样本数量和完成状态。
- `curve_summary.csv`：跨种子均值与样本标准差。
- `seed_curves.csv`：每个种子在各纠正数量下的组合平均值、纠正收益及相邻点变化。
- `subset_results.csv`：全部 480 个组合条件的指标，供复核；不按结果筛选组合。
- `intervention_curve.png/pdf/svg`：准确率和纠正收益两幅面板，阴影为种子间标准差。
- `seedN/MODE/maskNN/joint.pt`：原概念概率、纠正后的 label 分支概率质量、样本身份和真实标注。
- 各条件的 `evaluation.json`、来源锁、结果锁、心跳和日志。

本报告连接上一轮四模式、反馈消融与 shots 报告；旧报告不重写。

## 工程检查

```bash
bash experiments/grouped_robot_intervention_curve/scripts/check.sh
```

包含 Ruff、ty、Pylint 和 CUDA 测试：真实 X 门逐分支核对、全部组合计数、先评分再平均、种子级统计、端点复用、冻结推理、连续与恢复结果一致、历史来源只读及篡改拒绝。

实际检查记录见 [DEVELOPMENT_CHECK.md](DEVELOPMENT_CHECK.md)。
