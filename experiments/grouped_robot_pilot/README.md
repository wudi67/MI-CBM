# Robot P0：11-qubit Independent 有／无反馈预实验

独立于 dSprites 和旧 Robot 实验。复用 `experiments/grouped_dynamic_vqc/model.py`
中的完整 `GroupedDynamicVQC`：底层仍实际调用 FusionModel.TQLayer 和 TorchQuantum。
不改动原模型源码，不读取或覆盖旧实验权重。

## 这次运行什么

| 阶段 | 训练内容 | 默认预算 |
|---|---|---|
| concept | 只训练前十个 qubit 的演化，监督真实五位概念组合的 Born 概率 | 100 epochs |
| independent | 冻结前半段；用真实概念控制后五个 qubit 的 X 门，只训练后半段分类电路 | 100 epochs |
| no_feedback | 同一冻结前半段、相同的后半段初始权重；保留中途测量，关闭反馈 X 门 | 100 epochs |

seed 固定为 0；Adam lr=0.01，batch=1024，eval batch=2048，梯度裁剪 5。
全部阶段用固定最终 epoch，不挑最佳验证点。两份后半段使用相同样本顺序，分别初始化新的 Adam。
默认每阶段 18 batches × 100 epochs = 1800 次更新，合计 5400 次更新。

后半段训练完成后，自动完成 **8 个精确概率验证条件**：

- Independent 正常预测：用实际测量记录控制 X 门。
- 分别纠正 head_shape、body_shape、has_antennae、ears_shape、foot_shape（5 个条件）。
- 同时纠正全部五个概念。
- 无反馈模型正常预测：零控制。

所有正向和负向结果都保留。此次是 **seed 0 验证集预实验**，不读取测试 CSV 或测试图片，
不运行五种子正式实验或 shots 扫描。

## 结构和数据

当前 `data/robot/` 是 Robot v5，五个二元概念按以下顺序编码：

`head_shape, body_shape, has_antennae, ears_shape, foot_shape`。

第一位为最高位，五位组合编号是 `16*c0 + 8*c1 + 4*c2 + 2*c3 + c4`。
全部 32 个编码合法，不能沿用 dSprites 的 18 个合法概念组合筛选规则。

- 前十个 qubit：A5 + B5，共同参与图像上传和前段演化。
- 前段四层，每层上传同一份 40 维输入，每个 qubit 使用四个数值；随后执行 U3/CU3 环。
- 测量 A5；每个测量记录位一对一控制对应 B 位的 X 门。
- B5 保留对应测量分支的量子态，加上初始化为零的第 11 个 readout qubit，
  进入沿用 dSprites 的一层 RY/RZZ/RY/RXX 分类电路及末端旋转。
- 共 264 个可训练参数（前段 240、后段 24）。没有新增经典可训练概念读出层。
- 预测位不参与图像上传。标签 0=drent、1=glorp，按 P(readout=1)≥0.5 判断。

预处理：`1-gray/255` → 非白前景包围盒整数平移居中 → 均值池化 10×4。
保留灰度和抗锯齿值，不裁剪、不缩放前景、不二值化。
沿用 dSprites 的 `fit_routing`：只在完整训练图像上计算 40 维方差，稳定降序排序，
轮转分配到十个 qubit，每个四值，角度统一乘 π 一次。验证集只应用该固定排列。
不统计池化输入种类，不筛掉重复输入，不用多数标签替换原始随机标签。

只加载原始 train/val CSV，核对两者 robot_id 不交叉、每个身份四次渲染完整且概念一致。
默认训练 18,432 张、验证 6,144 张。预处理缓存和源图片均记录哈希；数据、源码、配置变化会拒绝续跑。

## 启动和恢复

在仓库根目录执行一条命令，后台顺序跑完三个训练阶段和八个验证条件：

```bash
bash experiments/grouped_robot_pilot/scripts/launch.sh
```

训练及状态窗格使用 Rich。正常结束后 tmux 自动退出，结果和日志保留。

```bash
# 查看状态
bash experiments/grouped_robot_pilot/scripts/status.sh

# 进入 tmux
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_robot_pilot

# 实时日志
tail -f outputs/grouped_robot_pilot/robot_v5_l4_seed0/evaluation.log

# 中断后恢复
bash experiments/grouped_robot_pilot/scripts/launch.sh --resume
```

恢复保存模型、Adam、随机数状态、当前 epoch 的样本排列和 batch 位置。
已经完成的训练与评价只校验，不重复执行。每 25 次更新和每个 epoch 末保存恢复点。
SIGINT/SIGTERM 会在当前 batch 后保存并暂停；硬终止从最近保存点恢复。

```bash
# 只核对 CSV、CUDA 和计划，不创建输出或预处理图片
bash experiments/grouped_robot_pilot/scripts/run.sh --plan-only

# 仅准备输入、初始模型和来源清单，不做优化
bash experiments/grouped_robot_pilot/scripts/launch.sh --preflight-only
# 完成后通过正常 --resume 命令继续
```

默认禁止改变固定 P0 预算；需要工程检查时使用单独输出及 `--development`。
此标记的结果不能作为正式实验精度。

```bash
bash experiments/grouped_robot_pilot/scripts/launch.sh \
  --session grouped_robot_pilot_dev \
  --out outputs/grouped_robot_pilot/development_check \
  --development --concept-epochs 2 --head-epochs 2 \
  --train-limit 128 --val-limit 64 \
  --batch-size 64 --eval-batch-size 128 --checkpoint-steps 1
```

## 输出和解释

默认输出：`outputs/grouped_robot_pilot/robot_v5_l4_seed0/`。

| 文件 | 内容 |
|---|---|
| `summary.md` / `summary.json` | 验证结果、反馈与纠正变化 |
| `condition_results.csv` | 全部八个验证条件的 label 和概念指标 |
| `concept_results.csv` | 五个概念分别的准确率、balanced accuracy、BCE |
| `paired_results.csv` | 反馈及概念纠正的 label 准确率变化（百分点） |
| `training/*/endpoint.pt` | 三阶段固定终点，含完整 Adam 和训练历史 |
| `validation/*/*/predictions.pt` | 原始 32 分支概率、label 分支概率质量、样本身份及目标 |
| `learning_curves.*` / `interventions.*` | 可导出的 PNG、PDF、SVG 图 |
| `manifest.json` / `data_lock.json` / `result_lock.json` | 来源、缓存和结果完整性记录 |

概念指标分开报告：逐概念准确率、平均 bit 准确率、五个边缘阈值预测全部正确率、
联合 MAP、真实五位组合的单次测量正确概率。它们不能互换。
纠正后的真实控制也不计为模型原始概念预测成功。

这里保留输入相关的 B 状态，Independent 是本项目已采用的定义。
有反馈与无反馈两边都进行测量；该消融检验的是记录控制 X 门的贡献。
全部纠正保留原测量分支及分支权重，只替换经典控制记录。

## 代码检查

```bash
bash experiments/grouped_robot_pilot/scripts/check.sh
```

包括 Ruff、ty、Pylint 和 CUDA 测试：五位编码及纠正映射、显式 X 门对照、
灰度保持与训练集方差拟合、完整 Adam 中途恢复、冻结参数、默认 1024 batch、
概念训练不依赖任务标签、原始验证结果复算、源码与数据隔离。
