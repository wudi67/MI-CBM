# 测量结果反馈消融：Joint 单种子与 Sequential 五种子

本实验比较同一 VQC 中使用／不使用概念测量记录执行条件 X 门。
两边都保留后五个 qubit 的图像相关状态，概念读出、分类电路及参数量相同。
无反馈方案在训练和评价时均使用恒等反馈，后半段重新训练；不是训练后临时关闭反馈。
这项消融衡量“根据测量结果反馈”模块的作用，不单独声称测量坍缩产生性能提升。

代码和输出与旧实验隔离。量子电路直接调用已有 `GroupedDynamicVQC`：
`FusionFrontend` → `FusionModel.TQLayer`，四层，每层上传每个 qubit 的四个值，
再执行 U3 和环形相邻 CU3。前段 10 qubit、240 参数；后五位加读出位、24 参数。

## 默认执行顺序

1. Joint，seed 0：有反馈／无反馈，各 200 轮；两部分同时用概念 NLL＋Label BCE 训练。
2. Sequential，seeds 0、1、2、3、4：每个种子分别训练概念前段 100 轮并冻结，
   两个方案各训练后半段 100 轮，只优化 Label BCE。
3. 每个已完成模型按自己的反馈方式重新评价精确概率和每张图片 256 次联合 `(m,y)` 测量。
   有反馈用实际记录，无反馈始终不执行条件 X。有限采样不是带采样噪声训练。
4. 每完成一对立即更新报告。Joint 单种子单列，Sequential 报均值、样本标准差和逐种子差值。
   不根据 Joint 正负结果自动选择是否报告，不把 Joint 混入 Sequential 的种子统计。

## 配对与旧结果复用

- 默认训练 25,593、验证 5,479。所有种子复用同一居中、Pool(10,4)、训练集方差分配缓存，
  不重做预处理、不改变切分、不访问测试集。开发子集也固定抽样，不随训练 seed 改变。
- 不同种子有独立初始化并重新训练前段；同一种子的两个方案共用初始参数、样本顺序、
  优化器设置及训练预算。Sequential 再共用冻结前段；Joint 的前段允许分别学习。
- 默认 Adam lr=.01、batch=1024、eval batch=2048、梯度裁剪 5，CUDA 强制开启，
  确定性算法、float32/complex64、关闭 AMP/TF32。没有 CPU 训练回退。
- Sequential 后半段用第 101–200 轮的样本顺序，从全新 Adam 开始；冻结前段的状态缓存
  在 CUDA 上，避免每个后半段 epoch 重算。所有 32 个测量分支均保留，包括非法概念码。
- 固定最终轮模型，不按验证集最高分选择。训练使用分支概率混合后的 BCE，
  不取 MAP 概念后训练，也不对分支振幅跨测量记录求和。
- 默认核验并只读复用旧 seed 0 的 Joint 有反馈、Sequential 概念前段、有反馈和无反馈
  四个完成端点；源权重、历史、样本顺序、数据及代码哈希均核验，复制到独立新目录。
  Joint 无反馈从相同原始初始化完整训练 200 轮。
- 因此默认新增 **1 个 Joint 无反馈训练、4 个概念前段训练、8 个后半段训练**。
  `--retrain-reference` 强制全部重新训练；训练协议不一致则不复用相应历史端点。
  有限 shots 无论是否复用旧权重均重新按正确控制评价，旧诊断文件不改写。

## 一次启动全部实验

在仓库根目录执行：

```bash
# 显示计划，不创建实验结果或启动训练
experiments/grouped_feedback_ablation/scripts/run.sh --plan-only

# 一个 CUDA worker 依次完成 Joint pilot 和全部 Sequential 种子
experiments/grouped_feedback_ablation/scripts/launch.sh

# 查看状态或连接 tmux
experiments/grouped_feedback_ablation/scripts/status.sh
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_feedback_ablation
tail -f outputs/grouped_feedback_ablation/dsprites_l4_paired/train.log

# 中断后恢复：完成的结果校验后跳过，未完成的保留 Adam 和轮内进度
experiments/grouped_feedback_ablation/scripts/launch.sh \
  --session grouped_feedback_resume --resume
```

正常完成、暂停、失败以后 tmux 自动退出，日志及结果保留。不要重复启动同一输出目录，
目录锁会阻止并发写入。恢复要求配置、源码、CUDA 环境和已锁定原始文件保持一致。

如果希望先看 Joint 后再继续：

```bash
experiments/grouped_feedback_ablation/scripts/launch.sh --stage joint
# Joint 结束后，继续同一协议的 Sequential；不需要重新跑 Joint
experiments/grouped_feedback_ablation/scripts/launch.sh \
  --session grouped_feedback_sequential --resume --stage sequential
```

`--stage` 只选择这次执行哪些任务，不修改清单中的种子和实验协议。
Joint 只跑完时任务会退出，但 `summary.json.status` 保持 partial，明确全套实验尚未完成。

## 结果文件与解释

默认目录：`outputs/grouped_feedback_ablation/dsprites_l4_paired/`。

- `summary.md/json`、`paired_results.csv`：Label accuracy、BCE、balanced accuracy、
  256-shots accuracy、逐种子配对差值、概念分组同时正确率、错变对／对变错数量。
  Sequential 单列 5 种子的均值和样本标准差，所有正负差值都保留。
- 各 `joint/seed0/{feedback,no_feedback}` 与 `sequential/seedN/...` 目录：
  `result.json`、`endpoint.pt`、`history.json`、`evaluation.json`、`predictions.pt`。
  完整概念指标（含联合 MAP、单次正确概念概率、非法码概率）保存在 `evaluation.json`。
- 新训练目录另有 `resume.pt`、`initialization.json`、`gradient_checks.json`；
  checkpoint 保存参数、Adam、随机状态、当前 epoch 的排列和 offset。
- `manifest.json`、`reference_lock.json`、`result_lock.json`、各 `evaluation_lock.json`
  记录协议、来源和产物哈希。部分完成报告与完整实验状态区分保存。

概念前段冻结时，两个方案的概念分布应相同，程序会核验；Joint 中不要求相同，
需要一并查看概念指标，不能把 Label 提升解释成固定概念预测器后的纯后段效应。
5 个训练种子是重复单位，不把 5,479 张验证图片当成独立训练重复。
这些仍然是验证集实验。最终测试留待训练模式及协议固定之后另行执行。

## 工程验证

```bash
experiments/grouped_feedback_ablation/scripts/check.sh

# 独立短跑，不能作为论文实验结果
experiments/grouped_feedback_ablation/scripts/launch.sh \
  --session feedback_smoke \
  --out outputs/grouped_feedback_ablation/manual_smoke \
  --seeds 0,1 --joint-epochs 1 --concept-epochs 1 --head-epochs 1 \
  --train-limit 36 --val-limit 18 --batch-size 18 --eval-batch-size 18 --shots 32
```

`--max-steps 1` 在首个未完成训练任务累计达到一步后暂停，可检查轮内恢复。
测试包含实际 CUDA 运算、直接量子演化与分支优化的一致性、无反馈联合采样语义、
历史 Joint 单步一致性、两模式所有阶段的 Adam 精确恢复、来源及预测文件篡改检测。
