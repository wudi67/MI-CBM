# 与 grouped VQC 相近参数的 MLP 对照

本目录独立于量子实验。直接复制已完成 VQC 运行的 `data.pt` 和 `preprocessing.json`，
不重新居中、池化、拟合方差路由、标准化或改变角度倍率。
默认来源是 `outputs/grouped_vqc_training_modes/dsprites_l4_seed0`。
默认使用完整训练 25,593 / 验证 5,479 张，测试数据不参与处理和评价。

已完成的工程验证及证据见 [DEVELOPMENT_CHECK.md](DEVELOPMENT_CHECK.md)。

## 固定模型和训练预算

| 任务 | 结构（含 bias） | 参数 | Epoch | VQC 比较对象 |
|---|---|---:|---:|---|
| 图片→Concept | Linear(40,3) → Tanh → Linear(3,32) | 251 | 100 | 前端 240 参数；概念单训100 / Joint100 |
| 图片→Label | Linear(40,6) → Tanh → Linear(6,1) | 253 | 200 | 整体 264 参数；Joint200 / Sequential200 |

概念输出32个logits，目标仍为 `8*shape+scale`，使用交叉熵，即真实联合码的NLL。
所有32种编码均保留，不屏蔽14个非法码，不改为独立的shape/scale分类头。
Label模型只使用二元分类损失 `BCEWithLogitsLoss`，不使用概念监督。
这两个模型分别诊断概念预测与整体分类能力，不组成经典CBM。

固定Tanh，seed 0，batch 1024，验证batch 2048，梯度裁剪5，Adam；
默认学习率网格为 **0.001、0.003、0.01**，共6个训练单元。
默认每个概念单元2,500次更新、每个Label单元5,000次更新。
同一任务的各学习率读取相同初始权重文件；每轮样本顺序与VQC实际历史逐一核对。
MLP和VQC架构不同，不声称二者初始权重相同。

每个候选只取固定末轮结果，不选择最佳epoch。
各任务按末轮验证NLL/BCE最小选择学习率；并列时选较低学习率。
保留全部6个候选及原始曲线，不只导出获选模型。

## 启动、查看和恢复

在仓库根目录执行。脚本使用VQC环境、CUDA和确定性计算；CUDA不可用时直接报错。
训练输入和参数都在GPU，保持float32，不使用AMP或TF32。

```bash
# 只查看计划
experiments/grouped_mlp_controls/scripts/run.sh \
  --out outputs/grouped_mlp_controls/dsprites_seed0 --plan-only

# 一次启动两个模型 × 三个学习率，自动顺序运行
experiments/grouped_mlp_controls/scripts/launch.sh \
  --session grouped_mlp_seed0 \
  --out outputs/grouped_mlp_controls/dsprites_seed0

# 当前单元、学习率、epoch、步数、进程和心跳
experiments/grouped_mlp_controls/scripts/status.sh \
  --out outputs/grouped_mlp_controls/dsprites_seed0

# 上方日志、下方实时状态。程序结束后会话自动退出。
/root/miniforge3/envs/VQC/bin/tmux attach -t grouped_mlp_seed0

tail -f outputs/grouped_mlp_controls/dsprites_seed0/train.log

# 原worker退出后恢复；已完成单元核验后跳过
experiments/grouped_mlp_controls/scripts/launch.sh \
  --session grouped_mlp_seed0_resume \
  --out outputs/grouped_mlp_controls/dsprites_seed0 --resume
```

恢复保留模型、Adam、随机数状态、epoch内排列和偏移、部分损失累计及完整历史。
配置、源码、输入缓存和上游VQC结果均有哈希检查；恢复时禁止改变这些内容。
每25步和每个epoch保存；SIGINT/SIGTERM会在安全边界保存并暂停。
独占锁防止同一输出目录同时运行两个worker。

## 工程检查

```bash
experiments/grouped_mlp_controls/scripts/check.sh

# 完整输入、缩短预算，验证六单元和后台恢复；使用独立目录
experiments/grouped_mlp_controls/scripts/launch.sh \
  --session grouped_mlp_cuda_check \
  --out outputs/grouped_mlp_controls/cuda_check \
  --concept-epochs 2 --label-epochs 3 --max-steps 1

# 第1步暂停后恢复，自动跑完其余单元并退出tmux
experiments/grouped_mlp_controls/scripts/launch.sh \
  --session grouped_mlp_cuda_check_resume \
  --out outputs/grouped_mlp_controls/cuda_check --resume
```

`--max-steps`是每个尚未完成单元的累计步数上限，不是本次新增步数；恢复时省略即可继续。
`--train-limit` / `--val-limit`只用于工程测试，截取源缓存前若干行，默认均为0。
输入使用子集或预算不匹配时，汇总不生成MLP减VQC的性能差值，避免把短程检查当正式对照。

## 输出与解释

| 文件 | 内容 |
|---|---|
| `config.json`, `manifest.json` | 模型、预算、学习率网格、源码、CUDA环境 |
| `reference_lock.json` | 上游VQC端点、参数量、200轮顺序哈希和原始文件哈希 |
| `data.pt`, `preprocessing.json` | 与上游文件字节相同的共享输入缓存 |
| `initialization.pt` | 各任务在不同学习率下共享的初始权重和RNG |
| `{concept,label}/lr_*/resume.pt` | 模型、Adam、训练进度及历史 |
| `{concept,label}/lr_*/history.json` | 每轮训练/验证损失、指标和实际样本顺序哈希 |
| `{concept,label}/lr_*/result.json` | 固定末轮结果、权重哈希、训练时间和显存 |
| `summary.json`, `summary.csv`, `summary.md` | 全候选、按验证损失选中的配置、与VQC的对照 |
| `heartbeat.json`, `train.log` | 当前任务/学习率、进度及日志 |

概念指标复用VQC的分组解码、联合MAP、分组同时正确率和非法编码概率。
`true_concept_probability`是经典模型分配给真实组合的概率，可与VQC的
`joint_single_shot_probability`作为概率质量参照，但它不是物理量子测量。
本实验不生成经典“量子shots”或控制门干预指标。

Label模型报告accuracy、balanced accuracy和BCE。
概念模型不会通过真实规则查表生成Label分数；两项任务保持各自的监督和指标。

251参数比240参数前端多4.58%；253参数比264参数整体少4.17%。参数相近不等于
表达能力、计算量或调参预算相同。Label-only MLP没有概念监督，而Joint VQC有；
MLP进行了三学习率筛选，历史VQC固定一个学习率。结果用于定位问题，不能单独支撑
量子优势结论。两边相同0.01学习率的固定预算结果也完整保留在主表中。

训练秒数仅累计训练步，排除预处理、验证和I/O；显存是当前进程的PyTorch峰值分配值。
这轮是单个seed的开发验证，不是最终测试结果。
