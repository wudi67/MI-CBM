# 机器人数据集 v5（无泄漏按身份划分版）

v5 = **v4 的数据 + 修好的划分**。图片、概念标注、标签与 v4 逐字节/逐行完全相同（同一管线、同一 seed=1014），唯一区别是 **train/val/test 的切分方式**：

- v4 按“渲染行”随机切 → 同一机器人身份的 4 次渲染（以及仅隐形子特征不同的近重复）会散落到不同划分，构成泄漏。
- v5 **按机器人身份分组切分**（同一身份的 4 次渲染必在同一划分），并**按标签分层**保持类别平衡。

生成时间：2026-07-20 · 生成脚本：本目录 `make_dataset.py`（从仓库根目录 `python data/robot/make_dataset.py` 完整复现，自包含、不依赖 v4/Robot.zip）

---

## 为什么 v4 文档里给的修法是错的

v4 的 `DATASET.md` 建议 `sample(groups=dataset.meta["robot_ids"])`。这是 **no-op**：`meta["robot_ids"]` 就是 `catalog_df["id"]` == 行号（`0..30719`，**逐行唯一**），按它分组等于每行自成一组，什么也拦不住。

catalog 实际是把 7,680 个唯一特征组合**重复拼接 4 次**得到 30,720 行，所以第 r 行的真实身份是：

```
identity = robot_ids % num_unique_robots      # 0..7679，每个身份出现 4 次
```

v5 用这个 `identity` 作为分组键、用 `label` 作为分层键：

```python
dataset.sample(test_size=0.2, val_size=0.2,
               groups=robot_ids % num_unique_robots,
               stratify=y, seed=1014)
```

> 注：v5 的 CSV 直接给出了 `robot_id`（0..7679）列，无需再自己算取模——按 `robot_id` 分组即可复现或重划分。

---

## 概览

| 项目 | 内容 |
|---|---|
| 图片 | 30,720 张（7,680 身份 × 4 渲染），**32 × 32 灰度**（与 v4 逐字节相同） |
| 概念 | **5 个**二元概念，全部因果参与 label（与 v3/v4 相同） |
| 类别 | 2 类（glorp=1 / drent=0），**每个划分内 ≈ 50 : 50** |
| 划分 | 60 / 20 / 20，**按身份分组（无泄漏）+ 按标签分层** |
| 种子 | 1014（渲染、打标、划分同源） |

---

## 概念与标签规则（同 v3/v4）

| 概念 | 取值 0 / 1 | 公式权重 |
|---|---|---|
| `body_shape`   | square / round     | **+5** |
| `foot_shape`   | flat / pointy      | **+4** |
| `has_antennae` | 无 / 有            | **+3** |
| `head_shape`   | square / round     | **+2** |
| `ears_shape`   | square / triangle  | **+1** |

```
score = 5·[body=round] + 4·[foot=pointy] + 3·[antennae=有] + 2·[head=round] + 1·[ears=triangle] − 7.5
P(glorp) = sigmoid(8.4 × score)      (stochastic，rng_seed=1014)
```

32 种概念组合按 score 正负恰好 16 : 16 → 类别平衡；每个概念都存在能翻转确定性标签的语境（无“乘客”概念）；最小间隔 |score| = 0.5、temperature 8.4 → 整体等效标签噪声 ≈ 0.3%（近确定性）。

图片仍按完整 9 特征渲染（含膝盖/嘴/手型/foot 子类型等），但这些在 32px 下不可见或极弱，未列入标注，等价于无害背景噪声。

---

## 生成后验证（全部通过）

1. **与 v4 同源**：抽样 399 张图片 MD5 与 `data/robot_v4/` **逐字节相同**；30,720 行按文件名一一对齐，`label` 与 5 个概念列与 v4 **完全一致**。→ v5 只改了划分。
2. **无泄漏**：`train/val/test` 的 `robot_id` 集合两两不相交，并集恰为全部 7,680 个身份；每个身份的 4 行全部落在**同一个**划分（splits-per-id = 1）。
3. **类别平衡**：见下表，三个划分 glorp 占比均 ≈ 50%。
4. **概念 → label 可分性**（logistic 回归，train 拟合 / test 评估）：accuracy = balanced accuracy = **0.9971**（与 v4 的 0.9972 一致）；权重排序 body > foot > antennae > head > ears，与真实公式一致。

| 划分 | 行数 | glorp 占比 |
|---|---|---|
| train | 18,432 | 49.97% |
| val   | 6,144  | 49.95% |
| test  | 6,144  | 50.07% |

> 局限：按身份分组消除了“同一身份的 4 次渲染跨划分”这一确定性泄漏。但**不同身份**之间仍可能存在近重复（例如仅 foot 子类型 trapezoid↔rounded 不同、32px 下几乎无差别），它们是不同 `robot_id`，仍可能分属不同划分。要彻底消除像素级近重复，需按“可见特征向量”分组，但这会把可用组合数从 7,680 压到 32（= 2⁵ 个标注概念组合），使划分无法覆盖所有概念组合——故 v5 不这么做，保留身份级无泄漏这一正确且实用的粒度。

---

## 目录 / 格式

```
data/robot/
├── DATASET.md
├── make_dataset.py                 # 端到端生成脚本（可复现）
├── robot_images/                   # 30,720 张 32×32 灰度 PNG
├── robot_images_train_labels.csv   # 18,432 行
├── robot_images_val_labels.csv     #  6,144 行
└── robot_images_test_labels.csv    #  6,144 行
```

CSV 列：`image_path, robot_id, render_id, head_shape, body_shape, has_antennae, ears_shape, foot_shape, label, class`。

- `robot_id`（0..7679）：真实机器人身份，**同一 id 的 4 行必在同一划分**；用于审计 / 重划分。
- `render_id`（0..3）：同一身份的第几次渲染（仅 color_scheme 不同，灰度下近乎重复）。

## 确认性实验的外部渲染集

v5 已经枚举全部 7,680 个 latent identity，因此不能通过换 seed 后重新切分来
构造外部测试集；那会让开发集 train identity 进入新的 test。使用：

```bash
python data/robot/make_external_test_dataset.py \
  --source-root data/robot \
  --seed 2027 \
  --out data/robot_external_2027
```

该脚本保持原 identity split，只生成新的轻微平移/亮度/对比度/噪声渲染，并按
同一标签公式做独立随机抽样。它会写入生成 manifest 和跨 root 审计，要求外部
test 与开发 train/val 的 identity 交集为 0，且新旧图片没有逐字节重复。

```python
import pandas as pd
df = pd.read_csv("data/robot/robot_images_train_labels.csv")
concept_cols = ["head_shape", "body_shape", "has_antennae", "ears_shape", "foot_shape"]
C = df[concept_cols].values     # (N, 5)
y = df["label"].values          # (N,)
groups = df["robot_id"].values  # (N,) 无泄漏分组键
```
