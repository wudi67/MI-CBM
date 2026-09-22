"""All fixed-endpoint results, paired initialization differences and learning curves."""

import csv
from statistics import mean, stdev

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import load

from .protocol import CELLS, INDICES, METHODS, cell_name

METRICS = (
    "all_concepts_accuracy",
    "mean_bit_accuracy",
    "joint_map_accuracy",
    "joint_single_shot_probability",
    "joint_nll",
)
TITLES = {"uniform": "原均匀初始化 [0, π]", "eft_gaussian": "小角度高斯初始化"}


def summarize_values(evaluations: list) -> tuple[list, list, list]:
    lookup = {(v["cell_name"], v["role"]): v for v in evaluations}
    expected = {
        (cell_name(*cell), role) for cell in CELLS for role in ("train", "validation")
    }
    if set(lookup) != expected or len(evaluations) != len(expected):
        raise ValueError("All six cells and twelve unique evaluations are required")
    rows, aggregates, paired = [], [], []
    for role in ("train", "validation"):
        for method in METHODS:
            values: list[dict] = []
            for index in INDICES:
                v = lookup[cell_name(method, index), role]
                row = {
                    "method": method,
                    "initialization_index": index,
                    "role": role,
                    "n_samples": v["n_samples"],
                    **{k: v["metrics"]["concept"][k] for k in METRICS},
                }
                values.append(row)
                rows.append(row)
            aggregates.append(
                {
                    "method": method,
                    "role": role,
                    "n_initializations": len(INDICES),
                    **{
                        f"{key}_{stat}": fn(v[key] for v in values)
                        for key in METRICS
                        for stat, fn in (("mean", mean), ("std", stdev))
                    },
                }
            )
        differences = []
        for index in INDICES:
            a, b = (
                lookup[cell_name(method, index), role]["metrics"]["concept"]
                for method in METHODS
            )
            differences.append(
                {
                    "initialization_index": index,
                    **{
                        key: (b[key] - a[key]) * (1 if key == "joint_nll" else 100)
                        for key in METRICS
                    },
                }
            )
        paired.append(
            {
                "role": role,
                "direction": "eft_gaussian minus uniform",
                "units": "percentage points, except joint_nll in nats",
                "differences": differences,
                **{
                    f"{key}_{stat}": fn(v[key] for v in differences)
                    for key in METRICS
                    for stat, fn in (("mean", mean), ("std", stdev))
                },
            }
        )
    return rows, aggregates, paired


def plots(output, curves) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in METHODS:
        epochs = sorted({v["epoch"] for v in curves if v["method"] == method})
        for ax, key, scale in zip(
            axes, ("all_concepts_accuracy", "joint_nll"), (100, 1), strict=True
        ):
            averages, spreads = [], []
            for epoch in epochs:
                values = [
                    v[key] * scale
                    for v in curves
                    if v["method"] == method and v["epoch"] == epoch
                ]
                averages.append(mean(values))
                spreads.append(stdev(values))
            ax.plot(epochs, averages, label=method)
            ax.fill_between(
                epochs,
                [a - s for a, s in zip(averages, spreads)],
                [a + s for a, s in zip(averages, spreads)],
                alpha=0.15,
            )
            ax.set_xlabel("Epoch")
            ax.legend()
    axes[0].set_ylabel("Validation: all 5 marginal predictions correct (%)")
    axes[1].set_ylabel("Validation joint concept NLL")
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=180)
    plt.close(figure)


def write_csv(path, values) -> None:
    fields = list(dict.fromkeys(k for v in values for k in v))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def write_results(shared, evaluations, training) -> None:
    rows, aggregates, paired = summarize_values(evaluations)
    curves, orders, per_concept = [], [], []
    for method, index in CELLS:
        history = load(
            shared.output / cell_name(method, index) / "training/concept/endpoint.pt"
        )["progress"]["history"]
        orders.append([(v["order_epoch"], v["order_sha256"]) for v in history])
        curves.extend(
            {
                "method": method,
                "initialization_index": index,
                "epoch": row["epoch"],
                "train_batch_joint_nll": row["train_loss"],
                **{key: row["validation"]["concept"][key] for key in METRICS},
            }
            for row in history
        )
        for value in evaluations:
            if value["cell_name"] != cell_name(method, index):
                continue
            for concept, metric in value["metrics"]["concept"]["per_concept"].items():
                per_concept.append(
                    {
                        "method": method,
                        "initialization_index": index,
                        "role": value["role"],
                        "concept": concept,
                        **metric,
                    }
                )
    if any(order != orders[0] for order in orders):
        raise ValueError("Paired methods did not see the same training sample order")
    payload = {
        "manifest_sha256": shared.manifest_hash,
        "training": training,
        "rows": rows,
        "aggregates": aggregates,
        "paired_vs_uniform": paired,
        "learning_curves": curves,
        "per_concept": per_concept,
        "initial_diagnostics": read_json(shared.output / "initial_diagnostics.json"),
        "same_sample_orders_verified": True,
        "completed_training_cells": len(training),
        "completed_conditions": len(evaluations),
        "new_adam_updates": sum(v["global_step"] for v in training),
        "engineering_run": shared.config.development,
        "test_read": False,
        "test_evaluated": False,
        "interpretation": (
            "Three frontend initializations; fixed dataset split and shuffle order; "
            "fixed final epoch, sample SD ddof=1; "
            "no label training or barren-plateau guarantee"
        ),
    }
    atomic_json(shared.output / "summary.json", payload)
    for name, values in (
        ("summary", rows),
        ("aggregates", aggregates),
        ("per_concept", per_concept),
        ("learning_curves", curves),
    ):
        write_csv(shared.output / f"{name}.csv", values)
    lines = [
        "# Robot：概念预测 VQC 初始化对照",
        "",
        "相同四层、240 个 U3/CU3 参数；每种方法三个初始化，"
        f"各训练 {shared.config.concept_epochs} 轮。",
        f"高斯 σ={shared.config.sigma:.6f}。仅训练图片→概念，不训练或评价 label 电路。",
        "下表为固定末轮的验证集均值 ± 样本标准差；不挑最好种子或中间轮次。",
        "",
        "| 初始化 | 五个概念同时正确 | 平均概念准确率 | 联合 MAP | 概念 NLL |",
        "|---|---:|---:|---:|---:|",
    ]
    for value in aggregates:
        if value["role"] != "validation":
            continue
        cells = [TITLES[value["method"]]]
        for key in (
            "all_concepts_accuracy",
            "mean_bit_accuracy",
            "joint_map_accuracy",
            "joint_nll",
        ):
            scale = 1 if key == "joint_nll" else 100
            suffix = "" if scale == 1 else "%"
            cells.append(
                f"{value[key + '_mean'] * scale:.3f} ± "
                f"{value[key + '_std'] * scale:.3f}{suffix}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "五个概念同时正确指逐概念边缘概率以 0.5 判定后全对；"
            "联合 MAP 是另一个指标。",
            "单次测量真实记录概率、逐概念准确率、训练集指标和配对差值见 JSON/CSV。",
            "所有实验共享同一数据划分和训练顺序，只有初始化不同；未读取或评价测试集。",
            "这是 H-EFT-VA 启发的初始化尺度在现有 Fusion 电路上的对照，"
            "不是原论文电路复现。",
        ]
    )
    (shared.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plots(shared.output, curves)
    names = sorted(
        str(p.relative_to(shared.output))
        for p in shared.output.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix != ".log"
        and p.name not in {"heartbeat.json", "result_lock.json"}
    )
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {n: sha256(shared.output / n) for n in names},
            "test_evaluated": False,
        },
    )
