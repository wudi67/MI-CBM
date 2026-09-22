"""Fixed-endpoint scores, paired differences and learning curves for all methods."""

import csv
from pathlib import Path
from statistics import mean, stdev

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_robot_mlp_diagnostic.model import transitions
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import load

from .protocol import CELLS, CONDITIONS, INDICES, METHODS, cell_name
from .training import Job

TITLES = {
    "uniform": "原均匀初始化",
    "eft_gaussian": "小角度高斯",
    "eft_readout": "小角度高斯＋读出调整",
}
ENGLISH = {
    "uniform": "Uniform",
    "eft_gaussian": "Small Gaussian",
    "eft_readout": "Gaussian + readout shift",
}


def summarize_values(evaluations: list) -> tuple[list, list, list]:
    lookup = {
        (
            v["initialization_method"],
            v["initialization_index"],
            v["role"],
            v["condition"],
        ): v
        for v in evaluations
    }
    expected = {
        (method, index, role, name)
        for method, index in CELLS
        for role in ("train", "validation")
        for name, _ in CONDITIONS
    }
    if set(lookup) != expected or len(evaluations) != len(expected):
        raise ValueError("All nine cells and 36 unique conditions are required")
    rows, aggregates, pairs = [], [], []
    for role in ("train", "validation"):
        for method in METHODS:
            for index in INDICES:
                normal = lookup[method, index, role, "measured"]["metrics"]["label"][
                    "accuracy"
                ]
                for name, _ in CONDITIONS:
                    v = lookup[method, index, role, name]
                    rows.append(
                        {
                            "method": method,
                            "initialization_index": index,
                            "role": role,
                            "condition": name,
                            "n_samples": v["n_samples"],
                            **v["metrics"]["label"],
                            "correction_gain_pp": 100
                            * (v["metrics"]["label"]["accuracy"] - normal),
                            "reused_historical_training": v[
                                "reused_historical_training"
                            ],
                        }
                    )
            for name, _ in CONDITIONS:
                values = [
                    r
                    for r in rows
                    if (r["method"], r["role"], r["condition"]) == (method, role, name)
                ]
                aggregate = {
                    "method": method,
                    "role": role,
                    "condition": name,
                    "n_initializations": 3,
                }
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "bce",
                    "correction_gain_pp",
                ):
                    aggregate[f"{key}_mean"] = mean(r[key] for r in values)
                    aggregate[f"{key}_std"] = stdev(r[key] for r in values)
                aggregates.append(aggregate)
        for method in METHODS[1:]:
            for name, _ in CONDITIONS:
                differences = []
                for index in INDICES:
                    a, b = (
                        next(
                            r
                            for r in rows
                            if (
                                r["method"],
                                r["initialization_index"],
                                r["role"],
                                r["condition"],
                            )
                            == (m, index, role, name)
                        )
                        for m in ("uniform", method)
                    )
                    differences.append(
                        {
                            "initialization_index": index,
                            "accuracy_change_pp": 100 * (b["accuracy"] - a["accuracy"]),
                            "bce_change": b["bce"] - a["bce"],
                            "correction_gain_change_pp": b["correction_gain_pp"]
                            - a["correction_gain_pp"],
                        }
                    )
                pairs.append(
                    {
                        "method": method,
                        "role": role,
                        "condition": name,
                        "direction": "method minus uniform",
                        "differences": differences,
                        **{
                            f"{key}_{stat}": fn(v[key] for v in differences)
                            for key in (
                                "accuracy_change_pp",
                                "bce_change",
                                "correction_gain_change_pp",
                            )
                            for stat, fn in (("mean", mean), ("std", stdev))
                        },
                    }
                )
    return rows, aggregates, pairs


def plots(output: Path, aggregates: list, curves: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for method, offset in zip(METHODS, (-0.24, 0, 0.24), strict=True):
        values = [
            next(
                v
                for v in aggregates
                if (v["method"], v["role"], v["condition"])
                == (method, "validation", name)
            )
            for name, _ in CONDITIONS
        ]
        axes[0].bar(
            [i + offset for i in range(2)],
            [100 * v["accuracy_mean"] for v in values],
            yerr=[100 * v["accuracy_std"] for v in values],
            width=0.24,
            capsize=3,
            label=ENGLISH[method],
        )
        epochs = sorted(
            {
                v["epoch"]
                for v in curves
                if v["method"] == method and "validation_true_bce" in v
            }
        )
        samples = [
            [
                v["validation_true_bce"]
                for v in curves
                if v["method"] == method
                and v["epoch"] == epoch
                and "validation_true_bce" in v
            ]
            for epoch in epochs
        ]
        if any(len(v) != 3 for v in samples):
            raise ValueError(
                "Learning-curve aggregation requires all three initializations"
            )
        means, stds = [mean(v) for v in samples], [stdev(v) for v in samples]
        axes[1].plot(epochs, means, label=ENGLISH[method])
        axes[1].fill_between(
            epochs,
            [m - s for m, s in zip(means, stds, strict=True)],
            [m + s for m, s in zip(means, stds, strict=True)],
            alpha=0.15,
        )
    axes[0].set_xticks([0, 1], ["Normal", "Correct all concepts"])
    axes[0].set_ylabel("Validation label accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_xlabel("Label training epoch")
    axes[1].set_ylabel("Validation BCE with true concept controls")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def summarize(shared, training: list, evaluations: list) -> None:
    rows, aggregates, pairs = summarize_values(evaluations)
    orders, curves, sample_changes = [], [], []
    for method, index in CELLS:
        job = Job(shared, method, index)
        history = load(job.checkpoint_path)["progress"]["history"]
        orders.append([v["order_sha256"] for v in history])
        for point in history:
            row = {
                "method": method,
                "initialization_index": index,
                "epoch": point["epoch"],
                "train_batch_bce": point["train_loss"],
                "validation_normal_accuracy": point["validation"]["label"]["accuracy"],
            }
            for key in ("train_true", "validation_true"):
                if key in point:
                    for metric in ("accuracy", "bce"):
                        row[f"{key}_{metric}"] = point[key]["label"][metric]
            curves.append(row)
        for role in ("train", "validation"):
            directory = shared.output / cell_name(method, index) / role
            a, b = (
                load(directory / name / "predictions.pt")
                for name in ("measured", "correct_all_five")
            )
            sample_changes.append(
                {
                    "method": method,
                    "initialization_index": index,
                    "role": role,
                    **transitions(
                        a["branch_label_mass"].sum(1),
                        b["branch_label_mass"].sum(1),
                        a["labels"],
                    ),
                }
            )
    if any(order != orders[0] for order in orders[1:]):
        raise ValueError("Training sample orders differ between methods")
    diagnostics = read_json(shared.output / "initial_diagnostics.json")
    payload = {
        "manifest_sha256": shared.manifest_hash,
        "training": training,
        "rows": rows,
        "aggregates": aggregates,
        "paired_vs_uniform": pairs,
        "learning_curves": curves,
        "all_correction_sample_changes": sample_changes,
        "initial_diagnostics": diagnostics,
        "same_sample_orders_verified": True,
        "completed_training_cells": 9,
        "completed_conditions": 36,
        "new_adam_updates": sum(
            v["global_step"] for v in training if not v["reused_historical_training"]
        ),
        "test_read": False,
        "test_evaluated": False,
        "engineering_run": shared.config.development,
        "interpretation": (
            "One fixed frontend, three label initializations per method; "
            "sample SD uses ddof=1. Fixed final epoch, not best checkpoint. "
            "No barren-plateau guarantee is inferred."
        ),
    }
    atomic_json(shared.output / "summary.json", payload)
    for name, values in (
        ("summary.csv", rows),
        ("aggregates.csv", aggregates),
        ("learning_curves.csv", curves),
    ):
        fields = list(dict.fromkeys(k for v in values for k in v))
        with (shared.output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    lines = [
        "# Robot：五层 VQC 参数初始化对照",
        "",
        "三组均为五层、112 参数；每组三个初始化，"
        f"训练 {shared.config.head_epochs} 轮。",
        f"σ={shared.config.sigma:.8f}，复用均匀初始化历史训练：{shared.reuse_uniform}。",
        "Independent；前半段、保留量子态、概念控制 X、优化器及样本顺序相同。",
        "下表为验证集均值 ± 样本标准差，反映同一个前半段下的后半段初始化波动。",
        "",
        "| 初始化 | 条件 | label 准确率 | BCE | 纠正收益（百分点） |",
        "|---|---|---:|---:|---:|",
    ]
    for v in aggregates:
        if v["role"] == "validation":
            condition = "正常预测" if v["condition"] == "measured" else "全部概念纠正"
            lines.append(
                f"| {TITLES[v['method']]} | {condition} | "
                f"{100 * v['accuracy_mean']:.2f} ± {100 * v['accuracy_std']:.2f}% | "
                f"{v['bce_mean']:.5f} ± {v['bce_std']:.5f} | "
                f"{v['correction_gain_pp_mean']:+.2f} ± "
                f"{v['correction_gain_pp_std']:.2f} |"
            )
    lines += [
        "",
        "## 相对现有均匀初始化的变化",
        "",
        "| 方法 | 条件 | 准确率变化（百分点） | BCE 变化 |",
        "|---|---|---:|---:|",
    ]
    for v in pairs:
        if v["role"] == "validation":
            lines.append(
                f"| {TITLES[v['method']]} | {v['condition']} | "
                f"{v['accuracy_change_pp_mean']:+.2f} ± "
                f"{v['accuracy_change_pp_std']:.2f} | "
                f"{v['bce_change_mean']:+.5f} |"
            )
    lines += [
        "",
        "## 每个初始化",
        "",
        "| 方法 | 编号 | 条件 | 验证准确率 |",
        "|---|---:|---|---:|",
    ]
    for v in rows:
        if v["role"] == "validation":
            lines.append(
                f"| {TITLES[v['method']]} | {v['initialization_index']} | "
                f"{v['condition']} | {v['accuracy']:.2%} |"
            )
    lines += [
        "",
        "## 初始输出检查（尚未训练）",
        "",
        "| 方法 | 编号 | 平均 P(label=1) | BCE | 梯度范数（裁剪前） |",
        "|---|---:|---:|---:|---:|",
    ]
    for v in diagnostics["rows"]:
        lines.append(
            f"| {TITLES[v['method']]} | {v['initialization_index']} | "
            f"{v['p_mean']:.6f} | {v['initial_bce']:.5f} | "
            f"{v['gradient_l2_before_clip']:.5f} |"
        )
    lines += [
        "",
        "训练集真实控制指标、逐样本变化和每轮曲线"
        "见 summary.json 与 learning_curves.csv。",
        "保留所有初始化、报告固定末轮，不以验证集挑种子或最佳 epoch；未读取测试集。",
        "H-EFT-VA 启发的小角度尺度与本电路适配，"
        "不等同于复现原文电路，也不证明避免贫瘠高原。",
    ]
    if shared.config.development:
        lines += ["", "**工程短预算；不能作为正式比较结论。**"]
    (shared.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plots(shared.output, aggregates, curves)
    shared.verify_unchanged()
    artifacts = {
        str(p.relative_to(shared.output)): sha256(p)
        for p in shared.output.rglob("*")
        if p.is_file()
        and p.suffix in {".pt", ".json", ".csv", ".md", ".png"}
        and p.name not in {"heartbeat.json", "result_lock.json"}
    }
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": artifacts,
            "test_evaluated": False,
        },
    )
