"""Report all three fixed budgets and matched-control train/validation metrics."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.results import write_csv

from .protocol import ARMS


def budgets(shared, arm: str) -> tuple[int, int]:
    source = shared.reference.config
    return (
        shared.config.concept_epochs
        if arm == "long_concept"
        else source.concept_epochs,
        shared.config.head_epochs if arm == "long_label" else source.head_epochs,
    )


def plots(shared, comparisons: list[dict]) -> list[str]:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for arm in ("baseline", *ARMS):
        root = shared.reference.output if arm == "baseline" else shared.output / arm
        if arm != "long_label":
            history = read_json(root / "training/concept/history.json")
            epochs = [r["epoch"] for r in history]
            axes[0, 0].plot(
                epochs,
                [r["validation"]["concept"]["joint_nll"] for r in history],
                label=arm,
            )
            axes[0, 1].plot(
                epochs,
                [
                    100 * r["validation"]["concept"]["all_concepts_accuracy"]
                    for r in history
                ],
                label=arm,
            )
        for cell in ("independent", "no_feedback"):
            history = read_json(root / "training" / cell / "history.json")
            style = "-" if cell == "independent" else "--"
            label = f"{arm}: {'conditional X' if cell == 'independent' else 'no X'}"
            epochs = [r["epoch"] for r in history]
            axes[1, 0].plot(
                epochs, [r["train_loss"] for r in history], style, label=label
            )
            axes[1, 1].plot(
                epochs,
                [100 * r["validation"]["label"]["accuracy"] for r in history],
                style,
                label=label,
            )
    titles = (
        "Concept validation NLL",
        "All five concepts correct (%)",
        "Label training BCE (true / zero controls)",
        "Normal validation label accuracy (%)",
    )
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set(xlabel="Epoch within phase", ylabel=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    figure.suptitle("Robot seed 0: fixed training-budget comparison")
    figures = [(figure, "learning_curves")]
    figure, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(comparisons))
    for offset, (field, label) in enumerate(
        (
            ("normal_accuracy", "Independent: measured X controls"),
            ("zero_accuracy", "Comparison: conditional X gates disabled"),
            ("corrected_accuracy", "Independent: all true X controls"),
        )
    ):
        ax.bar(
            positions + (offset - 1) * 0.25,
            [100 * r[field] for r in comparisons],
            width=0.25,
            label=label,
        )
    ax.set(
        xticks=positions,
        xticklabels=[
            f"{r['arm']}\nconcept {r['concept_epochs']} / label {r['head_epochs']}"
            for r in comparisons
        ],
        ylabel="Validation label accuracy (%)",
        ylim=(0, 100),
    )
    ax.legend(fontsize=8)
    figures.append((figure, "budget_comparison"))
    names = []
    for figure, stem in figures:
        figure.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            name = f"{stem}.{suffix}"
            figure.savefig(shared.output / name, dpi=180, bbox_inches="tight")
            names.append(name)
        plt.close(figure)
    return names


def summarize(shared, training: dict, evaluations: list[dict]) -> None:
    rows, comparisons, concepts = [], [], []
    for value in evaluations:
        m = value["metrics"]
        rows.append(
            {
                **{
                    k: value[k]
                    for k in ("arm", "role", "training", "condition", "n_samples")
                },
                **{f"label_{k}": v for k, v in m["label"].items()},
                **{
                    f"concept_{k}": v
                    for k, v in m["concept"].items()
                    if k not in ("per_concept", "max_normalization_error")
                },
            }
        )
    lookup = {(r["arm"], r["role"], r["training"], r["condition"]): r for r in rows}
    for arm in ("baseline", *ARMS):
        normal = lookup[(arm, "validation", "independent", "measured")]
        zero = lookup[(arm, "validation", "no_feedback", "zero")]
        corrected = lookup[(arm, "validation", "independent", "correct_all_five")]
        front, head = budgets(shared, arm)
        comparisons.append(
            {
                "arm": arm,
                "concept_epochs": front,
                "head_epochs": head,
                "normal_accuracy": normal["label_accuracy"],
                "zero_accuracy": zero["label_accuracy"],
                "corrected_accuracy": corrected["label_accuracy"],
                "conditional_x_gain_pp": 100
                * (normal["label_accuracy"] - zero["label_accuracy"]),
                "correction_gain_pp": 100
                * (corrected["label_accuracy"] - normal["label_accuracy"]),
                "normal_change_from_baseline_pp": 100
                * (
                    normal["label_accuracy"]
                    - lookup[("baseline", "validation", "independent", "measured")][
                        "label_accuracy"
                    ]
                ),
                **{k: v for k, v in normal.items() if k.startswith("concept_")},
            }
        )
        value = next(
            v
            for v in evaluations
            if (v["arm"], v["role"], v["training"], v["condition"])
            == (arm, "validation", "independent", "measured")
        )
        concepts += [
            {"arm": arm, "concept": name, **v}
            for name, v in value["metrics"]["concept"]["per_concept"].items()
        ]
    for row in comparisons:
        row["conditional_x_gain_change_pp"] = (
            row["conditional_x_gain_pp"] - comparisons[0]["conditional_x_gain_pp"]
        )
        row["correction_gain_change_pp"] = (
            row["correction_gain_pp"] - comparisons[0]["correction_gain_pp"]
        )
    updates = [
        {
            "job": name,
            "start_epoch": r["start_epoch"],
            "end_epoch": r["epochs"],
            "start_step": r["start_step"],
            "end_step": r["global_step"],
            "new_updates": r["new_updates"],
            "fresh_adam": r["fresh_adam"],
            "additional_training_seconds": r["additional_training_seconds"],
        }
        for name, r in training.items()
    ]
    for name, values in (
        ("comparison.csv", comparisons),
        ("concept_results.csv", concepts),
        ("condition_results.csv", [r for r in rows if r["role"] == "validation"]),
        ("training_diagnostics.csv", [r for r in rows if r["role"] == "train"]),
        ("training_budgets.csv", updates),
    ):
        write_csv(shared.output / name, values)
    atomic_json(
        shared.output / "summary.json",
        {
            "status": "complete",
            "seed": shared.reference.config.seed,
            "engineering_subset": shared.config.development,
            "test_evaluated": False,
            "test_read": False,
            "n_train": len(shared.data["train"]["labels"]),
            "n_validation": len(shared.data["validation"]["labels"]),
            "comparisons": comparisons,
            "conditions": rows,
            "concepts": concepts,
            "training": updates,
            "completed_training_cells": len(training),
            "completed_conditions": len(rows),
            "new_adam_updates": sum(r["new_updates"] for r in updates),
            "interpretation": (
                "Single-seed duration diagnostic. Both variants measure; only "
                "conditional X control differs. Long-concept training also "
                "changes retained B states."
            ),
        },
    )
    lines = [
        "# Robot：延长概念训练与延长 label 训练",
        "",
        f"seed 0；验证集 {len(shared.data['validation']['labels'])} 张；"
        f"开发运行：{shared.config.development}；测试集未使用。",
        "",
        "固定原电路、损失、学习率及 batch；比较预先指定的最终训练点。"
        "原基准只复用，不重训。",
        "",
        "| 配置 | 概念轮数 | label轮数 | Independent正常 | 关闭条件X门 | "
        "全概念纠正 | 条件X增益(pp) | 纠正增益(pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparisons:
        lines.append(
            f"| {r['arm']} | {r['concept_epochs']} | {r['head_epochs']} | "
            f"{r['normal_accuracy']:.2%} | {r['zero_accuracy']:.2%} | "
            f"{r['corrected_accuracy']:.2%} | {r['conditional_x_gain_pp']:+.2f} | "
            f"{r['correction_gain_pp']:+.2f} |"
        )
    lines += [
        "",
        "long_concept：从原概念终点续训，随后两份后半段使用原始初始权重、"
        "新 Adam 和原来的后半段样本顺序。",
        "long_label：继续使用原概念终点，分别恢复原 Independent／"
        "关闭 X 门模型及完整 Adam。",
        "",
        "condition_results.csv 保留三种预算各八个验证条件；"
        "training_diagnostics.csv 记录相同控制条件下的训练集表现。",
        "请将训练集 measured 与验证集 measured 比较、true 与 true 比较，"
        "避免把控制条件差异解释为过拟合。",
        "",
        "概念指标分别报告边缘阈值全部正确、联合 MAP、真实记录单次测量概率；"
        "不把纠正后的真实控制计为预测成功。",
        "所有正负变化均保留。原设置延长到当前预算无改善，不等于模型已达最优，也不能证明整个方法不可行。",
        "",
    ]
    (shared.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    plots(shared, comparisons)
    names = [
        str(p.relative_to(shared.output))
        for p in shared.output.rglob("*")
        if p.is_file()
        and p.name
        not in {"heartbeat.json", "evaluation.log", ".worker.lock", "result_lock.json"}
    ]
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {name: sha256(shared.output / name) for name in sorted(names)},
        },
    )
