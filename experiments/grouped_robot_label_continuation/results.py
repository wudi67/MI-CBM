"""Compare both fixed label budgets and plot true-control validation diagnostics."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.results import write_csv


def diagnostic_rows(shared, evaluations: list[dict]) -> list[dict]:
    source_epoch = shared.reference.config.head_epochs
    points = []
    for cell, name in (
        ("independent", "measured"),
        ("independent", "correct_all_five"),
        ("no_feedback", "zero"),
    ):
        baseline = next(
            r
            for r in evaluations
            if (r["arm"], r["role"], r["training"], r["condition"])
            == ("baseline", "validation", cell, name)
        )
        points.append(
            {
                "epoch": source_epoch,
                "training": cell,
                "condition": name,
                **baseline["metrics"]["label"],
            }
        )
        history = read_json(
            shared.output / "long_label/training" / cell / "history.json"
        )
        for row in history:
            if row["epoch"] <= source_epoch:
                continue
            key = "validation_true" if name == "correct_all_five" else "validation"
            if key not in row:
                continue
            points.append(
                {
                    "epoch": row["epoch"],
                    "training": cell,
                    "condition": name,
                    **row[key]["label"],
                }
            )
    return points


def plots(shared, comparisons: list[dict], points: list[dict]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for cell in ("independent", "no_feedback"):
        history = read_json(
            shared.output / "long_label/training" / cell / "history.json"
        )
        axes[0].plot(
            [r["epoch"] for r in history],
            [r["train_loss"] for r in history],
            label=f"{cell}: {'true' if cell == 'independent' else 'zero'} control",
        )
    for cell, condition in (
        ("independent", "measured"),
        ("independent", "correct_all_five"),
        ("no_feedback", "zero"),
    ):
        rows = [
            r for r in points if r["training"] == cell and r["condition"] == condition
        ]
        style = "o-" if condition == "correct_all_five" else "-"
        label = (
            "all concepts corrected" if condition == "correct_all_five" else condition
        )
        for ax, metric, scale in ((axes[1], "accuracy", 100), (axes[2], "bce", 1)):
            ax.plot(
                [r["epoch"] for r in rows],
                [scale * r[metric] for r in rows],
                style,
                label=label,
            )
    for ax, title in zip(
        axes,
        ("Label training BCE", "Validation label accuracy (%)", "Validation label BCE"),
        strict=True,
    ):
        ax.axvline(
            shared.reference.config.head_epochs, linestyle=":", color="black", alpha=0.4
        )
        ax.set(xlabel="Label epoch", ylabel=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    figure.suptitle(
        f"Robot seed 0: frozen concept epoch {shared.reference.actual_concept_epochs}"
    )
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            shared.output / f"learning_curves.{suffix}", dpi=180, bbox_inches="tight"
        )
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(8, 4.5))
    for i, (field, title) in enumerate(
        (
            ("normal_accuracy", "measured controls"),
            ("corrected_accuracy", "all concepts corrected"),
            ("zero_accuracy", "conditional X disabled"),
        )
    ):
        ax.bar(
            [j + (i - 1) * 0.25 for j in range(2)],
            [100 * r[field] for r in comparisons],
            width=0.25,
            label=title,
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=[f"label epoch {r['head_epochs']}" for r in comparisons],
        ylabel="Validation label accuracy (%)",
        ylim=(0, 100),
    )
    ax.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            shared.output / f"budget_comparison.{suffix}", dpi=180, bbox_inches="tight"
        )
    plt.close(figure)


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
    for arm in ("baseline", "long_label"):
        normal = lookup[(arm, "validation", "independent", "measured")]
        corrected = lookup[(arm, "validation", "independent", "correct_all_five")]
        zero = lookup[(arm, "validation", "no_feedback", "zero")]
        comparisons.append(
            {
                "arm": arm,
                "concept_epochs": shared.reference.actual_concept_epochs,
                "head_epochs": shared.reference.config.head_epochs
                if arm == "baseline"
                else shared.config.head_epochs,
                **{
                    f"{name}_{metric}": value[f"label_{metric}"]
                    for name, value in (
                        ("normal", normal),
                        ("corrected", corrected),
                        ("zero", zero),
                    )
                    for metric in ("accuracy", "bce")
                },
                "conditional_x_gain_pp": 100
                * (normal["label_accuracy"] - zero["label_accuracy"]),
                "correction_gain_pp": 100
                * (corrected["label_accuracy"] - normal["label_accuracy"]),
            }
        )
        value = next(
            v
            for v in evaluations
            if (v["arm"], v["role"], v["training"], v["condition"])
            == (arm, "validation", "independent", "measured")
        )
        concepts += [
            {"arm": arm, "concept": name, **metrics}
            for name, metrics in value["metrics"]["concept"]["per_concept"].items()
        ]
    for r in comparisons:
        for key in ("normal", "corrected", "zero"):
            r[f"{key}_change_pp"] = 100 * (
                r[f"{key}_accuracy"] - comparisons[0][f"{key}_accuracy"]
            )
    updates = [
        {
            "job": name,
            **{
                k: r[k]
                for k in (
                    "start_epoch",
                    "epochs",
                    "start_step",
                    "global_step",
                    "new_updates",
                    "fresh_adam",
                    "additional_training_seconds",
                )
            },
        }
        for name, r in training.items()
    ]
    points = diagnostic_rows(shared, evaluations)
    for name, values in (
        ("comparison.csv", comparisons),
        ("condition_results.csv", [r for r in rows if r["role"] == "validation"]),
        ("training_diagnostics.csv", [r for r in rows if r["role"] == "train"]),
        ("concept_results.csv", concepts),
        ("training_budgets.csv", updates),
        ("intervention_learning_curve.csv", points),
    ):
        write_csv(shared.output / name, values)
    atomic_json(
        shared.output / "summary.json",
        {
            "status": "complete",
            "seed": shared.reference.config.seed,
            "engineering_subset": shared.config.development,
            "test_read": False,
            "test_evaluated": False,
            "n_train": len(shared.data["train"]["labels"]),
            "n_validation": len(shared.data["validation"]["labels"]),
            "concept_epochs": shared.reference.actual_concept_epochs,
            "completed_training_cells": 2,
            "completed_conditions": len(rows),
            "new_adam_updates": sum(r["new_updates"] for r in updates),
            "comparisons": comparisons,
            "conditions": rows,
            "training": updates,
            "concepts": concepts,
            "intervention_learning_curve": points,
            "interpretation": (
                "Single-seed fixed-budget validation. Both circuits retain "
                "measurement; only conditional X control differs. No guarantee "
                "that BCE and accuracy improve together."
            ),
        },
    )
    lines = [
        "# Robot：固定概念 300 轮，延长 label 训练",
        "",
        f"实际概念轮数：{shared.reference.actual_concept_epochs}；seed 0；"
        f"验证集 {len(shared.data['validation']['labels'])} 张；"
        f"开发运行：{shared.config.development}。测试集未读取。",
        "",
        "| label 轮数 | 正常准确率 | 全概念纠正 | 关闭条件 X | "
        "纠正增益(pp) | 条件 X 增益(pp) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparisons:
        lines.append(
            f"| {r['head_epochs']} | {r['normal_accuracy']:.2%} | "
            f"{r['corrected_accuracy']:.2%} | {r['zero_accuracy']:.2%} | "
            f"{r['correction_gain_pp']:+.2f} | {r['conditional_x_gain_pp']:+.2f} |"
        )
    lines += [
        "",
        "两份后半段分别恢复 long_concept 的原模型、完整 Adam、随机数和样本顺序。"
        "前半段始终冻结，概念概率与基准核对一致。",
        "",
        "comparison.csv 同时报准确率和 BCE；training_diagnostics.csv "
        "提供匹配控制方式的训练集指标。intervention_learning_curve.csv "
        "包含起点及预先指定轮数的全概念纠正验证指标。",
        "",
        "训练损失的下降不能代替纠正准确率上升。所有正负变化均报告；"
        "300 轮无改善也不证明模型达到最优。",
        "",
    ]
    (shared.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    plots(shared, comparisons, points)
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
