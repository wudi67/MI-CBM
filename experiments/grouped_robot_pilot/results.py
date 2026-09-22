"""Export all seed-zero results, including negative intervention effects."""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256

from .protocol import CELLS, read_json


def write_csv(path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(shared, rows: list[dict]) -> list[str]:
    names = []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cell in CELLS:
        history = read_json(shared.output / "training" / cell / "history.json")
        target = axes[0] if cell == "concept" else axes[1]
        target.plot(
            [r["epoch"] for r in history],
            [r["train_loss"] for r in history],
            label=cell,
        )
    for ax, title in zip(axes, ("Concept joint NLL", "Label BCE"), strict=True):
        ax.set(xlabel="Epoch within phase", ylabel=title)
        ax.grid(alpha=0.2)
        ax.legend()
    figure.suptitle("Robot P0 seed 0: training curves")
    figures = [(figure, "learning_curves")]
    figure, ax = plt.subplots(figsize=(10, 4.8))
    labels = [
        r["condition"].removeprefix("correct_")
        if r["training"] == "independent"
        else "no feedback"
        for r in rows
    ]
    ax.bar(labels, [100 * r["label_accuracy"] for r in rows])
    ax.set(
        ylabel="Validation label accuracy (%)",
        ylim=(0, 100),
        title="Robot P0 seed 0: independent controls and corrections",
    )
    ax.tick_params(axis="x", rotation=30)
    figures.append((figure, "interventions"))
    for figure, stem in figures:
        figure.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            name = f"{stem}.{suffix}"
            figure.savefig(shared.output / name, dpi=180, bbox_inches="tight")
            names.append(name)
        plt.close(figure)
    return names


def summarize(shared, training: dict, evaluations: list[dict]) -> None:
    rows = []
    for item in evaluations:
        m = item["metrics"]
        rows.append(
            {
                "training": item["training"],
                "condition": item["condition"],
                "correction_mask": item["correction_mask"],
                "n_samples": item["n_samples"],
                **{f"label_{k}": v for k, v in m["label"].items()},
                **{
                    f"concept_{k}": v
                    for k, v in m["concept"].items()
                    if k not in ("per_concept", "max_normalization_error")
                },
            }
        )
    lookup = {(r["training"], r["condition"]): r for r in rows}
    normal, zero = lookup[("independent", "measured")], lookup[("no_feedback", "zero")]
    effects = [
        {
            "effect": "feedback",
            "label_accuracy_gain_pp": 100
            * (normal["label_accuracy"] - zero["label_accuracy"]),
        }
    ]
    effects += [
        {
            "effect": row["condition"],
            "label_accuracy_gain_pp": 100
            * (row["label_accuracy"] - normal["label_accuracy"]),
        }
        for row in rows
        if row["training"] == "independent" and row["condition"] != "measured"
    ]
    concept = evaluations[0]["metrics"]["concept"]
    summary = {
        "status": "complete",
        "role": "validation",
        "seed": shared.config.seed,
        "engineering_subset": shared.config.development,
        "test_evaluated": False,
        "test_read": False,
        "n_samples": normal["n_samples"],
        "completed_training_cells": len(training),
        "completed_conditions": len(rows),
        "concept": concept,
        "conditions": rows,
        "effects": effects,
        "pairing": {
            "same_frontend": True,
            "same_initial_label_circuit": True,
            "same_sample_order": True,
            "fresh_adam_each_label_route": True,
        },
        "interpretation": (
            "Single-seed validation pilot; input-dependent B states "
            "retained. Both variants measure; feedback ablation removes "
            "only record-conditioned X gates."
        ),
    }
    files = []
    for name, values in (
        ("condition_results.csv", rows),
        ("paired_results.csv", effects),
        (
            "concept_results.csv",
            [{"concept": k, **v} for k, v in concept["per_concept"].items()],
        ),
    ):
        write_csv(shared.output / name, values)
        files.append(name)
    atomic_json(shared.output / "summary.json", summary)
    lines = [
        "# Robot P0：Independent 有反馈／无反馈及概念纠正",
        "",
        f"seed 0，验证集 {normal['n_samples']} 张。"
        f"开发子集：{shared.config.development}。测试集未使用。",
        "",
        "这是一种子验证结果，没有跨种子均值或标准差。",
        "",
        "| 模式 | 条件 | Label accuracy | Balanced accuracy | BCE |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['training']} | {row['condition']} | {row['label_accuracy']:.2%} | "
            f"{row['label_balanced_accuracy']:.2%} | {row['label_bce']:.4f} |"
        )
    lines += ["", "| 效应 | Label 变化（百分点） |", "|---|---:|"]
    for row in effects:
        lines.append(f"| {row['effect']} | {row['label_accuracy_gain_pp']:+.2f} |")
    lines += [
        "",
        "五概念分别以边缘概率 ≥0.5 判断后全部正确："
        f"{concept['all_concepts_accuracy']:.2%}；"
        f"联合 MAP：{concept['joint_map_accuracy']:.2%}；"
        f"真实五位组合的单次测量正确概率：{concept['joint_single_shot_probability']:.2%}。",
        "",
        "三个指标定义不同。纠正后的真实控制不计为原模型概念预测成功。",
        "",
        "有反馈模型训练后半段时使用真实概念，正常评价使用实际测量记录；"
        "无反馈模型使用零控制。两者保留相同测量与同一冻结前半段。"
        "全部纠正只替换控制记录，不重置后五位，也不筛选或改写测量分支概率。",
        "",
        "所有结果按固定训练终点报告，包括负向干预结果；"
        "本次不运行测试集、shots 扫描或五种子正式实验。",
        "",
    ]
    (shared.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    files += [
        "summary.json",
        "summary.md",
        "manifest.json",
        "data_lock.json",
        *plots(shared, rows),
    ]
    for cell, result in training.items():
        files += [
            f"training/{cell}/{name}" for name in ("result.json", *result["artifacts"])
        ]
    files += [
        str(p.relative_to(shared.output))
        for p in (shared.output / "validation").glob("*/*/evaluation_lock.json")
    ]
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {name: sha256(shared.output / name) for name in files},
        },
    )
