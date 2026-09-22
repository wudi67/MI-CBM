"""Per-seed metrics, paired gains, sample standard deviations and final locks."""

import csv
from statistics import mean, stdev

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_robot_pilot.protocol import read_json


def statistics(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else None,
    }


def write_csv(path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(output, aggregates: list[dict], pairs: list[dict]) -> None:
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    rows = [
        r
        for r in aggregates
        if r["role"] == "validation" and r["metric"] == "label_accuracy"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        [r["condition"] for r in rows],
        [100 * r["mean"] for r in rows],
        yerr=[100 * (r["sample_std"] or 0) for r in rows],
        capsize=4,
    )
    axes[0].set_ylabel("Validation label accuracy (%)")
    axes[0].tick_params(axis="x", labelrotation=15)
    validation = [r for r in pairs if r["role"] == "validation"]
    for field, label in (
        ("feedback_gain_pp", "Feedback gain"),
        ("correction_gain_pp", "Correction gain"),
    ):
        axes[1].plot(
            [r["seed"] for r in validation],
            [r[field] for r in validation],
            "o-",
            label=label,
        )
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set(xlabel="Complete training seed", ylabel="Paired accuracy change (pp)")
    axes[1].set_xticks([r["seed"] for r in validation])
    axes[1].legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"validation_results.{extension}", dpi=180)
    plt.close(fig)


def write_results(shared, evaluations: list[dict], training: list[dict]) -> None:
    rows: list[dict] = []
    concepts: list[dict] = []
    for evaluation in evaluations:
        concept, label = (
            evaluation["metrics"]["concept"],
            evaluation["metrics"]["label"],
        )
        row = {
            k: evaluation[k] for k in ("seed", "role", "cell", "condition", "n_samples")
        }
        row.update(
            {"label_" + k: label[k] for k in ("accuracy", "balanced_accuracy", "bce")}
        )
        row.update(
            {
                k: concept[k]
                for k in (
                    "mean_bit_accuracy",
                    "all_concepts_accuracy",
                    "joint_map_accuracy",
                    "joint_single_shot_probability",
                    "joint_nll",
                )
            }
        )
        rows.append(row)
        if evaluation["condition"] == "measured":
            for name, metric in concept["per_concept"].items():
                concepts.append(
                    {
                        "seed": row["seed"],
                        "role": row["role"],
                        "concept": name,
                        **{
                            k: metric[k]
                            for k in ("accuracy", "balanced_accuracy", "bce")
                        },
                    }
                )
    pairs: list[dict] = []
    for role in ("train", "validation"):
        for seed in shared.config.seed_values:
            items = {
                r["condition"]: r
                for r in rows
                if r["role"] == role and r["seed"] == seed
            }
            normal, corrected, zero = (
                items[k] for k in ("measured", "correct_all_five", "zero")
            )
            pairs.append(
                {
                    "seed": seed,
                    "role": role,
                    "normal_label_accuracy": normal["label_accuracy"],
                    "corrected_label_accuracy": corrected["label_accuracy"],
                    "no_feedback_label_accuracy": zero["label_accuracy"],
                    "feedback_gain_pp": 100
                    * (normal["label_accuracy"] - zero["label_accuracy"]),
                    "correction_gain_pp": 100
                    * (corrected["label_accuracy"] - normal["label_accuracy"]),
                    "feedback_balanced_gain_pp": 100
                    * (
                        normal["label_balanced_accuracy"]
                        - zero["label_balanced_accuracy"]
                    ),
                    "correction_balanced_gain_pp": 100
                    * (
                        corrected["label_balanced_accuracy"]
                        - normal["label_balanced_accuracy"]
                    ),
                }
            )
    aggregates = []
    metric_names = [
        k
        for k in rows[0]
        if k not in {"seed", "role", "cell", "condition", "n_samples"}
    ]
    for role in ("train", "validation"):
        for condition in ("measured", "correct_all_five", "zero"):
            selected = [
                r for r in rows if r["role"] == role and r["condition"] == condition
            ]
            for name in metric_names:
                aggregates.append(
                    {
                        "role": role,
                        "condition": condition,
                        "metric": name,
                        **statistics([r[name] for r in selected]),
                    }
                )
    paired_summary = {
        role: {
            name: statistics([r[name] for r in pairs if r["role"] == role])
            for name in (
                "feedback_gain_pp",
                "correction_gain_pp",
                "feedback_balanced_gain_pp",
                "correction_balanced_gain_pp",
            )
        }
        for role in ("train", "validation")
    }
    summary = {
        "schema": "grouped_robot_independent.v1",
        "status": "complete",
        "development": shared.config.development,
        "seeds": list(shared.config.seed_values),
        "completed_training_cells": len(training),
        "completed_conditions": len(rows),
        "reused_training_cells": sum(r["reused"] for r in training),
        "new_adam_updates": sum(r["global_step"] for r in training if not r["reused"]),
        "total_logical_adam_updates": sum(r["global_step"] for r in training),
        "test_read": False,
        "test_evaluated": False,
        "statistical_unit": (
            "Complete training seed; standard deviation uses n-1; "
            "paired gains computed within each seed"
        ),
        "selection": "Fixed final endpoint; all seeds included",
        "rows": rows,
        "aggregates": aggregates,
        "paired_rows": pairs,
        "paired_summary": paired_summary,
        "training": training,
    }
    atomic_json(shared.output / "summary.json", summary)
    write_csv(shared.output / "summary.csv", rows)
    write_csv(shared.output / "aggregates.csv", aggregates)
    write_csv(shared.output / "paired_gains.csv", pairs)
    write_csv(shared.output / "per_concept.csv", concepts)
    curves = []
    for item in training:
        history = read_json(
            shared.checkpoint_path(item["seed"], item["cell"]).parent / "history.json"
        )
        for row in history:
            curves.append(
                {
                    "seed": item["seed"],
                    "cell": item["cell"],
                    "reused": item["reused"],
                    "epoch": row["epoch"],
                    "order_epoch": row["order_epoch"],
                    "train_loss": row["train_loss"],
                    "validation_concept_exact": row["validation"]["concept"][
                        "all_concepts_accuracy"
                    ],
                    "validation_label_accuracy": row["validation"]
                    .get("label", {})
                    .get("accuracy", ""),
                }
            )
    write_csv(shared.output / "learning_curves.csv", curves)
    lines = [
        "# Robot Independent five-seed P0",
        "",
        "Train / validation only. Test has not been evaluated.",
        "",
        f"Complete training seeds: {list(shared.config.seed_values)}. "
        "Fixed final epochs; sample SD (n-1).",
        f"Training cells: {len(training)}, "
        f"reused: {summary['reused_training_cells']}, "
        f"new updates: {summary['new_adam_updates']}.",
        "",
        "| Validation condition | Label accuracy (%) |",
        "|---|---:|",
    ]
    for row in aggregates:
        if row["role"] == "validation" and row["metric"] == "label_accuracy":
            sd = (
                "n/a" if row["sample_std"] is None else f"{100 * row['sample_std']:.3f}"
            )
            lines.append(f"| {row['condition']} | {100 * row['mean']:.3f} ± {sd} |")
    lines += ["", "| Paired validation gain | Mean ± sample SD (pp) |", "|---|---:|"]
    for name in ("feedback_gain_pp", "correction_gain_pp"):
        metric = paired_summary["validation"][name]
        sd = "n/a" if metric["sample_std"] is None else f"{metric['sample_std']:.3f}"
        lines.append(f"| {name} | {metric['mean']:.3f} ± {sd} |")
    lines += [
        "",
        "The zero-control baseline retains measurement and the image-dependent "
        "residual state; this isolates feedback, not measurement itself.",
        "All-concept correction replaces classical controls before X gates; "
        "it preserves the physical Born branches.",
        "Normal label inference averages exact physical measurement branches; "
        "concept MAP accuracy is a separate diagnostic.",
        "Independent train loss uses true controls; epoch validation uses measured "
        "controls. Compare matching conditions in summary.csv.",
        "Architecture selection used earlier seed-zero validation results. "
        "These are validation results, not untouched test evidence.",
        "Reused concept checkpoints contribute only their trained frontend; "
        "their unused old backend is never copied into this model.",
        "",
    ]
    (shared.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    plots(shared.output, aggregates, pairs)
    excluded = {"heartbeat.json", "result_lock.json", "train.log", ".worker.lock"}
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "test_evaluated": False,
            "artifacts": {
                str(p.relative_to(shared.output)): sha256(p)
                for p in sorted(shared.output.rglob("*"))
                if p.is_file()
                and p.name not in excluded
                and not p.name.endswith(".tmp")
            },
        },
    )
