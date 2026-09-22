"""Five-condition summaries, within-seed contrasts and immutable output locks."""

from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_robot_independent.results import statistics, write_csv
from experiments.grouped_robot_pilot.protocol import read_json

from .protocol import CONDITIONS

CONTRASTS = {
    "sequential_vs_independent": (
        ("sequential", "measured"),
        ("independent", "measured"),
    ),
    "sequential_feedback": (("sequential", "measured"), ("no_feedback", "zero")),
    "sequential_correction": (
        ("sequential", "correct_all_five"),
        ("sequential", "measured"),
    ),
    "independent_feedback": (("independent", "measured"), ("no_feedback", "zero")),
    "independent_correction": (
        ("independent", "correct_all_five"),
        ("independent", "measured"),
    ),
}


def plots(output: Path, aggregates: list[dict], paired: list[dict]) -> None:
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    rows = [
        r
        for r in aggregates
        if r["role"] == "validation" and r["metric"] == "label_accuracy"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(
        [f"{r['cell']}\n{r['condition']}" for r in rows],
        [100 * r["mean"] for r in rows],
        yerr=[100 * (r["sample_std"] or 0) for r in rows],
        capsize=4,
    )
    axes[0].set_ylabel("Validation label accuracy (%)")
    axes[0].tick_params(axis="x", labelsize=8)
    for name in (
        "sequential_vs_independent",
        "sequential_feedback",
        "sequential_correction",
    ):
        selected = [
            r for r in paired if r["role"] == "validation" and r["comparison"] == name
        ]
        axes[1].plot(
            [r["seed"] for r in selected],
            [r["accuracy_gain_pp"] for r in selected],
            "o-",
            label=name.replace("_", " "),
        )
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set(xlabel="Complete training seed", ylabel="Paired accuracy change (pp)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"validation_results.{extension}", dpi=180)
    plt.close(fig)


def write_results(shared, evaluations: list[dict], training: list[dict]) -> None:
    rows: list[dict] = []
    for item in evaluations:
        row = {
            k: item[k]
            for k in (
                "seed",
                "role",
                "cell",
                "condition",
                "n_samples",
                "reused",
                "evaluation_path",
                "predictions_path",
            )
        }
        row.update({"label_" + k: v for k, v in item["metrics"]["label"].items()})
        row.update(
            {
                k: item["metrics"]["concept"][k]
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
    paired: list[dict] = []
    for role in ("train", "validation"):
        for seed in shared.config.seed_values:
            selected = {
                (r["cell"], r["condition"]): r
                for r in rows
                if r["role"] == role and r["seed"] == seed
            }
            for name, (a, b) in CONTRASTS.items():
                left, right = selected[a], selected[b]
                paired.append(
                    {
                        "seed": seed,
                        "role": role,
                        "comparison": name,
                        "left_accuracy": left["label_accuracy"],
                        "right_accuracy": right["label_accuracy"],
                        "accuracy_gain_pp": 100
                        * (left["label_accuracy"] - right["label_accuracy"]),
                        "balanced_accuracy_gain_pp": 100
                        * (
                            left["label_balanced_accuracy"]
                            - right["label_balanced_accuracy"]
                        ),
                        "bce_change": left["label_bce"] - right["label_bce"],
                    }
                )
    metric_names = (
        "label_accuracy",
        "label_balanced_accuracy",
        "label_bce",
        "mean_bit_accuracy",
        "all_concepts_accuracy",
        "joint_map_accuracy",
        "joint_single_shot_probability",
        "joint_nll",
    )
    aggregates: list[dict] = []
    paired_summary: list[dict] = []
    for role in ("train", "validation"):
        for cell, condition, _ in CONDITIONS:
            selected = [
                r
                for r in rows
                if (r["role"], r["cell"], r["condition"]) == (role, cell, condition)
            ]
            for name in metric_names:
                aggregates.append(
                    {
                        "role": role,
                        "cell": cell,
                        "condition": condition,
                        "metric": name,
                        **statistics([r[name] for r in selected]),
                    }
                )
        for name in CONTRASTS:
            selected = [
                r for r in paired if r["role"] == role and r["comparison"] == name
            ]
            for metric in (
                "accuracy_gain_pp",
                "balanced_accuracy_gain_pp",
                "bce_change",
            ):
                paired_summary.append(
                    {
                        "role": role,
                        "comparison": name,
                        "metric": metric,
                        **statistics([r[metric] for r in selected]),
                    }
                )
    summary = {
        "schema": "grouped_robot_sequential.v1",
        "status": "complete",
        "development": shared.config.development,
        "seeds": list(shared.config.seed_values),
        "completed_training_cells": len(training),
        "completed_conditions": len(rows),
        "new_adam_updates": sum(r["global_step"] for r in training),
        "reused_label_models": 2 * len(training),
        "reused_concept_models": len(training),
        "reused_evaluations": sum(r["reused"] for r in rows),
        "statistical_unit": (
            "Complete training seed; sample SD uses n-1; gains are paired within seed"
        ),
        "selection": "All declared seeds, fixed final epoch",
        "test_read": False,
        "test_evaluated": False,
        "rows": rows,
        "aggregates": aggregates,
        "paired_rows": paired,
        "paired_summary": paired_summary,
        "training": training,
    }
    atomic_json(shared.output / "summary.json", summary)
    for name, values in (
        ("summary", rows),
        ("aggregates", aggregates),
        ("paired_gains", paired),
        ("paired_summary", paired_summary),
    ):
        write_csv(shared.output / f"{name}.csv", values)
    curves = []
    for seed in shared.config.seed_values:
        for cell in ("sequential", "independent", "no_feedback"):
            path = (
                shared.output / f"seed_{seed}/training/sequential/endpoint.pt"
                if cell == "sequential"
                else shared.reference.checkpoint_path(seed, cell)
            )
            for row in read_json(path.parent / "history.json"):
                curves.append(
                    {
                        "seed": seed,
                        "cell": cell,
                        "reused": cell != "sequential",
                        "epoch": row["epoch"],
                        "order_epoch": row["order_epoch"],
                        "train_loss": row["train_loss"],
                        "validation_label_accuracy": row["validation"]["label"][
                            "accuracy"
                        ],
                        "validation_label_bce": row["validation"]["label"]["bce"],
                    }
                )
    write_csv(shared.output / "learning_curves.csv", curves)
    lines = [
        "# Robot Sequential five-seed comparison",
        "",
        "Train / validation only. Test has not been evaluated.",
        "",
        f"Seeds: {list(shared.config.seed_values)}; "
        f"new label trainings: {len(training)}; "
        f"new Adam updates: {summary['new_adam_updates']}.",
        "",
        "| Validation model | Condition | Label accuracy (%) |",
        "|---|---|---:|",
    ]
    for row in aggregates:
        if row["role"] == "validation" and row["metric"] == "label_accuracy":
            sd = (
                "n/a" if row["sample_std"] is None else f"{100 * row['sample_std']:.3f}"
            )
            lines.append(
                f"| {row['cell']} | {row['condition']} | "
                f"{100 * row['mean']:.3f} ± {sd} |"
            )
    lines += [
        "",
        "| Paired validation comparison | Mean ± sample SD (pp) |",
        "|---|---:|",
    ]
    for row in paired_summary:
        if row["role"] == "validation" and row["metric"] == "accuracy_gain_pp":
            sd = "n/a" if row["sample_std"] is None else f"{row['sample_std']:.3f}"
            lines.append(f"| {row['comparison']} | {row['mean']:.3f} ± {sd} |")
    lines += [
        "",
        "Sample SD uses n-1; all predefined seeds and negative gains are retained.",
        "Sequential trains only the label circuit, from the same initial parameters "
        "and RNG as the reference, using measured controls.",
        "The shared frontend is frozen. "
        "Concept predictions must agree across all five conditions.",
        "Normal inference uses the exact Born-weighted 32-branch distribution, "
        "without MAP substitution.",
        "All-concept correction replaces classical X controls and preserves "
        "Born weights and retained states.",
        "The no-feedback baseline still contains measurement; "
        "this ablates measurement-record feedback.",
        "Independent and no-feedback checkpoints/predictions are referenced "
        "read-only. No control model is retrained.",
        "Architecture was selected using earlier validation results; "
        "this report is not untouched test evidence.",
        "",
    ]
    (shared.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    plots(shared.output, aggregates, paired)
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
