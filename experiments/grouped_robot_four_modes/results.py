"""Four-mode validation tables, feedback/intervention contrasts and result locks."""

from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_robot_independent.results import statistics, write_csv
from experiments.grouped_robot_pilot.protocol import read_json

from .protocol import CELLS, CONDITIONS

CONTRASTS = {
    "joint_vs_standard": (("joint", "measured"), ("standard", "measured")),
    "independent_vs_standard": (("independent", "measured"), ("standard", "measured")),
    "sequential_vs_standard": (("sequential", "measured"), ("standard", "measured")),
    "joint_vs_independent": (("joint", "measured"), ("independent", "measured")),
    "joint_feedback": (("joint", "measured"), ("joint_no_feedback", "zero")),
    "joint_correction": (("joint", "correct_all_five"), ("joint", "measured")),
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].bar(
        [f"{r['cell']}\n{r['condition']}" for r in rows],
        [100 * r["mean"] for r in rows],
        yerr=[100 * (r["sample_std"] or 0) for r in rows],
        capsize=4,
    )
    axes[0].set_ylabel("Validation label accuracy (%)")
    axes[0].tick_params(axis="x", labelsize=8)
    for name in (
        "joint_vs_standard",
        "joint_feedback",
        "joint_correction",
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
    diagnostics = [dict(row) for row in rows if row["cell"] == "standard"]
    for row in rows:
        row["concept_supervised"] = row["cell"] != "standard"
        if row["cell"] == "standard":
            for name in (
                "mean_bit_accuracy",
                "all_concepts_accuracy",
                "joint_map_accuracy",
                "joint_single_shot_probability",
                "joint_nll",
            ):
                row[name] = None
    write_csv(shared.output / "standard_concept_diagnostics.csv", diagnostics)
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
                if cell == "standard" and not name.startswith("label_"):
                    continue
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
        "schema": "grouped_robot_four_modes.v1",
        "status": "complete",
        "development": shared.config.development,
        "seeds": list(shared.config.seed_values),
        "completed_training_cells": len(training),
        "completed_conditions": len(rows),
        "new_adam_updates": sum(r["global_step"] for r in training),
        "reused_label_models": 3 * len(shared.config.seed_values),
        "reused_concept_models": len(shared.config.seed_values),
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
    main_rows = [
        r
        for r in rows
        if r["condition"] == "measured"
        and r["cell"] in ("standard", "independent", "sequential", "joint")
    ]
    write_csv(shared.output / "four_modes.csv", main_rows)
    curves = []
    for seed in shared.config.seed_values:
        for cell in (*CELLS, "sequential", "independent", "no_feedback"):
            path = (
                shared.output / f"seed_{seed}/training/{cell}/endpoint.pt"
                if cell in CELLS
                else shared.reference.checkpoint_path(seed, cell)
            )
            for row in read_json(path.parent / "history.json"):
                curves.append(
                    {
                        "seed": seed,
                        "cell": cell,
                        "reused": cell not in CELLS,
                        "epoch": row["epoch"],
                        "order_epoch": row["order_epoch"],
                        "train_loss": row["train_loss"],
                        "train_concept_nll": row.get("train_concept_nll"),
                        "train_label_bce": row.get("train_label_bce"),
                        "validation_label_accuracy": row["validation"]["label"][
                            "accuracy"
                        ],
                        "validation_label_bce": row["validation"]["label"]["bce"],
                    }
                )
    write_csv(shared.output / "learning_curves.csv", curves)
    lines = [
        "# Robot four-mode five-seed comparison",
        "",
        "Train / validation only. Test has not been evaluated.",
        "",
        f"Seeds: {list(shared.config.seed_values)}; "
        f"new full-circuit trainings: {len(training)}; "
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
        "Standard trains label BCE only; its intermediate bits are unsupervised. "
        "Concept metrics are diagnostic and excluded from semantic main tables; "
        "no Standard concept intervention is performed.",
        "Joint uses concept joint-record NLL plus label BCE; "
        "both circuit modules train. "
        "All three new routes start from identical original parameters within seed.",
        "Full-circuit epochs match concept + label total updates, "
        "not per-module update counts or wall time.",
        "Normal inference uses all 32 Born branches without MAP substitution. "
        "All-concept correction replaces X controls while retaining physical states.",
        "Joint no-feedback is trained jointly from scratch with zero X controls. "
        "It retains measurement and tests the contribution of "
        "measurement-record feedback.",
        "Independent, Sequential and their frozen-frontend no-feedback predictions "
        "are read-only references, not newly evaluated/trained models.",
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
