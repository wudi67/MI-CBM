"""Four-mode tables, matched feedback effects and full intervention/shot exports."""

from __future__ import annotations

import statistics

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_feedback_ablation.protocol import read_json
from experiments.grouped_shots_final.results import (
    aggregate,
    evaluation_rows,
    plot_series,
    plt,
    save_figure,
    write_csv,
)

MODES = ("standard", "independent", "sequential", "joint")
SEMANTIC_MODES = MODES[1:]


def paired_rows(rows: list[dict]) -> list[dict]:
    lookup = {
        (r["seed"], r["training"], r["control_mode"], r["shots"]): r for r in rows
    }
    effects = []
    for normal in rows:
        training = normal["training"]
        if training not in SEMANTIC_MODES or normal["control_mode"] != "measured":
            continue
        seed, shots = normal["seed"], normal["shots"]
        for effect in ("feedback", "shape", "scale", "both"):
            if effect == "feedback":
                baseline = "joint_no_feedback" if training == "joint" else "no_feedback"
                a, b = normal, lookup[(seed, baseline, "zero", shots)]
            else:
                a, b = lookup[(seed, training, effect, shots)], normal
            effects.append(
                {
                    "role": normal["role"],
                    "seed": seed,
                    "training": training,
                    "effect": effect,
                    "shots": shots,
                    "accuracy_gain_pp": 100
                    * (a["label_accuracy"] - b["label_accuracy"]),
                    "balanced_accuracy_gain_pp": 100
                    * (a["label_balanced_accuracy"] - b["label_balanced_accuracy"]),
                    "bce_change": a["label_bce"] - b["label_bce"],
                }
            )
    return effects


def plots(root, summary: dict, budgets: list[int]) -> list[str]:
    figure, ax = plt.subplots(figsize=(7.2, 4.6))
    for training in MODES:
        rows = [
            r
            for r in summary["conditions"]
            if r["training"] == training and r["control_mode"] == "measured"
        ]
        plot_series(ax, rows, "label_accuracy", training, budgets)
    ax.set_ylabel("Label accuracy (%)")
    ax.set_title("Four training modes; dashed lines: exact probabilities")
    ax.legend()
    files = save_figure(figure, root, "four_modes_shots")
    figure, ax = plt.subplots(figsize=(7.2, 4.6))
    for training in SEMANTIC_MODES:
        rows = [
            r
            for r in summary["correction_counts"]
            if r["training"] == training and r["shots"] == 0
        ]
        rows.sort(key=lambda r: r["corrected_concepts"])
        ax.errorbar(
            [r["corrected_concepts"] for r in rows],
            [100 * r["label_accuracy"]["mean"] for r in rows],
            yerr=[100 * r["label_accuracy"]["std"] for r in rows],
            fmt="o-",
            capsize=3,
            label=training,
        )
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Corrected concepts (one = equal mean of shape and scale)")
    ax.set_ylabel("Label accuracy (%)")
    ax.legend()
    ax.grid(alpha=0.2)
    return files + save_figure(figure, root, "four_modes_interventions")


def summarize_stage(shared, evaluations: list[dict]) -> None:
    root = shared.output / shared.stage
    averages, repetitions = evaluation_rows(evaluations)
    effects = paired_rows(averages)
    conditions = aggregate(
        averages,
        ("training", "control_mode", "shots"),
        tuple(k for k in averages[0] if k.startswith(("label_", "concept_"))),
    )
    lookup = {
        (r["seed"], r["training"], r["control_mode"], r["shots"]): r for r in averages
    }
    corrections = []
    for row in averages:
        if row["training"] not in SEMANTIC_MODES or row["control_mode"] != "measured":
            continue
        for count, controls in (
            (0, ("measured",)),
            (1, ("shape", "scale")),
            (2, ("both",)),
        ):
            corrections.append(
                {
                    **{k: row[k] for k in ("role", "seed", "training", "shots")},
                    "corrected_concepts": count,
                    "label_accuracy": statistics.mean(
                        lookup[(row["seed"], row["training"], mode, row["shots"])][
                            "label_accuracy"
                        ]
                        for mode in controls
                    ),
                }
            )
    summary = {
        "status": "complete",
        "role": shared.stage,
        "n_seeds": len(shared.config.seed_list()),
        "test_evaluated": shared.stage == "test",
        "engineering_subset": shared.config.development,
        "conditions": conditions,
        "effects": aggregate(
            effects,
            ("training", "effect", "shots"),
            ("accuracy_gain_pp", "balanced_accuracy_gain_pp", "bce_change"),
        ),
        "correction_counts": aggregate(
            corrections,
            ("training", "shots", "corrected_concepts"),
            ("label_accuracy",),
        ),
        "standard_concepts": "Unsupervised bits; diagnostic only",
        "test_history": "Extension on existing test split; not a new untouched holdout",
        "replication": (
            "Sample SD across training seeds; average MC repetitions within seed"
        ),
    }
    files = []
    for name, rows in (
        ("condition_results.csv", averages),
        ("sampling_repeats.csv", repetitions),
        ("paired_results.csv", effects),
        ("correction_count_results.csv", corrections),
    ):
        write_csv(root / name, rows)
        files.append(name)
    # Main table omits concept claims for the label-only Standard baseline.
    main = []
    for row in conditions:
        if (
            row["training"] not in MODES
            or row["control_mode"] != "measured"
            or row["shots"] != 0
        ):
            continue
        item = {
            "training": row["training"],
            "n_seeds": row["n_seeds"],
            "label_accuracy_mean": row["label_accuracy"]["mean"],
            "label_accuracy_std": row["label_accuracy"]["std"],
            "concept_joint_map_mean": row["concept_joint_map_accuracy"]["mean"]
            if row["training"] != "standard"
            else "N/A",
            "concept_joint_map_std": row["concept_joint_map_accuracy"]["std"]
            if row["training"] != "standard"
            else "N/A",
        }
        for name, metric in (
            ("shape", "shape_marginal_argmax_accuracy"),
            ("scale", "scale_marginal_argmax_accuracy"),
            ("group_exact", "group_argmax_exact_accuracy"),
            ("true_record_probability", "joint_single_shot_probability"),
        ):
            for statistic in ("mean", "std"):
                item[f"concept_{name}_{statistic}"] = (
                    row[f"concept_{metric}"][statistic]
                    if row["training"] != "standard"
                    else "N/A"
                )
        main.append(item)
    write_csv(root / "four_modes_main.csv", main)
    files.append("four_modes_main.csv")
    atomic_json(root / "summary.json", summary)
    lines = [
        f"# dSprites {shared.stage}: four training modes",
        "",
        "Exact probability results: mean ± sample SD across training seeds.",
        "",
        "| Mode | Label accuracy (%) | Concept group exact (%) | Joint MAP (%) |",
        "|---|---:|---:|---:|",
    ]
    for row in main:
        concept = (
            "N/A (no concept supervision)"
            if row["training"] == "standard"
            else (
                f"{100 * row['concept_joint_map_mean']:.2f} ± "
                f"{100 * row['concept_joint_map_std']:.2f}"
            )
        )
        grouped = (
            "N/A"
            if row["training"] == "standard"
            else (
                f"{100 * row['concept_group_exact_mean']:.2f} ± "
                f"{100 * row['concept_group_exact_std']:.2f}"
            )
        )
        lines.append(
            f"| {row['training']} | {100 * row['label_accuracy_mean']:.2f} ± "
            f"{100 * row['label_accuracy_std']:.2f} | {grouped} | {concept} |"
        )
    lines += ["", "| Mode | Effect | Label accuracy change (pp) |", "|---|---|---:|"]
    for row in summary["effects"]:
        if row["shots"] == 0:
            metric = row["accuracy_gain_pp"]
            lines.append(
                f"| {row['training']} | {row['effect']} | "
                f"{metric['mean']:+.2f} ± {metric['std']:.2f} |"
            )
    lines += [
        "",
        "Joint uses its own jointly trained no-feedback comparator. Independent and "
        "Sequential reuse the frozen-frontend no-feedback comparator. "
        "Both sides retain measurement; this isolates record-conditioned X feedback, "
        "not measurement collapse alone. All signs of effects are retained.",
        "",
        "Independent/Sequential retain input-dependent B states. These are "
        "adapted modes, not strict concept-only bottlenecks. Standard uses "
        "label BCE only. End-to-end modes receive 200 epochs; staged modes use "
        "100 concept plus 100 classification epochs; per-module updates differ.",
        "",
        "This extends an already used test split. No endpoint, seed or mode is "
        "selected from these test results. MC draws are not training seeds.",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    files += [
        "summary.json",
        "summary.md",
        "data_lock.json",
        *plots(root, summary, shared.config.shot_list()),
    ]
    files += [
        str(p.relative_to(root)) for p in root.glob("seed*/*/*/evaluation_lock.json")
    ]
    atomic_json(
        root / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {name: sha256(root / name) for name in files},
        },
    )


def finish_pipeline(shared) -> None:
    if shared.verify_report_lock(shared.output) is not None:
        return
    cells = read_json(shared.output / "training_lock.json")["cells"]
    atomic_json(
        shared.output / "summary.json",
        {
            "status": "complete",
            "test_evaluated": not shared.config.development,
            "engineering_subset": shared.config.development,
            "trained_cells": sum(row["origin"] == "trained" for row in cells),
            "reused_cells": sum(row["origin"] == "historical_seed0" for row in cells),
            "formal_report": f"{shared.final_role}/summary.md",
        },
    )
    (shared.output / "summary.md").write_text(
        "# dSprites four-mode extension complete\n\n"
        f"Development: {shared.config.development}. "
        f"Real test evaluated: {not shared.config.development}.\n\n"
        "- [Training registry](training_lock.json)\n"
        "- [Validation](validation/summary.md)\n"
        f"- [Final results]({shared.final_role}/summary.md)\n",
        encoding="utf-8",
    )
    names = [
        "summary.json",
        "summary.md",
        "training_lock.json",
        "protocol_lock.json",
        "test_access.json",
        "validation/result_lock.json",
        f"{shared.final_role}/result_lock.json",
    ]
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {name: sha256(shared.output / name) for name in names},
        },
    )
