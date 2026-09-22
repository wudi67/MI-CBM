"""Seed-level summaries, paired gains, Monte Carlo variation and exportable plots."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256

from .protocol import CONDITIONS


def stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot export an empty evaluation table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten(metric: dict) -> dict:
    return {
        f"{group}_{key}": value
        for group in ("label", "concept")
        for key, value in metric[group].items()
        if key != "max_normalization_error"
    }


def evaluation_rows(evaluations: list[dict]) -> tuple[list[dict], list[dict]]:
    raw_rows, averages = [], []
    for item in evaluations:
        identity = {k: item[k] for k in ("role", "seed", "training", "control_mode")}
        exact = {**identity, "shots": 0, **flatten(item["exact"])}
        averages.append({**exact, "mc_label_accuracy_std": 0.0})
        grouped = defaultdict(list)
        for row in item["repeats"]:
            raw_rows.append(
                {
                    **identity,
                    "shots": row["shots"],
                    "repeat": row["repeat"],
                    "sampling_seed": row["sampling_seed"],
                    **flatten(row),
                }
            )
            grouped[row["shots"]].append(flatten(row))
        for shots, rows in grouped.items():
            averages.append(
                {
                    **identity,
                    "shots": shots,
                    **{key: statistics.mean(r[key] for r in rows) for key in rows[0]},
                    "mc_label_accuracy_std": stats([r["label_accuracy"] for r in rows])[
                        "std"
                    ],
                }
            )
    return averages, raw_rows


def paired_rows(averages: list[dict], repeats: list[dict]) -> list[dict]:
    lookup = {
        (r["seed"], r["training"], r["control_mode"], r["shots"]): r for r in averages
    }
    sampling = defaultdict(dict)
    for r in repeats:
        sampling[(r["seed"], r["training"], r["control_mode"], r["shots"])][
            r["repeat"]
        ] = r
    rows = []
    for normal in averages:
        if normal["training"] == "no_feedback" or normal["control_mode"] != "measured":
            continue
        seed, training, shots = normal["seed"], normal["training"], normal["shots"]
        origin = (seed, training, "measured", shots)
        for effect in ("feedback", "shape", "scale", "both"):
            if effect == "feedback":
                target, baseline = origin, (seed, "no_feedback", "zero", shots)
            else:
                target, baseline = (seed, training, effect, shots), origin
            a, b = lookup[target], lookup[baseline]
            mc = [
                sampling[target][i]["label_accuracy"]
                - sampling[baseline][i]["label_accuracy"]
                for i in sampling[target]
            ]
            rows.append(
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
                    "mc_accuracy_gain_std_pp": 100 * stats(mc)["std"] if mc else 0.0,
                }
            )
    return rows


def aggregate(
    rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]
) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [
        {
            **dict(zip(keys, key, strict=True)),
            "n_seeds": len(items),
            **{metric: stats([x[metric] for x in items]) for metric in metrics},
        }
        for key, items in groups.items()
    ]


def plot_series(
    ax,
    rows: list[dict],
    metric: str,
    label: str,
    budgets: list[int],
    scale: float = 100,
) -> None:
    values = {r["shots"]: r[metric] for r in rows}
    means = [scale * values[n]["mean"] for n in budgets]
    spread = [scale * values[n]["std"] for n in budgets]
    (line,) = ax.plot(budgets, means, "o-", label=label)
    ax.fill_between(
        budgets,
        [m - s for m, s in zip(means, spread, strict=True)],
        [m + s for m, s in zip(means, spread, strict=True)],
        alpha=0.12,
        color=line.get_color(),
    )
    ax.axhline(
        scale * values[0]["mean"], linestyle="--", color=line.get_color(), alpha=0.65
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets, [str(n) for n in budgets])
    ax.set_xlabel("Shots per image")
    ax.grid(alpha=0.2)


def save_figure(figure, root: Path, name: str) -> list[str]:
    figure.tight_layout()
    names = []
    for suffix in ("png", "pdf", "svg"):
        path = root / f"{name}.{suffix}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        names.append(path.name)
    plt.close(figure)
    return names


def plots(root: Path, summary: dict, budgets: list[int]) -> list[str]:
    names = []
    conditions, effects = summary["conditions"], summary["effects"]
    figure, ax = plt.subplots(figsize=(7.2, 4.6))
    for training, mode in (
        ("independent", "measured"),
        ("sequential", "measured"),
        ("no_feedback", "zero"),
    ):
        rows = [
            r
            for r in conditions
            if r["training"] == training and r["control_mode"] == mode
        ]
        plot_series(
            ax, rows, "label_accuracy", training.replace("_", " ").title(), budgets
        )
    ax.set_ylabel("Label accuracy (%)")
    ax.set_title("Normal prediction (dashed: exact probabilities)")
    ax.legend()
    names += save_figure(figure, root, "classification_shots")
    figure, ax = plt.subplots(figsize=(7.2, 4.6))
    for training in ("independent", "sequential"):
        rows = [
            r
            for r in effects
            if r["training"] == training and r["effect"] == "feedback"
        ]
        plot_series(ax, rows, "accuracy_gain_pp", training.title(), budgets, scale=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Feedback minus no-feedback accuracy (pp)")
    ax.set_title("Measurement-record feedback ablation")
    ax.legend()
    names += save_figure(figure, root, "feedback_shots")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, training in zip(axes, ("independent", "sequential"), strict=True):
        for effect in ("shape", "scale", "both"):
            rows = [
                r
                for r in effects
                if r["training"] == training and r["effect"] == effect
            ]
            plot_series(ax, rows, "accuracy_gain_pp", effect.title(), budgets, scale=1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(training.title())
        ax.set_ylabel("Correction minus normal accuracy (pp)")
        ax.legend()
    names += save_figure(figure, root, "intervention_shots")
    figure, ax = plt.subplots(figsize=(7.2, 4.6))
    rows = [
        r
        for r in conditions
        if r["training"] == "independent" and r["control_mode"] == "measured"
    ]
    for metric, label in (
        ("concept_shape_marginal_argmax_accuracy", "Shape"),
        ("concept_scale_marginal_argmax_accuracy", "Scale"),
        ("concept_group_argmax_exact_accuracy", "Both marginal predictions correct"),
        ("concept_joint_map_accuracy", "Joint MAP"),
    ):
        plot_series(ax, rows, metric, label, budgets)
    ax.set_ylabel("Concept prediction accuracy (%)")
    ax.set_title("Shared pre-feedback frontend")
    ax.legend(fontsize=9)
    names += save_figure(figure, root, "concept_shots")
    return names


def summarize_stage(shared, evaluations: list[dict]) -> None:
    expected = len(CONDITIONS) * len(shared.config.seed_list())
    if len(evaluations) != expected or any(x is None for x in evaluations):
        raise ValueError("All planned conditions are required before stage reporting")
    root = shared.output / shared.stage
    averages, repeats = evaluation_rows(evaluations)
    paired = paired_rows(averages, repeats)
    metric_names = tuple(k for k in averages[0] if k.startswith(("label_", "concept_")))
    summary = {
        "status": "complete",
        "role": shared.stage,
        "test_evaluated": shared.stage == "test",
        "engineering_subset": shared.config.development,
        "n_samples": len(shared.data["labels"]),
        "training_seeds": shared.config.seed_list(),
        "shots": shared.config.shot_list(),
        "sampling_repetitions": shared.config.repeats,
        "conditions": aggregate(
            averages, ("training", "control_mode", "shots"), metric_names
        ),
        "effects": aggregate(
            paired,
            ("training", "effect", "shots"),
            ("accuracy_gain_pp", "balanced_accuracy_gain_pp", "bce_change"),
        ),
        "uncertainty": (
            "mean +/- sample SD across training seeds after averaging sampling "
            "repetitions; MC spread in CSV"
        ),
        "concept_scope": (
            "original predictions; corrections do not count as prediction successes"
        ),
        "scope": (
            "frozen ideal-simulator inference; no shot-noise training "
            "or hardware gate-noise experiment"
        ),
    }
    files = []
    for name, rows in (
        ("condition_results.csv", averages),
        ("sampling_repeats.csv", repeats),
        ("paired_results.csv", paired),
        (
            "concept_results.csv",
            [
                r
                for r in averages
                if r["training"] == "independent" and r["control_mode"] == "measured"
            ],
        ),
    ):
        write_csv(root / name, rows)
        files.append(name)
    # The one-concept budget averages Shape and Scale instead of picking a winner.
    corrected = []
    lookup = {
        (r["seed"], r["training"], r["control_mode"], r["shots"]): r for r in averages
    }
    for row in averages:
        if row["training"] == "no_feedback" or row["control_mode"] != "measured":
            continue
        for count, modes in (
            (0, ("measured",)),
            (1, ("shape", "scale")),
            (2, ("both",)),
        ):
            corrected.append(
                {k: row[k] for k in ("role", "seed", "training", "shots")}
                | {
                    "corrected_concepts": count,
                    "label_accuracy": statistics.mean(
                        lookup[(row["seed"], row["training"], m, row["shots"])][
                            "label_accuracy"
                        ]
                        for m in modes
                    ),
                }
            )
    write_csv(root / "correction_count_results.csv", corrected)
    files.append("correction_count_results.csv")
    atomic_json(root / "summary.json", summary)
    lines = [
        f"# dSprites {shared.stage}: shots and CBM evaluation",
        "",
        f"Five-seed formal protocol: {not shared.config.development}. "
        f"Test evaluated: {shared.stage == 'test'}.",
        "",
        "Exact results are mean ± sample SD across training seeds. "
        "Finite-shot results average repetitions within each training seed.",
        "",
        "| Training | Control | Label accuracy (%) |",
        "|---|---|---:|",
    ]
    for row in summary["conditions"]:
        if row["shots"] == 0:
            s = row["label_accuracy"]
            lines.append(
                f"| {row['training']} | {row['control_mode']} | "
                f"{100 * s['mean']:.2f} ± {100 * s['std']:.2f} |"
            )
    lines += [
        "",
        "| Training | Effect | Exact accuracy change (pp) |",
        "|---|---|---:|",
    ]
    for row in summary["effects"]:
        if row["shots"] == 0:
            s = row["accuracy_gain_pp"]
            lines.append(
                f"| {row['training']} | {row['effect']} | "
                f"{s['mean']:+.2f} ± {s['std']:.2f} |"
            )
    lines += [
        "",
        "Full shot budgets, concept metrics, BCE, balanced accuracy and Monte Carlo "
        "variation are in the CSV/JSON files. Both circuits retain measurement; "
        "the ablation removes record-conditioned X feedback. "
        "All positive and negative results are retained.",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    files += [
        "summary.json",
        "summary.md",
        *plots(root, summary, shared.config.shot_list()),
        "data_lock.json",
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
    value = {
        "status": "complete",
        "test_evaluated": not shared.config.development,
        "engineering_subset": shared.config.development,
        "stages": ["validation", shared.final_role],
        "training": "none; existing fixed final endpoints",
        "formal_report": f"{shared.final_role}/summary.md",
    }
    atomic_json(shared.output / "summary.json", value)
    (shared.output / "summary.md").write_text(
        "# Sequential evaluation complete\n\n"
        f"Development: {shared.config.development}. "
        f"Real test evaluated: {not shared.config.development}.\n\n"
        "- [Validation results](validation/summary.md)\n"
        f"- [Final results]({shared.final_role}/summary.md)\n\n"
        "Models, seeds, shot budgets and metrics were fixed before final evaluation. "
        "Monte Carlo repetitions are not additional training seeds.\n",
        encoding="utf-8",
    )
    names = [
        "summary.json",
        "summary.md",
        "protocol_lock.json",
        "test_access.json",
        "validation/result_lock.json",
        f"{shared.final_role}/result_lock.json",
    ]
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {n: sha256(shared.output / n) for n in names},
        },
    )
