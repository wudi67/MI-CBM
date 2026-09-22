"""Seed-level inference reports; sampling repetitions are not training seeds."""

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_robot_four_modes.results import CONTRASTS
from experiments.grouped_robot_independent.results import write_csv

from .protocol import CONDITIONS, MODES, SUPERVISED

LABEL_METRICS = ("label_accuracy", "label_balanced_accuracy", "label_bce")
CONCEPT_METRICS = (
    "concept_mean_bit_accuracy",
    "concept_all_concepts_accuracy",
    "concept_joint_map_accuracy",
    "concept_joint_single_shot_probability",
    "concept_joint_nll",
)


def stats(values: list[float]) -> dict:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }


def flatten(metric: dict, standard: bool) -> dict:
    return {
        **{
            "label_" + k: metric["label"][k]
            for k in ("accuracy", "balanced_accuracy", "bce")
        },
        **{
            name: None if standard else metric["concept"][name.removeprefix("concept_")]
            for name in CONCEPT_METRICS
        },
    }


def evaluation_rows(
    evaluations: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    averages, repeated, diagnostics = [], [], []
    for item in evaluations:
        identity = {k: item[k] for k in ("role", "seed", "training", "control_mode")}
        standard = item["training"] == "standard"
        averages.append(
            {
                **identity,
                "shots": 0,
                **flatten(item["exact"], standard),
                "mc_label_accuracy_std": 0.0,
            }
        )
        if standard:
            diagnostics.append(
                {**identity, "shots": 0, "repeat": -1, **flatten(item["exact"], False)}
            )
        grouped = defaultdict(list)
        for row in item["repeats"]:
            values = flatten(row, standard)
            repeated.append(
                {
                    **identity,
                    "shots": row["shots"],
                    "repeat": row["repeat"],
                    "sampling_seed": row["sampling_seed"],
                    **values,
                }
            )
            grouped[row["shots"]].append(values)
            if standard:
                diagnostics.append(
                    {
                        **identity,
                        "shots": row["shots"],
                        "repeat": row["repeat"],
                        **flatten(row, False),
                    }
                )
        for shots, rows in grouped.items():
            averages.append(
                {
                    **identity,
                    "shots": shots,
                    **{
                        k: None if rows[0][k] is None else mean(r[k] for r in rows)
                        for k in rows[0]
                    },
                    "mc_label_accuracy_std": stats([r["label_accuracy"] for r in rows])[
                        "std"
                    ],
                }
            )
    return averages, repeated, diagnostics


def paired_rows(averages: list[dict], repeated: list[dict]) -> list[dict]:
    lookup = {
        (r["seed"], r["training"], r["control_mode"], r["shots"]): r for r in averages
    }
    sampling = defaultdict(dict)
    for r in repeated:
        sampling[r["seed"], r["training"], r["control_mode"], r["shots"]][
            r["repeat"]
        ] = r
    rows = []
    seeds = sorted({r["seed"] for r in averages})
    budgets = sorted({r["shots"] for r in averages})
    for seed in seeds:
        for shots in budgets:
            for comparison, (left, right) in CONTRASTS.items():
                akey, bkey = (seed, *left, shots), (seed, *right, shots)
                a, b = lookup[akey], lookup[bkey]
                mc = [
                    100
                    * (
                        sampling[akey][r]["label_accuracy"]
                        - sampling[bkey][r]["label_accuracy"]
                    )
                    for r in sampling[akey]
                ]
                rows.append(
                    {
                        "role": a["role"],
                        "seed": seed,
                        "shots": shots,
                        "comparison": comparison,
                        "left_accuracy": a["label_accuracy"],
                        "right_accuracy": b["label_accuracy"],
                        "accuracy_gain_pp": 100
                        * (a["label_accuracy"] - b["label_accuracy"]),
                        "balanced_accuracy_gain_pp": 100
                        * (a["label_balanced_accuracy"] - b["label_balanced_accuracy"]),
                        "bce_change": a["label_bce"] - b["label_bce"],
                        "mc_accuracy_gain_std_pp": stats(mc)["std"] if mc else 0.0,
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
            **{
                metric: None
                if items[0][metric] is None
                else stats([r[metric] for r in items])
                for metric in metrics
            },
        }
        for key, items in groups.items()
    ]


def plots(root: Path, summary: dict, budgets: list[int]) -> list[str]:
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    names = []

    def series(ax, rows, metric, label, scale=100):
        values = {r["shots"]: r[metric] for r in rows}
        y = [scale * values[n]["mean"] for n in budgets]
        sd = [scale * (values[n]["std"] or 0) for n in budgets]
        (line,) = ax.plot(budgets, y, "o-", label=label)
        ax.fill_between(
            budgets,
            [a - b for a, b in zip(y, sd, strict=True)],
            [a + b for a, b in zip(y, sd, strict=True)],
            alpha=0.12,
            color=line.get_color(),
        )
        ax.axhline(
            scale * values[0]["mean"], linestyle="--", color=line.get_color(), alpha=0.6
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(v) for v in budgets])
        ax.set_xlabel("Shots per image")
        ax.grid(alpha=0.2)

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            file = f"{name}.{suffix}"
            fig.savefig(root / file, dpi=180, bbox_inches="tight")
            names.append(file)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for mode in MODES:
        rows = [
            r
            for r in summary["conditions"]
            if r["training"] == mode and r["control_mode"] == "measured"
        ]
        series(ax, rows, "label_accuracy", mode.title())
    ax.set_ylabel("Label accuracy (%)")
    ax.set_title("Normal predictions; dashed lines: exact probabilities")
    ax.legend()
    save(fig, "classification_shots")
    for effect, title in (
        ("feedback", "Measurement-record feedback"),
        ("correction", "All five concepts corrected"),
    ):
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        for mode in SUPERVISED:
            rows = [
                r for r in summary["effects"] if r["comparison"] == f"{mode}_{effect}"
            ]
            series(ax, rows, "accuracy_gain_pp", mode.title(), scale=1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Paired accuracy change (pp)")
        ax.set_title(title)
        ax.legend()
        save(fig, f"{effect}_shots")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, mode in zip(
        axes, ("independent", "joint", "joint_no_feedback"), strict=True
    ):
        condition = "zero" if mode == "joint_no_feedback" else "measured"
        rows = [
            r
            for r in summary["conditions"]
            if (r["training"], r["control_mode"]) == (mode, condition)
        ]
        for metric, label in (
            ("concept_mean_bit_accuracy", "Mean bit"),
            ("concept_all_concepts_accuracy", "All five correct"),
            ("concept_joint_map_accuracy", "Joint MAP"),
        ):
            series(ax, rows, metric, label)
        ax.set_title(
            "Independent / Sequential"
            if mode == "independent"
            else mode.replace("_", " ").title()
        )
        ax.set_ylabel("Concept accuracy (%)")
        ax.legend(fontsize=8)
    save(fig, "concept_shots")
    return names


def summarize_stage(shared, evaluations: list[dict]) -> None:
    expected = {(s, t, c) for s in shared.config.seed_list() for t, c, _ in CONDITIONS}
    actual = {(r["seed"], r["training"], r["control_mode"]) for r in evaluations}
    if len(evaluations) != len(expected) or expected != actual:
        raise ValueError("All planned evaluation conditions are required")
    root = shared.output / shared.stage
    averages, repeated, diagnostics = evaluation_rows(evaluations)
    paired = paired_rows(averages, repeated)
    summary = {
        "status": "complete",
        "role": shared.stage,
        "test_read": shared.stage == "test",
        "test_evaluated": shared.stage == "test",
        "development": shared.config.development,
        "n_samples": len(shared.data["labels"]),
        "training_seeds": shared.config.seed_list(),
        "shots": shared.config.shot_list(),
        "sampling_repetitions": shared.config.repeats,
        "completed_conditions": len(evaluations),
        "conditions": aggregate(
            averages,
            ("training", "control_mode", "shots"),
            LABEL_METRICS + CONCEPT_METRICS,
        ),
        "effects": aggregate(
            paired,
            ("comparison", "shots"),
            ("accuracy_gain_pp", "balanced_accuracy_gain_pp", "bce_change"),
        ),
        "uncertainty": (
            "Mean and sample SD across training seeds after averaging repeats "
            "within seed; MC variation separate"
        ),
        "concept_scope": (
            "Original predicted concepts; corrected controls do not count as "
            "concept prediction successes; Standard diagnostic only"
        ),
        "scope": (
            "Frozen ideal-circuit finite-shot evaluation; "
            "no retraining or hardware gate noise"
        ),
    }
    concepts = [
        r
        for r in averages
        if (r["training"], r["control_mode"])
        in (
            ("independent", "measured"),
            ("joint", "measured"),
            ("joint_no_feedback", "zero"),
        )
    ]
    lookup = {
        (r["seed"], r["training"], r["control_mode"], r["shots"]): r for r in averages
    }
    main_rows = []
    for row in averages:
        if row["training"] not in MODES or row["control_mode"] != "measured":
            continue
        source = "independent" if row["training"] == "sequential" else row["training"]
        concept = lookup[row["seed"], source, "measured", row["shots"]]
        main_rows.append(
            {
                **row,
                **{k: concept[k] for k in CONCEPT_METRICS},
                "concept_source_training": source,
            }
        )
    for name, rows in (
        ("condition_results", averages),
        ("sampling_repeats", repeated),
        ("paired_results", paired),
        ("concept_results", concepts),
        ("four_modes_main", main_rows),
        ("standard_concept_diagnostics", diagnostics),
    ):
        write_csv(root / f"{name}.csv", rows)
    atomic_json(root / "summary.json", summary)
    lines = [
        f"# Robot {shared.stage}: four modes, finite shots and concept correction",
        "",
        f"Development: {shared.config.development}. "
        f"Test evaluated: {shared.stage == 'test'}.",
        "",
        "All values: mean ± sample SD across training seeds; "
        "finite-shot repetitions averaged within seed.",
        "shots=0 denotes exact probabilities, not a physical zero-shot experiment.",
        "",
        "| Training | Control | Exact label accuracy (%) |",
        "|---|---|---:|",
    ]

    def formatted(value, scale=1):
        sd = "n/a" if value["std"] is None else f"{scale * value['std']:.3f}"
        return f"{scale * value['mean']:.3f} ± {sd}"

    for row in summary["conditions"]:
        if row["shots"] == 0:
            lines.append(
                f"| {row['training']} | {row['control_mode']} | "
                f"{formatted(row['label_accuracy'], 100)} |"
            )
    lines += ["", "| Comparison | Shots | Accuracy change (pp) |", "|---|---:|---:|"]
    for row in summary["effects"]:
        if row["comparison"] in {
            f"{m}_{e}" for m in SUPERVISED for e in ("feedback", "correction")
        }:
            lines.append(
                f"| {row['comparison']} | {row['shots'] or 'exact'} | "
                f"{formatted(row['accuracy_gain_pp'])} |"
            )
    lines += [
        "",
        "Standard has no concept supervision or semantic concept correction. "
        "Joint uses its jointly trained no-feedback control; Independent/Sequential "
        "use their shared frozen-frontend control.",
        "Both sides retain measurement. Interventions replace only X controls, "
        "preserving physical branches and states.",
        "Concept tables report the Independent/Sequential frontend once; "
        "Standard diagnostics are separate. "
        "All positive and negative gains are retained.",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    plots(root, summary, shared.config.shot_list())
    atomic_json(
        root / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {
                str(p.relative_to(root)): sha256(p)
                for p in sorted(root.rglob("*"))
                if p.is_file()
                and p.name != "result_lock.json"
                and not p.name.endswith(".tmp")
            },
        },
    )


def finish_pipeline(shared) -> None:
    if shared.verify_report_lock(shared.output) is not None:
        return
    value = {
        "status": "complete",
        "test_read": not shared.config.development,
        "test_evaluated": not shared.config.development,
        "development": shared.config.development,
        "stages": ["validation", shared.final_role],
        "training": "none; all models frozen",
        "completed_conditions": 2 * len(CONDITIONS) * len(shared.config.seed_list()),
        "formal_report": f"{shared.final_role}/summary.md",
    }
    atomic_json(shared.output / "summary.json", value)
    (shared.output / "summary.md").write_text(
        "# Robot shots and final evaluation complete\n\n"
        f"Development: {shared.config.development}; "
        f"real test evaluated: {not shared.config.development}.\n\n"
        "- [Validation](validation/summary.md)\n"
        f"- [Final results]({shared.final_role}/summary.md)\n\n"
        "Models, seeds, conditions, budgets and rules were fixed before test access. "
        "Sampling repeats are not additional training seeds.\n",
        encoding="utf-8",
    )
    names = (
        "manifest.json",
        "config.json",
        "reference_lock.json",
        "summary.json",
        "summary.md",
        "protocol_lock.json",
        "test_access.json",
        "validation/result_lock.json",
        f"{shared.final_role}/result_lock.json",
    )
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {n: sha256(shared.output / n) for n in names},
        },
    )
