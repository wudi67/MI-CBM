"""Uniform subset averages within seed; uncertainty across training seeds."""

from math import comb
from statistics import mean, stdev

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_robot_independent.results import write_csv
from experiments.grouped_robot_shots_final.results import stats

from .protocol import MASKS, MODES

METRICS = ("accuracy", "balanced_accuracy", "bce")


def curve_rows(
    evaluations: list[dict], seeds: list[int]
) -> tuple[list[dict], list[dict]]:
    expected = {(s, t, m) for s in seeds for t in MODES for m in MASKS}
    actual = {(v["seed"], v["training"], v["mask"]) for v in evaluations}
    if actual != expected or len(evaluations) != len(expected):
        raise ValueError("All 32 subsets of every mode/seed are required exactly once")
    rows: list[dict] = []
    for seed in seeds:
        for mode in MODES:
            values = [
                v for v in evaluations if (v["seed"], v["training"]) == (seed, mode)
            ]
            baseline = next(v["metrics"]["label"] for v in values if v["mask"] == 0)
            previous = None
            for count in range(6):
                selected = [
                    v["metrics"]["label"]
                    for v in values
                    if v["mask"].bit_count() == count
                ]
                if len(selected) != comb(5, count):
                    raise ValueError("Wrong subset multiplicity")
                averaged = {k: mean(v[k] for v in selected) for k in METRICS}
                # Classify and score EACH subset first. Averaging probabilities
                # before thresholding would evaluate a different ensemble model.
                rows.append(
                    {
                        "role": values[0]["role"],
                        "seed": seed,
                        "training": mode,
                        "count": count,
                        "n_subsets": len(selected),
                        **averaged,
                        "gain_pp": 100 * (averaged["accuracy"] - baseline["accuracy"]),
                        "step_gain_pp": 0.0
                        if previous is None
                        else 100 * (averaged["accuracy"] - previous),
                        "subset_accuracy_std": stdev([v["accuracy"] for v in selected])
                        if len(selected) > 1
                        else 0.0,
                    }
                )
                previous = averaged["accuracy"]
    summary = []
    for mode in MODES:
        for count in range(6):
            selected = [
                r for r in rows if r["training"] == mode and r["count"] == count
            ]
            summary.append(
                {
                    "training": mode,
                    "count": count,
                    "n_subsets_per_seed": comb(5, count),
                    "n_seeds": len(selected),
                    **{
                        k: stats([r[k] for r in selected])
                        for k in (*METRICS, "gain_pp", "step_gain_pp")
                    },
                    "positive_gain_seeds": sum(r["gain_pp"] > 0 for r in selected),
                }
            )
    return rows, summary


def plots(root, summary: list[dict]) -> None:
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for mode in MODES:
        rows = [v for v in summary if v["training"] == mode]
        for ax, metric, scale in ((axes[0], "accuracy", 100), (axes[1], "gain_pp", 1)):
            y = [scale * v[metric]["mean"] for v in rows]
            sd = [scale * (v[metric]["std"] or 0) for v in rows]
            (line,) = ax.plot(range(6), y, "o-", label=mode.title())
            ax.fill_between(
                range(6),
                [a - b for a, b in zip(y, sd, strict=True)],
                [a + b for a, b in zip(y, sd, strict=True)],
                alpha=0.15,
                color=line.get_color(),
            )
            ax.set_xticks(range(6))
            ax.set_xlabel("Number of concept positions supplied with ground truth")
            ax.grid(alpha=0.2)
            ax.legend()
    axes[0].set_ylabel("Label accuracy (%)")
    axes[1].set_ylabel("Change from zero correction (pp)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    fig.suptitle("Uniform average over all subsets; shading: training-seed SD")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(root / f"intervention_curve.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(shared, evaluations: list[dict]) -> None:
    rows, curves = curve_rows(evaluations, shared.config.seed_list())
    root = shared.output
    mask_rows = [
        {
            **{
                k: v[k]
                for k in (
                    "role",
                    "seed",
                    "training",
                    "mask",
                    "count",
                    "reused_endpoint",
                )
            },
            "selected_concepts": ",".join(v["selected_concepts"]),
            **v["metrics"]["label"],
        }
        for v in evaluations
    ]
    write_csv(root / "subset_results.csv", mask_rows)
    write_csv(root / "seed_curves.csv", rows)
    write_csv(
        root / "curve_summary.csv",
        [
            {
                **{
                    k: v[k]
                    for k in (
                        "training",
                        "count",
                        "n_subsets_per_seed",
                        "n_seeds",
                        "positive_gain_seeds",
                    )
                },
                **{
                    f"{k}_{stat}": v[k][stat]
                    for k in (*METRICS, "gain_pp", "step_gain_pp")
                    for stat in ("mean", "std")
                },
            }
            for v in curves
        ],
    )
    summary = {
        "status": "complete",
        "role": shared.config.role,
        "development": shared.config.development,
        "test_read": not shared.config.development,
        "test_evaluated": not shared.config.development,
        "n_samples": len(shared.data["labels"]),
        "seeds": shared.config.seed_list(),
        "completed_conditions": len(evaluations),
        "reused_endpoints": 2 * len(MODES) * len(shared.config.seed_list()),
        "new_partial_conditions": 30 * len(MODES) * len(shared.config.seed_list()),
        "training": "none",
        "curves": curves,
        "aggregation": shared.manifest["aggregation"],
        "historical_final_report": str(shared.sources.stage / "summary.md"),
    }
    atomic_json(root / "summary.json", summary)
    lines = [
        "# Robot: correction count curves",
        "",
        f"Role: {shared.config.role}; development: {shared.config.development}; "
        f"samples: {len(shared.data['labels'])}.",
        "",
        "Each point scores every subset first, averages equally within seed, "
        "then reports training-seed mean ± sample SD.",
        "Count is queried concept positions, including positions whose "
        "original value was already correct.",
        "Exact probabilities; frozen models; no finite-shot simulation "
        "or retraining in this experiment.",
        "",
        "| Corrected positions | Independent accuracy (%) | "
        "Sequential accuracy (%) | Joint accuracy (%) |",
        "|---:|---:|---:|---:|",
    ]
    for count in range(6):
        cells = []
        for mode in MODES:
            value = next(
                v["accuracy"]
                for v in curves
                if v["count"] == count and v["training"] == mode
            )
            sd = "n/a" if value["std"] is None else f"{100 * value['std']:.3f}"
            cells.append(f"{100 * value['mean']:.3f} ± {sd}")
        lines.append(f"| {count} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "All 32 subsets are retained; no monotonic smoothing, "
        "chosen order, or best-subset selection.",
        "Zero/all correction endpoints reproduce the prior locked evaluation. "
        "Subsets are not additional independent training seeds.",
        "Original Born weights and pre-feedback states remain; "
        "selected classical control bits are replaced before X gates.",
        "",
        "Prior four-mode, feedback and shots report: "
        f"{shared.sources.stage / 'summary.md'}",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    plots(root, curves)
    atomic_json(
        root / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {
                str(p.relative_to(root)): sha256(p)
                for p in sorted(root.rglob("*"))
                if p.is_file()
                and p.name
                not in {
                    "result_lock.json",
                    "heartbeat.json",
                    "evaluation.log",
                    ".worker.lock",
                }
                and not p.name.endswith(".tmp")
            },
        },
    )
