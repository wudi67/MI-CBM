"""Draw Fig. 2 (intervention curves) and Fig. 3 (accuracy by mispredicted concepts).

Usage: python -m analysis.figures [--outputs outputs] [--save-dir figures]
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from analysis.common import (  # noqa: E402
    CORRECTED,
    DSPRITES_TEST,
    ROBOT_CURVE,
    ROBOT_TEST,
    SEEDS,
    load_joint,
    predicted_concepts,
    read_rows,
)

STYLE = {
    "independent": dict(color="#2a78d6", marker="o", label="Independent", lw=1.6, zorder=4),
    "sequential": dict(color="#eb6834", marker="s", label="Sequential", lw=1.2, zorder=3),
    "joint": dict(color="#1baf7a", marker="^", label="Joint", lw=1.2, zorder=2),
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    })


def standard_accuracy(path: Path) -> float:
    values = [100 * float(r["label_accuracy"]) for r in read_rows(path)
              if r["shots"] == "0" and r["training"] == "standard"
              and r["control_mode"] == "measured"]
    return statistics.mean(values)


def intervention_curves(root: Path) -> dict:
    per_count = defaultdict(list)
    for row in read_rows(root / DSPRITES_TEST / "correction_count_results.csv"):
        if row["shots"] == "0":
            per_count[(row["training"], int(row["corrected_concepts"]))].append(
                100 * float(row["label_accuracy"]))
    dsprites = {m: [(statistics.mean(per_count[(m, k)]), statistics.stdev(per_count[(m, k)]))
                    for k in range(3)] for m in STYLE}
    robot = defaultdict(dict)
    for row in read_rows(root / ROBOT_CURVE / "curve_summary.csv"):
        robot[row["training"]][int(row["count"])] = (
            100 * float(row["accuracy_mean"]), 100 * float(row["accuracy_std"]))
    robot = {m: [robot[m][k] for k in range(6)] for m in STYLE}
    return {
        "dSprites": (dsprites, standard_accuracy(root / DSPRITES_TEST / "condition_results.csv")),
        "Robot": (robot, standard_accuracy(root / ROBOT_TEST / "condition_results.csv")),
    }


def figure2(root: Path, save_dir: Path) -> None:
    curves = intervention_curves(root)
    fig, axes = plt.subplots(1, 2, figsize=(3.39, 1.78), sharey=True,
                             gridspec_kw=dict(width_ratios=[2, 5]))
    for ax, (name, label) in zip(axes, (("dSprites", "(a) dSprites"), ("Robot", "(b) Robot"))):
        data, standard = curves[name]
        counts = list(range(len(data["independent"])))
        ax.axhline(standard, color="#6b6b6b", ls="--", lw=0.9, zorder=1, label="Standard")
        for mode in ("joint", "sequential", "independent"):
            style = STYLE[mode]
            mean = [v[0] for v in data[mode]]
            std = [v[1] for v in data[mode]]
            ax.fill_between(counts, [a - b for a, b in zip(mean, std)],
                            [a + b for a, b in zip(mean, std)], color=style["color"],
                            alpha=0.14, lw=0, zorder=style["zorder"] - 1)
            ax.plot(counts, mean, color=style["color"], marker=style["marker"], ms=3.6,
                    lw=style["lw"], mec="white", mew=0.5, label=style["label"],
                    zorder=style["zorder"])
        ax.set_xticks(counts)
        pad = 0.25 if len(counts) == 3 else 0.3
        ax.set_xlim(-pad, counts[-1] + pad)
        ax.grid(axis="y", color="#e3e3e3", lw=0.5, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=2.5, pad=2)
        ax.text(0.5, -0.42, label, transform=ax.transAxes, ha="center", va="top")
    axes[0].set_ylim(55, 100)
    axes[0].set_yticks([60, 70, 80, 90, 100])
    axes[0].set_ylabel("Label accuracy (%)", labelpad=2)
    fig.supxlabel("Number of corrected concepts", fontsize=8, y=0.13)
    handles, labels = axes[1].get_legend_handles_labels()
    order = [labels.index(x) for x in ("Independent", "Sequential", "Joint", "Standard")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
               ncol=4, frameon=False, bbox_to_anchor=(0.54, 1.01), handlelength=1.8,
               columnspacing=1.0, handletextpad=0.4)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.30, wspace=0.08)
    fig.savefig(save_dir / "fig_intervention.pdf")
    plt.close(fig)


def accuracy_by_mispredicted(root: Path, name: str) -> dict:
    """Label accuracy before/after full correction, grouped by mispredicted concepts."""
    groups = [0, 1, 2] if name == "dSprites" else [0, 1, 2, 3]
    stats = {g: {"share": [], "before": [], "after": []} for g in groups}
    for seed in SEEDS:
        measured = load_joint(root, name, seed, "measured")
        corrected = load_joint(root, name, seed, CORRECTED[name])
        labels = measured["labels"].long()
        before = (measured["branch_label_mass"].sum(1) >= 0.5).long()
        after = (corrected["branch_label_mass"].sum(1) >= 0.5).long()
        predicted = predicted_concepts(name, measured["concept_probabilities"])
        wrong = (predicted != measured["concepts"]).sum(1).clamp(max=groups[-1])
        for g in groups:
            mask = wrong == g
            stats[g]["share"].append(100 * float(mask.float().mean()))
            stats[g]["before"].append(100 * float((before[mask] == labels[mask]).float().mean()))
            stats[g]["after"].append(100 * float((after[mask] == labels[mask]).float().mean()))
    return stats


def figure3(root: Path, save_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.39, 1.98), sharey=True,
                             gridspec_kw=dict(width_ratios=[3, 4]))
    panels = (("dSprites", "(a) dSprites (2 concepts)"), ("Robot", "(b) Robot (5 concepts)"))
    for ax, (name, label) in zip(axes, panels):
        stats = accuracy_by_mispredicted(root, name)
        groups = sorted(stats)
        xs = list(range(len(groups)))
        for key, color, marker, ls, lw, text in (
                ("before", "#8c8c8c", "s", "--", 1.2, "No correction"),
                ("after", "#2a78d6", "o", "-", 1.6, "All concepts corrected")):
            mean = [statistics.mean(stats[g][key]) for g in groups]
            std = [statistics.stdev(stats[g][key]) for g in groups]
            ax.fill_between(xs, [a - b for a, b in zip(mean, std)],
                            [a + b for a, b in zip(mean, std)], color=color, alpha=0.13, lw=0)
            ax.plot(xs, mean, color=color, marker=marker, ms=3.7, lw=lw, ls=ls, mec="white",
                    mew=0.5, label=text, zorder=3)
        for x, g in zip(xs, groups):
            b = statistics.mean(stats[g]["before"])
            a = statistics.mean(stats[g]["after"])
            ax.add_patch(FancyArrowPatch((x, b + 1.2), (x, a - 1.6), arrowstyle="-|>",
                                         mutation_scale=6, lw=0.8, color="#9ec5f4", zorder=2))
            share = statistics.mean(stats[g]["share"])
            ax.annotate(f"{share:.0f}%", (x, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -15), textcoords="offset points", ha="center", va="top",
                        fontsize=6.3, color="#7a7a7a")
        ticks = [str(g) for g in groups]
        if name == "Robot":
            ticks[-1] = r"$\geq$3"
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.35, len(groups) - 0.65)
        ax.set_ylim(40, 100)
        ax.set_yticks([40, 60, 80, 100])
        ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=2.5, pad=2)
        ax.text(0.5, -0.50, label, transform=ax.transAxes, ha="center", va="top")
    axes[0].set_ylabel("Label accuracy (%)", labelpad=2)
    fig.supxlabel("Number of mispredicted concepts (share of test images)", fontsize=8, y=0.13)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.54, 1.01), handlelength=1.8, columnspacing=1.2)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.875, bottom=0.35, wspace=0.08)
    fig.savefig(save_dir / "fig_bywrong.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--save-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    figure2(args.outputs, args.save_dir)
    figure3(args.outputs, args.save_dir)
    print(f"Saved figures to {args.save_dir}/")


if __name__ == "__main__":
    main()
