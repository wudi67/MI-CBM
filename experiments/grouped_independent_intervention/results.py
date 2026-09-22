"""Absolute performance, paired intervention gains and their between-mode gaps."""

from __future__ import annotations

import csv

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256  # noqa: E402
from experiments.grouped_feedback_ablation.protocol import load_checkpoint  # noqa: E402
from experiments.grouped_sequential_intervention.protocol import (  # noqa: E402
    LABELS,
    MODES,
)
from experiments.grouped_sequential_intervention.results import (
    paired_row,
    statistics_for,
)  # noqa: E402

from .artifacts import verify_complete  # noqa: E402
from .evaluation import directory, verify_condition  # noqa: E402
from .protocol import Experiment  # noqa: E402


def compare_modes(a: dict, b: dict) -> dict:
    """a=Independent, b=Sequential; never pool their seeds as independent repeats."""
    return {
        "seed": a["seed"],
        "mode": a["mode"],
        "independent_accuracy": a["accuracy"],
        "sequential_accuracy": b["accuracy"],
        "accuracy_difference_pp": 100 * (a["accuracy"] - b["accuracy"]),
        "independent_intervention_gain_pp": a["delta_accuracy_pp"],
        "sequential_intervention_gain_pp": b["delta_accuracy_pp"],
        "intervention_gain_difference_pp": a["delta_accuracy_pp"]
        - b["delta_accuracy_pp"],
        "shot_accuracy_difference_pp": 100 * (a["shot_accuracy"] - b["shot_accuracy"]),
        "shot_intervention_gain_difference_pp": a["shot_delta_accuracy_pp"]
        - b["shot_delta_accuracy_pp"],
        "bce_difference": a["bce"] - b["bce"],
    }


def write_csv(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(shared: Experiment, stats: dict) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for row, training in enumerate(("sequential", "independent")):
        for column, metric in enumerate(("accuracy", "shot_accuracy")):
            axis = axes[row, column]
            for path, label, marker in (
                (("measured", "shape", "both"), "Shape first", "o"),
                (("measured", "scale", "both"), "Scale first", "s"),
            ):
                points = [stats[training][m][metric] for m in path]
                axis.errorbar(
                    [0, 1, 2],
                    [100 * p["mean"] for p in points],
                    yerr=[100 * (p["sample_std"] or 0) for p in points],
                    marker=marker,
                    capsize=3,
                    label=label,
                )
            kind = (
                "Exact probabilities"
                if column == 0
                else f"{shared.config.shots} joint shots"
            )
            axis.set_title(f"{training.title()} | {kind}")
            axis.set_xticks([0, 1, 2])
            axis.set_xlabel("Number of concepts corrected")
            axis.set_ylabel("Label accuracy (%)")
            axis.grid(alpha=0.25)
            axis.legend()
    role = (
        "ENGINEERING CHECK" if shared.manifest["engineering_subset"] else "Validation"
    )
    count = stats["independent"]["measured"]["n_seeds"]
    fig.suptitle(f"{role} | {count} paired training seeds | mean +/- sample SD")
    fig.tight_layout()
    names = []
    for extension in ("png", "pdf", "svg"):
        name = f"mode_interventions.{extension}"
        fig.savefig(shared.output / name, dpi=180)
        names.append(name)
    plt.close(fig)
    return names


def summarize(shared: Experiment) -> dict:
    rows, pairs, completed, concepts, locks = [], [], [], {}, {}
    for seed in shared.config.seed_list():
        result_path = shared.output / f"independent/seed{seed}/result.json"
        if result_path.exists():
            verify_complete(shared, seed)
            locks[str(result_path.relative_to(shared.output))] = sha256(result_path)
        availability = {
            (training, mode): verify_condition(shared, training, seed, mode)
            for training in ("sequential", "independent")
            for mode in MODES
        }
        if not all(value is not None for value in availability.values()):
            continue
        completed.append(seed)
        by_mode = {}
        original_concepts = None
        for training in ("sequential", "independent"):
            original = load_checkpoint(
                directory(shared, training, seed, "measured") / "predictions.pt"
            )
            if original_concepts is not None:
                for key in (
                    "source_index",
                    "labels",
                    "concepts",
                    "concept_probabilities",
                ):
                    if not torch.equal(original[key], original_concepts[key]):
                        raise ValueError(
                            "Training modes differ in concepts "
                            "or paired evaluation rows"
                        )
            original_concepts = original
            baseline = availability[(training, "measured")]
            assert baseline is not None
            concepts[str(seed)] = baseline["concept"]
            for mode in MODES:
                path = directory(shared, training, seed, mode)
                raw = load_checkpoint(path / "predictions.pt")
                metrics = availability[(training, mode)]
                assert metrics is not None
                row = paired_row(seed, mode, metrics, raw, original)
                by_mode[(training, mode)] = row
                rows.append({"training": training, **row})
                lock_path = path / "evaluation_lock.json"
                locks[str(lock_path.relative_to(shared.output))] = sha256(lock_path)
        pairs += [
            compare_modes(by_mode[("independent", m)], by_mode[("sequential", m)])
            for m in MODES
        ]
    stats = {
        training: statistics_for(
            [
                {k: v for k, v in r.items() if k != "training"}
                for r in rows
                if r["training"] == training
            ],
            completed,
        )
        for training in ("sequential", "independent")
    }
    paired_stats = statistics_for(pairs, completed)
    result = {
        "status": "complete" if completed == shared.config.seed_list() else "partial",
        "manifest_sha256": shared.manifest_hash,
        "test_evaluated": False,
        "engineering_subset": shared.manifest["engineering_subset"],
        "paired_training_budget": shared.paired_training,
        "train_samples": len(shared.data["train"]["labels"]),
        "n_samples": len(shared.data["val"]["labels"]),
        "planned_seeds": shared.config.seed_list(),
        "complete_seeds": completed,
        "missing_seeds": [s for s in shared.config.seed_list() if s not in completed],
        "rows": rows,
        "paired_results": pairs,
        "statistics": stats,
        "paired_statistics": paired_stats,
        "original_concept_metrics": concepts,
        "artifacts": locks,
        "semantics": (
            "Independent uses true classical controls during label training and "
            "retained input-dependent B; measured is unassisted evaluation"
        ),
        "repeat_unit": (
            "paired training seed; only seeds with both modes and all four "
            "conditions enter aggregates"
        ),
        "finite_shot_scope": (
            "one fixed joint sampling realization per seed/condition; "
            "not noisy training or complete shot uncertainty"
        ),
    }
    atomic_json(shared.output / "summary.json", result)
    artifacts = ["summary.json", "summary.md"]
    if rows:
        write_csv(shared.output / "condition_results.csv", rows)
        write_csv(shared.output / "paired_results.csv", pairs)
        artifacts += ["condition_results.csv", "paired_results.csv"]
    lines = [
        "# Independent 与 Sequential：训练方式和概念纠正",
        "",
        f"状态：{result['status']}；完成配对种子：{completed}；验证图片：{result['n_samples']}。",
        "工程检查，不作为论文对照结果。"
        if result["engineering_subset"]
        else "同一冻结前段、初始化、训练预算及图片顺序；固定最终轮；验证集评价。",
        "",
        "Independent 训练时用真实概念控制 X 门，保留后五位条件态。",
        "不纠正评价仍使用实际测量概念；全部纠正的结果不是无人帮助时的准确率。",
        "",
        "| 训练模式 | 条件 | Label准确率（%） | 干预增益（百分点） | "
        "256-shots准确率（%） |",
        "|---|---|---:|---:|---:|",
    ]
    lines[-2] = lines[-2].replace("256-shots", f"{shared.config.shots}-shots")
    if completed:
        for training in ("sequential", "independent"):
            for mode in MODES:
                values = []
                for key, factor in (
                    ("accuracy", 100),
                    ("delta_accuracy_pp", 1),
                    ("shot_accuracy", 100),
                ):
                    item = stats[training][mode][key]
                    std = (
                        "未估计"
                        if item["sample_std"] is None
                        else f"{factor * item['sample_std']:.2f}"
                    )
                    values.append(f"{factor * item['mean']:.2f} ± {std}")
                lines.append(
                    f"| {training} | {LABELS[mode]} | " + " | ".join(values) + " |"
                )
        lines += [
            "",
            "| 条件 | Independent−Sequential 准确率差（百分点） | "
            "干预增益之差（百分点） |",
            "|---|---:|---:|",
        ]
        for mode in MODES:
            values = []
            for key in ("accuracy_difference_pp", "intervention_gain_difference_pp"):
                item = paired_stats[mode][key]
                std = (
                    "未估计"
                    if item["sample_std"] is None
                    else f"{item['sample_std']:.2f}"
                )
                values.append(f"{item['mean']:+.2f} ± {std}")
            lines.append(f"| {LABELS[mode]} | " + " | ".join(values) + " |")
        artifacts.extend(plot(shared, stats))
        lines += ["", "![训练模式与纠正路径](mode_interventions.png)"]
    lines += [
        "",
        "增益之差 = Independent(纠正后−纠正前) − Sequential(纠正后−纠正前)。",
        "同时比较绝对准确率，避免把较低起点造成的较大增幅当作更好的最终性能。",
        "均值±样本标准差；统计单位为训练种子；保留所有正负结果。",
        "逐种子各条件、BCE、balanced accuracy、错→对/对→错及有限shots见CSV/JSON。",
        "概念指标来自原始测量分布，两模式共用同一冻结前段；不计入真实概念替换带来的表面提升。",
        "全部32个实际测量分支及Born权重保留，仅替换经典X控制记录；没有重编码。尚未评价测试集。",
    ]
    (shared.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {n: sha256(shared.output / n) for n in artifacts},
        },
    )
    return result
