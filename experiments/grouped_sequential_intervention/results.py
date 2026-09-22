"""Paired label changes and two semantic-concept correction paths."""

from __future__ import annotations

import csv
import statistics

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256  # noqa: E402
from experiments.grouped_feedback_ablation.protocol import load_checkpoint  # noqa: E402

from .evaluation import paired_changes  # noqa: E402
from .protocol import LABELS, MODES  # noqa: E402
from .runner import Experiment  # noqa: E402


def statistics_for(rows: list[dict], complete_seeds: list[int]) -> dict:
    result: dict = {}
    for mode in MODES:
        selected = [
            r for r in rows if r["mode"] == mode and r["seed"] in complete_seeds
        ]
        result[mode] = {
            "n_seeds": len(selected),
            "seeds": [r["seed"] for r in selected],
        }
        if not selected:
            continue
        for key in selected[0]:
            if key in {"seed", "mode", "n_samples"}:
                continue
            values = [r[key] for r in selected]
            result[mode][key] = {
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            }
    return result


def paired_row(seed: int, mode: str, metrics: dict, raw: dict, original: dict) -> dict:
    for key in ("source_index", "labels", "concepts", "concept_probabilities"):
        if not torch.equal(original[key], raw[key]):
            raise ValueError(
                f"Correction changed original concepts or validation pairing: {key}"
            )
    row = {"seed": seed, "mode": mode, "n_samples": len(raw["labels"])}
    for prefix, section, raw_key in (
        ("", metrics["label"], "label_probabilities"),
        ("shot_", metrics["finite_shots"]["label"], "shot_label_probabilities"),
    ):
        row.update({prefix + key: value for key, value in section.items()})
        changes = paired_changes(original[raw_key], raw[raw_key], raw["labels"])
        row.update({prefix + key: value for key, value in changes.items()})
    return row


def plot_paths(shared: Experiment, stats: dict) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, metric, title in zip(
        axes,
        ("accuracy", "shot_accuracy"),
        ("Exact branch probabilities", f"{shared.config.shots} joint shots / image"),
        strict=True,
    ):
        for path, label, marker, offset in (
            (("measured", "shape", "both"), "Shape first", "o", -0.025),
            (("measured", "scale", "both"), "Scale first", "s", 0.025),
        ):
            means = [100 * stats[mode][metric]["mean"] for mode in path]
            stds = [100 * (stats[mode][metric]["sample_std"] or 0) for mode in path]
            axis.errorbar(
                [x + offset for x in (0, 1, 2)],
                means,
                yerr=stds,
                marker=marker,
                capsize=3,
                label=label,
            )
        axis.set_xticks([0, 1, 2])
        axis.set_xlabel("Number of concepts corrected")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Label accuracy (%)")
    count = stats["measured"]["n_seeds"]
    role = (
        "ENGINEERING SUBSET" if shared.manifest["engineering_subset"] else "Validation"
    )
    fig.suptitle(
        f"Sequential, frozen models | {role} | "
        f"{count} training seeds, mean +/- sample SD"
    )
    fig.tight_layout()
    artifacts = []
    for extension in ("png", "pdf", "svg"):
        name = f"intervention_paths.{extension}"
        fig.savefig(shared.output / name, dpi=180)
        artifacts.append(name)
    plt.close(fig)
    return artifacts


def summarize(shared: Experiment) -> dict:
    rows, concepts, locks, complete_seeds = [], {}, {}, []
    for seed in shared.config.seed_list():
        available = {mode: shared.verify_condition(seed, mode) for mode in MODES}
        if all(value is not None for value in available.values()):
            complete_seeds.append(seed)
        if available["measured"] is None:
            continue
        original = load_checkpoint(
            shared.directory(seed, "measured") / "predictions.pt"
        )
        concepts[str(seed)] = available["measured"]["concept"]
        for mode, metrics in available.items():
            if metrics is None:
                continue
            directory = shared.directory(seed, mode)
            raw = load_checkpoint(directory / "predictions.pt")
            rows.append(paired_row(seed, mode, metrics, raw, original))
            path = directory / "evaluation_lock.json"
            locks[str(path.relative_to(shared.output))] = sha256(path)
    stats = statistics_for(rows, complete_seeds)
    result = {
        "status": "complete"
        if len(complete_seeds) == len(shared.config.seed_list())
        else "partial",
        "manifest_sha256": shared.manifest_hash,
        "test_evaluated": False,
        "engineering_subset": shared.manifest["engineering_subset"],
        "n_samples": len(shared.data["labels"]),
        "planned_seeds": shared.config.seed_list(),
        "complete_seeds": complete_seeds,
        "missing_seeds": [
            s for s in shared.config.seed_list() if s not in complete_seeds
        ],
        "rows": rows,
        "statistics": stats,
        "original_concept_metrics": concepts,
        "artifacts": locks,
        "interpretation": (
            "Record correction in frozen Sequential models with retained B states; "
            "no retraining or full-state semantic replacement. "
            "All positive and negative effects reported."
        ),
        "repeat_unit": (
            "training seed; aggregates use only seeds "
            "with all four conditions completed"
        ),
        "finite_shot_scope": (
            "one fixed Monte Carlo realization per seed/condition; changes include "
            "shot noise, not paired individual quantum outcomes"
        ),
    }
    atomic_json(shared.output / "summary.json", result)
    artifacts = ["summary.json", "summary.md"]
    if rows:
        with (shared.output / "paired_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        artifacts.append("paired_results.csv")
    lines = [
        "# Sequential 概念记录纠正：验证集评价",
        "",
        f"状态：{result['status']}；全部四条件完成的种子：{complete_seeds}；"
        f"每条件 {result['n_samples']} 张图片。",
        "工程子集，不作为论文结果。"
        if result["engineering_subset"]
        else "固定最终训练端点和验证划分；未评价测试集。",
        "",
        "只替换测量后控制 X 门的经典记录；"
        "保留实际测量分支、Born 权重及反馈前的 B 状态。",
        "按 Shape/Scale 两个概念纠正；不重新训练，不修改前段概念预测，"
        "不进行重编码或 MAP 控制。",
        "",
        "| seed | 条件 | Label准确率 | 相对不纠正（百分点） | 错→对 | 对→错 | "
        "shots准确率 | shots差值 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {LABELS[row['mode']]} | {row['accuracy']:.2%} | "
            f"{row['delta_accuracy_pp']:+.2f} | {row['wrong_to_right_count']} | "
            f"{row['right_to_wrong_count']} | {row['shot_accuracy']:.2%} | "
            f"{row['shot_delta_accuracy_pp']:+.2f} |"
        )
    lines += [
        "",
        "仅对四条件均完成的相同种子汇总；均值±样本标准差，单种子不估计标准差。",
        "",
        "| 条件 | Label准确率（%） | 配对差值（百分点） | shots准确率（%） | "
        "shots配对差值（百分点） |",
        "|---|---:|---:|---:|---:|",
    ]
    if complete_seeds:
        for mode in MODES:
            values = []
            for key, factor in (
                ("accuracy", 100),
                ("delta_accuracy_pp", 1),
                ("shot_accuracy", 100),
                ("shot_delta_accuracy_pp", 1),
            ):
                item = stats[mode][key]
                std = (
                    "未估计"
                    if item["sample_std"] is None
                    else f"{factor * item['sample_std']:.2f}"
                )
                values.append(f"{factor * item['mean']:.2f} ± {std}")
            lines.append(f"| {LABELS[mode]} | " + " | ".join(values) + " |")
        artifacts.extend(plot_paths(shared, stats))
        lines += ["", "![概念纠正路径](intervention_paths.png)"]
    lines += [
        "",
        "BCE、balanced accuracy、逐图片预测及全部计数见 CSV/JSON/PT。"
        "概念指标来自原始测量分布，真实概念替换不计为预测提升。",
        "有限 shots 为每种条件各自联合 (m,y) 分布的一次固定采样评价；"
        "不是带采样噪声训练，种子标准差也不是完整测量误差估计。",
        "纠正导致的下降也保留。该实验回答此模型的经典概念记录能否有效纠正 Label，"
        "不单独证明隐藏量子信息的语义被纠正。",
    ]
    (shared.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {name: sha256(shared.output / name) for name in artifacts},
        },
    )
    return result
