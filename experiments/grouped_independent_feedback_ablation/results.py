"""Paired seed statistics and publication figures for unassisted prediction."""

from __future__ import annotations

import csv
import statistics

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256  # noqa: E402
from experiments.grouped_feedback_ablation.protocol import load_checkpoint  # noqa: E402
from experiments.grouped_sequential_intervention.evaluation import (
    paired_changes,  # noqa: E402
)

from .protocol import VARIANTS  # noqa: E402
from .runner import Experiment, recompute_metrics  # noqa: E402


def statistics_for(rows: list[dict]) -> dict:
    if not rows:
        return {}
    result = {}
    for key, value in rows[0].items():
        if (
            key in {"seed", "n_samples"}
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            continue
        values = [row[key] for row in rows]
        result[key] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        }
    return result


def paired_row(seed: int, feedback: dict, baseline: dict) -> dict:
    for key in ("source_index", "labels", "concepts", "concept_probabilities"):
        if not torch.equal(feedback[key], baseline[key]):
            raise ValueError(f"Paired images/concept distribution differ: {key}")
    row = {"seed": seed, "n_samples": len(feedback["labels"])}
    a, b = recompute_metrics(feedback), recompute_metrics(baseline)
    for prefix, section, key in (
        ("", "label", "label_probabilities"),
        ("shot_", "finite_shots", "shot_label_probabilities"),
    ):
        am, bm = (
            (a[section], b[section])
            if not prefix
            else (a[section]["label"], b[section]["label"])
        )
        row.update({prefix + "feedback_" + k: v for k, v in am.items()})
        row.update({prefix + "no_feedback_" + k: v for k, v in bm.items()})
        row.update(
            {
                prefix + k: v
                for k, v in paired_changes(
                    baseline[key], feedback[key], feedback["labels"]
                ).items()
            }
        )
        row[prefix + "delta_balanced_accuracy_pp"] = 100 * (
            am["balanced_accuracy"] - bm["balanced_accuracy"]
        )
        row[prefix + "delta_bce"] = am["bce"] - bm["bce"]
    row["max_concept_probability_difference"] = 0.0
    return row


def write_csv(path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot(shared: Experiment, pairs: list[dict]) -> list[str]:
    if not pairs:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    stats = statistics_for(pairs)
    for column, prefix in enumerate(("", "shot_")):
        axis = axes[0, column]
        for pair in pairs:
            axis.plot(
                [0, 1],
                [
                    100 * pair[prefix + v + "_accuracy"]
                    for v in ("no_feedback", "feedback")
                ],
                "o-",
                alpha=0.7,
                label=f"seed {pair['seed']}",
            )
        points = [stats[prefix + v + "_accuracy"] for v in ("no_feedback", "feedback")]
        axis.errorbar(
            [0, 1],
            [100 * p["mean"] for p in points],
            yerr=[100 * (p["sample_std"] or 0) for p in points],
            fmt="ks",
            capsize=5,
            label="mean +/- SD",
        )
        axis.set_xticks([0, 1], ["No feedback", "Independent feedback"])
        axis.set_ylabel("Label accuracy (%)")
        axis.set_title(
            "Exact probabilities"
            if not prefix
            else f"{shared.config.shots} joint shots / image"
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        axis = axes[1, column]
        gaps = [p[prefix + "delta_accuracy_pp"] for p in pairs]
        axis.bar(
            [str(p["seed"]) for p in pairs],
            gaps,
            color=["#2878b5" if gap >= 0 else "#d9534f" for gap in gaps],
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Paired training seed")
        axis.set_ylabel("Feedback gain (percentage points)")
    role = (
        "ENGINEERING SUBSET" if shared.manifest["engineering_subset"] else "Validation"
    )
    fig.suptitle(f"Independent feedback ablation | {role} | unassisted prediction")
    fig.tight_layout()
    names = []
    for extension in ("png", "pdf", "svg"):
        name = f"feedback_comparison.{extension}"
        fig.savefig(shared.output / name, dpi=180)
        names.append(name)
    plt.close(fig)
    return names


def summarize(shared: Experiment) -> dict:
    rows, pairs, complete, locks, concepts = [], [], [], {}, {}
    for seed in shared.config.seed_list():
        raw = {}
        for variant in VARIANTS:
            metrics = shared.verify_condition(seed, variant)
            if metrics is None:
                continue
            directory = shared.directory(seed, variant)
            raw[variant] = load_checkpoint(directory / "predictions.pt")
            shared.validate_rows(raw[variant])
            calculated = recompute_metrics(raw[variant])
            if (
                calculated["label"] != metrics["label"]
                or calculated["finite_shots"]["label"]
                != metrics["finite_shots"]["label"]
                or calculated["concept"] != metrics["concept"]
            ):
                raise ValueError("Evaluation metrics differ from raw predictions")
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "n_samples": metrics["n_samples"],
                    **metrics["label"],
                    **{
                        "shot_" + k: v
                        for k, v in metrics["finite_shots"]["label"].items()
                    },
                    **{"concept_" + k: v for k, v in metrics["concept"].items()},
                }
            )
            for name in ("evaluation_lock.json", "evaluation.json", "predictions.pt"):
                locks[str((directory / name).relative_to(shared.output))] = sha256(
                    directory / name
                )
        if len(raw) == 2:
            complete.append(seed)
            pairs.append(paired_row(seed, raw["feedback"], raw["no_feedback"]))
            concepts[str(seed)] = recompute_metrics(raw["feedback"])["concept"]
    stats = statistics_for(pairs)
    result = {
        "status": "complete"
        if len(complete) == len(shared.config.seed_list())
        else "partial",
        "manifest_sha256": shared.manifest_hash,
        "test_evaluated": False,
        "engineering_subset": shared.manifest["engineering_subset"],
        "paired_training_budget": True,
        "n_samples": len(shared.data["labels"]),
        "planned_seeds": shared.config.seed_list(),
        "complete_seeds": complete,
        "missing_seeds": [s for s in shared.config.seed_list() if s not in complete],
        "rows": rows,
        "paired_results": pairs,
        "statistics": stats,
        "positive_seeds": {
            prefix + "accuracy": sum(p[prefix + "delta_accuracy_pp"] > 0 for p in pairs)
            for prefix in ("", "shot_")
        },
        "original_concept_metrics": concepts,
        "artifacts": locks,
        "repeat_unit": "complete paired training seeds; all signs retained",
        "interpretation": (
            "unassisted Independent versus separately trained zero; "
            "both retain measurement and B states"
        ),
        "finite_shot_scope": "one joint draw per model/seed; evaluation only",
    }
    atomic_json(shared.output / "summary.json", result)
    atomic_json(
        shared.output / "pairing_audit.json",
        {
            "training": shared.sources.pairing,
            "evaluated_pairs": complete,
            "data_indices": shared.sources.independent.manifest["data_indices"],
            "concept_distributions_identical": bool(pairs),
            "test_evaluated": False,
        },
    )
    artifacts = ["summary.json", "summary.md", "pairing_audit.json"]
    for filename, values in (
        ("condition_results.csv", rows),
        ("paired_results.csv", pairs),
    ):
        if values:
            write_csv(shared.output / filename, values)
            artifacts.append(filename)
    artifacts += plot(shared, pairs)
    lines = [
        "# Independent 测量反馈消融",
        "",
        f"状态：{result['status']}；完成配对种子：{complete}；"
        f"每模型 {result['n_samples']} 张验证图片。",
        "工程子集，仅用于程序验证。"
        if result["engineering_subset"]
        else "固定最终训练端点；尚未评价测试集。",
        "",
        "有反馈：训练时真实概念控制 X，正常评价时实际测量记录控制 X。",
        "无反馈：从相同初始化独立训练，训练和评价都不执行条件 X。",
        "两边保留中间测量、实际 Born 分支和后五位量子态；"
        "前段、参数量、初始化、数据及训练预算配对。",
        "本实验没有概念纠正，报告正常分类性能；衡量条件反馈模块的贡献。",
        "",
        "| seed | 有反馈准确率 | 无反馈准确率 | 差值（百分点） | shots差值（百分点） |",
        "|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair['seed']} | {pair['feedback_accuracy']:.2%} | "
            f"{pair['no_feedback_accuracy']:.2%} | "
            f"{pair['delta_accuracy_pp']:+.2f} | "
            f"{pair['shot_delta_accuracy_pp']:+.2f} |"
        )
    if pairs:
        lines += ["", "| 指标 | 均值 ± 样本标准差 |", "|---|---:|"]
        for key, label, factor in (
            ("feedback_accuracy", "有反馈准确率（%）", 100),
            ("no_feedback_accuracy", "无反馈准确率（%）", 100),
            ("delta_accuracy_pp", "配对准确率增益（百分点）", 1),
            (
                "shot_feedback_accuracy",
                f"{shared.config.shots} shots 有反馈准确率（%）",
                100,
            ),
            (
                "shot_no_feedback_accuracy",
                f"{shared.config.shots} shots 无反馈准确率（%）",
                100,
            ),
            ("shot_delta_accuracy_pp", "有限测量配对增益（百分点）", 1),
        ):
            value = stats[key]
            deviation = (
                "不估计（单种子）"
                if value["sample_std"] is None
                else f"{factor * value['sample_std']:.2f}"
            )
            lines.append(f"| {label} | {factor * value['mean']:.2f} ± {deviation} |")
        lines += [
            "",
            f"精确概率正收益种子：{result['positive_seeds']['accuracy']}/{len(pairs)}；"
            f"有限测量正收益种子：{result['positive_seeds']['shot_accuracy']}/{len(pairs)}。",
            "",
            "![配对准确率及逐种子增益](feedback_comparison.png)",
        ]
    lines += [
        "",
        "统计单位为训练种子；不完整配对不进入统计。"
        "BCE、平衡准确率、概念指标、错→对/对→错计数见 CSV/JSON。",
        "有限测量在各模型自身的联合 (m,y) 分布下采样；"
        "配对图片不等于相同的个体测量轨迹。",
        "已有预测默认核验复用；可用 --recompute 在 CUDA 上重新评价，未进行新训练。",
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
