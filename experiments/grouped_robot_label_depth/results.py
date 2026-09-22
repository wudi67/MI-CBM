"""Per-initialization scores and paired A/B summaries; no result-based selection."""

import csv
from pathlib import Path
from statistics import mean, stdev

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_robot_mlp_diagnostic.model import transitions
from experiments.grouped_robot_pilot.training import load

from .protocol import CONDITIONS, DEPTHS, INITIALIZATIONS, cell_name


def summarize_values(evaluations: list) -> tuple[list, list, list]:
    lookup = {
        (v["head_layers"], v["initialization_index"], v["role"], v["condition"]): v
        for v in evaluations
    }
    if len(lookup) != 36:
        raise ValueError(
            "All six cells and 36 unique evaluation conditions are required"
        )
    rows, aggregates, pairs = [], [], []
    for role in ("train", "validation"):
        for depth in DEPTHS:
            for index in INITIALIZATIONS:
                normal = lookup[depth, index, role, "measured"]["metrics"]["label"][
                    "accuracy"
                ]
                for name, _ in CONDITIONS:
                    value = lookup[depth, index, role, name]
                    rows.append(
                        {
                            "group": "A" if depth == 1 else "B",
                            "head_layers": depth,
                            "label_parameters": 22 * depth + 2,
                            "initialization_index": index,
                            "role": role,
                            "condition": name,
                            "n_samples": value["n_samples"],
                            **value["metrics"]["label"],
                            "correction_gain_pp": 100
                            * (value["metrics"]["label"]["accuracy"] - normal),
                            "reused_historical_training": value[
                                "reused_historical_training"
                            ],
                        }
                    )
            for name, _ in CONDITIONS:
                values = [
                    r
                    for r in rows
                    if r["head_layers"] == depth
                    and r["role"] == role
                    and r["condition"] == name
                ]
                record = {
                    "head_layers": depth,
                    "role": role,
                    "condition": name,
                    "n_initializations": 3,
                }
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "bce",
                    "correction_gain_pp",
                ):
                    record[f"{key}_mean"] = mean(v[key] for v in values)
                    record[f"{key}_std"] = stdev(v[key] for v in values)
                aggregates.append(record)
        for name, _ in CONDITIONS:
            differences = []
            for index in INITIALIZATIONS:
                a = next(
                    r
                    for r in rows
                    if (
                        r["head_layers"],
                        r["initialization_index"],
                        r["role"],
                        r["condition"],
                    )
                    == (1, index, role, name)
                )
                b = next(
                    r
                    for r in rows
                    if (
                        r["head_layers"],
                        r["initialization_index"],
                        r["role"],
                        r["condition"],
                    )
                    == (5, index, role, name)
                )
                differences.append(
                    {
                        "initialization_index": index,
                        "accuracy_change_pp": 100 * (b["accuracy"] - a["accuracy"]),
                        "bce_change": b["bce"] - a["bce"],
                        "correction_gain_change_pp": b["correction_gain_pp"]
                        - a["correction_gain_pp"],
                    }
                )
            pairs.append(
                {
                    "role": role,
                    "condition": name,
                    "direction": "B minus A",
                    "differences": differences,
                    **{
                        f"{key}_{stat}": function(v[key] for v in differences)
                        for key in (
                            "accuracy_change_pp",
                            "bce_change",
                            "correction_gain_change_pp",
                        )
                        for stat, function in (("mean", mean), ("std", stdev))
                    },
                }
            )
    return rows, aggregates, pairs


def plots(output: Path, aggregates: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = [name for name, _ in CONDITIONS]
    for depth, offset, title in (
        (1, -0.18, "A: 1 layer, 24 params"),
        (5, 0.18, "B: 5 layers, 112 params"),
    ):
        values = [
            next(
                v
                for v in aggregates
                if (v["head_layers"], v["role"], v["condition"])
                == (depth, "validation", name)
            )
            for name in names
        ]
        axes[0].bar(
            [i + offset for i in range(3)],
            [100 * v["accuracy_mean"] for v in values],
            yerr=[100 * v["accuracy_std"] for v in values],
            width=0.36,
            capsize=3,
            label=title,
        )
        axes[1].bar(
            depth,
            values[-1]["correction_gain_pp_mean"],
            yerr=values[-1]["correction_gain_pp_std"],
            capsize=4,
            label=title,
        )
    axes[0].set_xticks(range(3), ["Normal", "Correct foot", "Correct all"])
    axes[0].set_ylabel("Validation label accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_xticks([1, 5], ["A: 1 layer", "B: 5 layers"])
    axes[1].set_ylabel("All-concept correction gain (percentage points)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def summarize(shared, training: list, evaluations: list) -> None:
    rows, aggregates, pairs = summarize_values(evaluations)
    sample_changes = []
    for depth in DEPTHS:
        for index in INITIALIZATIONS:
            for role in ("train", "validation"):
                directory = shared.output / cell_name(depth, index) / role
                before = load(directory / "measured/predictions.pt")
                after = load(directory / "correct_all_five/predictions.pt")
                sample_changes.append(
                    {
                        "head_layers": depth,
                        "initialization_index": index,
                        "role": role,
                        **transitions(
                            before["branch_label_mass"].sum(1),
                            after["branch_label_mass"].sum(1),
                            before["labels"],
                        ),
                    }
                )
    # Explicitly audit that every initialization and depth saw the same sample orders.
    from .training import Job

    histories = [
        load(Job(shared, depth, index).checkpoint_path)["progress"]["history"]
        for depth, index in ((d, i) for d in DEPTHS for i in INITIALIZATIONS)
    ]
    orders = [[v["order_sha256"] for v in history] for history in histories]
    if any(values != orders[0] for values in orders[1:]):
        raise ValueError("A/B training sample orders are not identical")
    payload = {
        "manifest_sha256": shared.manifest_hash,
        "training": training,
        "rows": rows,
        "aggregates": aggregates,
        "paired_B_minus_A": pairs,
        "all_correction_sample_changes": sample_changes,
        "same_sample_orders_verified": True,
        "test_read": False,
        "test_evaluated": False,
        "engineering_run": shared.config.development,
        "interpretation": (
            "One fixed frontend and split, three label initializations; "
            "SD uses ddof=1. "
            "Both A and B use measurement feedback. Depth changes both parameter count "
            "and circuit composition; this is not a measurement/no-measurement "
            "comparison "
            "or three independently trained end-to-end models."
        ),
    }
    atomic_json(shared.output / "summary.json", payload)
    for name, values in (("summary.csv", rows), ("aggregates.csv", aggregates)):
        with (shared.output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = [
        "# Robot：measurement-induced 后半段深度 A/B 对照",
        "",
        "A：1 层、24 参数；B：5 层、112 参数。固定同一个前半段 VQC，三个后半段初始化。",
        f"每单元 {shared.config.head_epochs} 轮；复用历史 A0：{shared.reuse_a0}。",
        "Independent 训练；相同样本顺序，全部使用真实概念控制 X 门。",
        "验证集均值 ± 样本标准差；这是后半段初始化波动，不是完整模型三种子的泛化误差。",
        "",
        "| 后半段 | 条件 | label 准确率 | BCE | 纠正收益（百分点） |",
        "|---|---|---:|---:|---:|",
    ]
    for value in aggregates:
        if value["role"] == "validation":
            lines.append(
                f"| L{value['head_layers']} | {value['condition']} | "
                f"{100 * value['accuracy_mean']:.2f} ± "
                f"{100 * value['accuracy_std']:.2f}% | "
                f"{value['bce_mean']:.5f} ± {value['bce_std']:.5f} | "
                f"{value['correction_gain_pp_mean']:+.2f} ± "
                f"{value['correction_gain_pp_std']:.2f} |"
            )
    lines += [
        "",
        "## 每个初始化的验证结果",
        "",
        "| 后半段 | 初始化 | 条件 | 准确率 | BCE |",
        "|---|---:|---|---:|---:|",
    ]
    for row in rows:
        if row["role"] == "validation":
            lines.append(
                f"| L{row['head_layers']} | {row['initialization_index']} | "
                f"{row['condition']} | {row['accuracy']:.2%} | {row['bce']:.5f} |"
            )
    lines += [
        "",
        "## 训练集真实概念控制",
        "",
        "| 后半段 | 准确率 | BCE |",
        "|---|---:|---:|",
    ]
    for value in aggregates:
        if value["role"] == "train" and value["condition"] == "correct_all_five":
            lines.append(
                f"| L{value['head_layers']} | {100 * value['accuracy_mean']:.2f} ± "
                f"{100 * value['accuracy_std']:.2f}% | "
                f"{value['bce_mean']:.5f} ± {value['bce_std']:.5f} |"
            )
    lines += [
        "",
        "所有成对 B−A 差值及逐样本对错变化见 summary.json。",
        "没有 C/D、概念重新制备、MLP 或无反馈组；没有重训前半段或读取测试集。",
        "只取固定末轮，保留所有初始化，不根据验证效果挑选最佳种子。",
        "加深有效支持后半段深度的作用；加深无效不能单独证明接口有问题。",
        "新增训练的 history.json 保存每轮正常验证和定期真实控制诊断；"
        "复用 A0 保留其原历史，缺失的历史诊断不补造。",
    ]
    if shared.config.development:
        lines += ["", "**工程预算：不能作为正式实验结论。**"]
    (shared.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plots(shared.output, aggregates)
    shared.verify_unchanged()
    artifacts = {
        str(p.relative_to(shared.output)): sha256(p)
        for p in shared.output.rglob("*")
        if p.is_file()
        and p.suffix in {".pt", ".json", ".csv", ".md", ".png"}
        and p.name not in {"heartbeat.json", "result_lock.json"}
    }
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": artifacts,
            "test_evaluated": False,
        },
    )
