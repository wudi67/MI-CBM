"""Reproducible intervention metrics, sample transitions and bounded comparisons."""

import csv
from pathlib import Path
from typing import cast

from experiments.grouped_dynamic_vqc.evaluation import label_metrics
from experiments.grouped_dynamic_vqc.runtime import (
    atomic_checkpoint,
    atomic_json,
    sha256,
)
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_pilot.evaluation import concept_metrics
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .model import CONDITIONS, ConceptMLP, predict, scores, transitions


def evaluate_role(shared, model: ConceptMLP, role: str) -> tuple[dict, dict]:
    data = shared.cpu_data[role]
    raw = predict(model, shared.data[role])
    raw.update(
        {k: data[k] for k in ("concepts", "labels", "source_index", "robot_ids")}
    )
    original = raw["branch_label_mass"]["measured"].sum(1)
    changed, indices = {}, {}
    for name, _ in CONDITIONS[1:]:
        after = raw["branch_label_mass"][name].sum(1)
        changed[name] = transitions(original, after, data["labels"])
        before_correct = (original >= 0.5) == data["labels"].bool()
        after_correct = (after >= 0.5) == data["labels"].bool()
        indices[name] = {
            "correct_to_wrong": data["source_index"][before_correct & ~after_correct],
            "wrong_to_correct": data["source_index"][~before_correct & after_correct],
        }
    raw["transition_source_indices"] = indices
    quantum = data["quantum_label_probabilities"]
    value = {
        "manifest_sha256": shared.manifest_hash,
        "checkpoint_sha256": sha256(shared.output / "training/endpoint.pt"),
        "data_lock_sha256": shared.data_hash,
        "role": role,
        "n_samples": len(data["labels"]),
        "mlp": scores(raw, data["labels"]),
        "mlp_transitions": changed,
        "quantum": {
            name: label_metrics(p, data["labels"]) for name, p in quantum.items()
        },
        "quantum_transitions": {
            name: transitions(quantum["measured"], p, data["labels"])
            for name, p in quantum.items()
            if name != "measured"
        },
        "concept": concept_metrics(data["concept_probabilities"], data["concepts"]),
        "test_evaluated": False,
    }
    return value, raw


def plots(output: Path, values: dict, history: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    names = [name for name, _ in CONDITIONS]
    for offset, family, title in (
        (-0.18, "quantum", "Existing VQC"),
        (0.18, "mlp", "Concept MLP"),
    ):
        axes[0].bar(
            [i + offset for i in range(3)],
            [100 * values["validation"][family][n]["accuracy"] for n in names],
            width=0.36,
            label=title,
        )
    axes[0].set_xticks(range(3), ["Normal", "Correct foot", "Correct all"])
    axes[0].set_ylabel("Validation label accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    rows = [r for r in history if "validation" in r]
    for role in ("train", "validation"):
        axes[1].plot(
            [r["epoch"] for r in rows],
            [r[role]["direct_true"]["bce"] for r in rows],
            marker=".",
            label=role,
        )
    axes[1].set_xlabel("MLP epoch")
    axes[1].set_ylabel("True-concept label BCE")
    axes[1].legend()
    fig.suptitle(
        "Robot diagnostic: fixed VQC concepts, different downstream interfaces"
    )
    fig.tight_layout()
    fig.savefig(output / "diagnostic.png", dpi=180)
    plt.close(fig)


def summarize(shared, values: dict) -> None:
    rows = []
    for role, result in values.items():
        for family in ("quantum", "mlp"):
            baseline = result[family]["measured"]["accuracy"]
            for condition, metric in result[family].items():
                rows.append(
                    {
                        "role": role,
                        "model": family,
                        "condition": condition,
                        "n_samples": result["n_samples"],
                        **metric,
                        "accuracy_change_pp": 100 * (metric["accuracy"] - baseline),
                    }
                )
    summary = {
        "manifest_sha256": shared.manifest_hash,
        "rows": rows,
        "transitions": {
            role: {family: v[f"{family}_transitions"] for family in ("mlp", "quantum")}
            for role, v in values.items()
        },
        "concept": {role: v["concept"] for role, v in values.items()},
        "test_read": False,
        "test_evaluated": False,
        "engineering_run": shared.config.development,
        "interpretation": (
            "Same frozen VQC concept probabilities and splits. "
            "MLP reads explicit concepts only; "
            "quantum decoder receives the retained state after conditional X. "
            "Downstream parameters: MLP 113 versus VQC 24. "
            "This diagnostic changes both the interface and decoder; "
            "it cannot isolate quantum expressivity or establish quantum advantage."
        ),
    }
    atomic_json(shared.output / "summary.json", summary)
    with (shared.output / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Robot：固定 VQC 概念预测的 Independent MLP 诊断",
        "",
        f"MLP 5→16→1，113 参数，固定 {shared.config.epochs} 轮，"
        f"seed {shared.config.seed}。",
        "训练仅使用训练集真实概念和 label；输入没有图像特征或保留量子态。",
        "正常与纠正预测均按原始 32 个测量分支的 Born 概率加权，阈值固定 0.5。",
        "没有读取或评价测试集；没有按最佳 epoch 或纠正收益挑选结果。",
        "",
        "| 数据 | 模型 | 条件 | label 准确率 | BCE | 相对正常预测变化（百分点） |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['role']} | {row['model']} | {row['condition']} | "
            f"{row['accuracy']:.2%} | {row['bce']:.5f} | "
            f"{row['accuracy_change_pp']:+.2f} |"
        )
    lines += [
        "",
        "## 纠正后的样本变化",
        "",
        "| 数据 | 模型 | 条件 | 对→错 | 错→对 | 净增加正确数 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for role, result in values.items():
        for family in ("quantum", "mlp"):
            for condition, change in result[f"{family}_transitions"].items():
                lines.append(
                    f"| {role} | {family} | {condition} | "
                    f"{change['correct_to_wrong']} | {change['wrong_to_correct']} | "
                    f"{change['net_correct']:+d} |"
                )
    lines += [
        "",
        "## 解释范围",
        "",
        "两者共享前半段和概念概率，但后半段输入形式、信息通路和参数量不同。",
        "若 MLP 能从真实概念准确预测，且全部纠正有效，"
        "说明应进一步检查当前量子接口和后半段；"
        "不能据此单独认定问题就是 VQC 的表达能力。",
        "部分概念纠正不保证单调提升。"
        "direct_true 是 MLP 直接接收真实概念的训练能力检查，"
        "应与 correct_all_five 在浮点精度内一致。",
        "原 VQC 没有保存训练集仅纠正脚的结果，该格不补造、不记为零。",
    ]
    if shared.config.development:
        lines += ["", "**这是工程验证预算，不能作为正式诊断结果。**"]
    (shared.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(shared, existing: bool) -> None:
    checkpoint = load(shared.output / "training/endpoint.pt")
    model = ConceptMLP().cuda()
    model.load_state_dict(checkpoint["model"])
    before = state_hash(model.state_dict())
    values = {}
    for role in ("train", "validation"):
        shared.tick("evaluating", role=role, epoch_completed=shared.config.epochs)
        value, raw = evaluate_role(shared, model, role)
        directory = shared.output / role
        if (directory / "evaluation_lock.json").exists():
            verify_files(
                directory, read_json(directory / "evaluation_lock.json")["artifacts"]
            )
            if read_json(directory / "evaluation.json") != value or tree_hash(
                load(directory / "predictions.pt")
            ) != tree_hash(raw):
                raise ValueError("Saved MLP metrics/predictions fail recomputation")
        else:
            if existing:
                raise ValueError("Locked result is missing an evaluation")
            atomic_checkpoint(directory / "predictions.pt", raw)
            atomic_json(directory / "evaluation.json", value)
            atomic_json(
                directory / "evaluation_lock.json",
                {
                    "artifacts": {
                        n: sha256(directory / n)
                        for n in ("predictions.pt", "evaluation.json")
                    }
                },
            )
        values[role] = value
    if before != state_hash(model.state_dict()):
        raise ValueError("Evaluation changed MLP weights")
    history = cast(list, read_json(shared.output / "training/history.json"))
    for role in ("train", "validation"):
        if history[-1][role] != values[role]["mlp"]:
            raise ValueError("Final training diagnostic differs from raw evaluation")
    shared.verify_unchanged()
    if not existing:
        summarize(shared, values)
        plots(shared.output, values, history)
        artifacts = {
            str(p.relative_to(shared.output)): sha256(p)
            for p in shared.output.rglob("*")
            if p.is_file()
            and p.suffix in {".json", ".pt", ".csv", ".md", ".png"}
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
