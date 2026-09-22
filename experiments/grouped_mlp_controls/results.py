"""All learning-rate cells and validation-selected comparisons with the VQC."""

from __future__ import annotations

import csv
import io

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256

from .model import TASKS
from .protocol import Experiment
from .runner import verify_complete


def summarize(experiment: Experiment) -> dict:
    results = [
        verify_complete(experiment, task, lr)
        for task in TASKS
        for lr in experiment.config.learning_rates
    ]
    selected = {
        task: min(
            (r for r in results if r["task"] == task),
            key=lambda r: (r["validation"]["loss"], r["learning_rate"]),
        )
        for task in TASKS
    }
    reference_config = experiment.reference["config"]
    budgets_match = experiment.full_reference_inputs and (
        experiment.config.concept_epochs == reference_config["concept_epochs"]
        and experiment.config.label_epochs == reference_config["epochs"]
    )
    comparisons: list[dict] = []
    if budgets_match:
        for task, keys in (
            (
                "concept",
                [
                    f"sequential_{reference_config['concept_epochs']}",
                    f"joint_{reference_config['concept_epochs']}",
                ],
            ),
            (
                "label",
                [
                    f"joint_{reference_config['epochs']}",
                    f"sequential_{reference_config['epochs']}",
                ],
            ),
        ):
            mlp = selected[task]
            for key in keys:
                quantum = experiment.reference["endpoints"][key]
                qm = quantum["validation"][task]
                mm = mlp["validation"][task]
                if task == "concept":
                    deltas = {
                        "joint_map_accuracy": mm["joint_map_accuracy"]
                        - qm["joint_map_accuracy"],
                        "true_concept_probability": mm["true_concept_probability"]
                        - qm["joint_single_shot_probability"],
                    }
                else:
                    deltas = {
                        "accuracy": mm["accuracy"] - qm["accuracy"],
                        "balanced_accuracy": mm["balanced_accuracy"]
                        - qm["balanced_accuracy"],
                    }
                comparisons.append(
                    {
                        "task": task,
                        "mlp_cell": mlp["cell"],
                        "vqc_endpoint": key,
                        "mlp_minus_vqc": deltas,
                    }
                )
    rows = []
    for result in results:
        validation = result["validation"]
        rows.append(
            {
                "task": result["task"],
                "learning_rate": result["learning_rate"],
                "parameters": result["parameters"],
                "epochs": result["epochs"],
                "validation_loss": validation["loss"],
                "joint_map_accuracy": validation.get("concept", {}).get(
                    "joint_map_accuracy"
                ),
                "true_concept_probability": validation.get("concept", {}).get(
                    "true_concept_probability"
                ),
                "group_argmax_exact_accuracy": validation.get("concept", {}).get(
                    "group_argmax_exact_accuracy"
                ),
                "invalid_code_probability": validation.get("concept", {}).get(
                    "invalid_code_probability"
                ),
                "label_accuracy": validation.get("label", {}).get("accuracy"),
                "label_balanced_accuracy": validation.get("label", {}).get(
                    "balanced_accuracy"
                ),
                "selected": selected[result["task"]]["cell"] == result["cell"],
            }
        )
    summary = {
        "status": "complete",
        "test_evaluated": False,
        "manifest_sha256": experiment.manifest_hash,
        "reference_lock_sha256": sha256(experiment.output / "reference_lock.json"),
        "selected_by_task": selected,
        "all_cells": results,
        "rows": rows,
        "comparisons": comparisons,
        "reference_endpoints": experiment.reference["endpoints"],
        "quantum_parameters": experiment.reference["quantum_parameters"],
        "checks": {
            "all_cells_complete": True,
            "same_initialization_across_lrs": True,
            "same_order_across_lrs": True,
            "full_reference_input_and_order": experiment.full_reference_inputs,
            "reference_training_budgets_match": budgets_match,
        },
        "selection": "lowest fixed-final validation NLL/BCE; tie lower learning rate",
        "evidence_role": "validation-selected classical diagnostic; not test evidence",
        "comparison_limits": [
            "251 concept parameters vs 240 frontend; "
            "253 label parameters vs 264 total VQC.",
            "Equal parameter counts do not imply equal model capacity or compute.",
            "Label-only MLP has no concept supervision; VQC Joint uses concept labels.",
            "MLP has a learning-rate sweep; historical VQC reference has one fixed LR.",
            "True-concept probability is a classical predictive probability, "
            "not a physical measurement.",
        ],
    }
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (experiment.output / "summary.csv").write_text(stream.getvalue(), encoding="utf-8")
    lines = [
        "# 相近参数 MLP 对照：开发验证",
        "",
        "| 任务 | LR | 参数 | Epoch | 验证损失 | 概念 MAP | "
        "真概念概率 | Label 准确率 | 选中 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def percentage(value):
        return "—" if value is None else f"{value:.2%}"

    for row in rows:
        lines.append(
            f"| {row['task']} | {row['learning_rate']} | {row['parameters']} | "
            f"{row['epochs']} | {row['validation_loss']:.5f} | "
            f"{percentage(row['joint_map_accuracy'])} | "
            f"{percentage(row['true_concept_probability'])} | "
            f"{percentage(row['label_accuracy'])} | {'是' if row['selected'] else ''} |"
        )
    lines += [
        "",
        "按各任务固定末轮的验证 NLL/BCE 选择学习率，完整保留所有候选结果。",
        f"与 VQC 使用完整相同输入和顺序：{experiment.full_reference_inputs}；"
        f"预算匹配：{budgets_match}。",
        "Label-only MLP 与 Joint VQC 的监督信息不同；MLP 还进行了学习率筛选。",
        "这两项是概念预测和分类能力诊断，不能据此单独宣称量子优势。",
        "",
    ]
    if comparisons:
        lines += ["| 比较：所选 MLP 减 VQC | 指标 | 差值（百分点） |", "|---|---|---:|"]
        for comparison in comparisons:
            for metric, delta in comparison["mlp_minus_vqc"].items():
                lines.append(
                    f"| {comparison['mlp_cell']} − {comparison['vqc_endpoint']} | "
                    f"{metric} | {100 * delta:+.2f} |"
                )
        lines.append("")
    (experiment.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(experiment.output / "summary.json", summary)
    return summary
