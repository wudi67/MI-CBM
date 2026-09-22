"""Seed-paired validation summaries; Joint pilot kept separate from Sequential."""

from __future__ import annotations

import csv
import statistics

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    atomic_checkpoint,
    atomic_json,
    sha256,
)

from .evaluation import evaluate
from .protocol import (
    Experiment,
    cell_name,
    cells,
    load_checkpoint,
    read_json,
)
from .runner import verify_complete


def evaluate_cell(shared: Experiment, cell: str) -> dict:
    result = verify_complete(shared, cell)
    directory = shared.output / cell
    lock_path = directory / "evaluation_lock.json"
    expected = {
        "manifest_sha256": shared.manifest_hash,
        "result_sha256": sha256(directory / "result.json"),
        "checkpoint_sha256": sha256(directory / "endpoint.pt"),
    }
    if lock_path.exists():
        lock = read_json(lock_path)
        if any(lock[key] != value for key, value in expected.items()):
            raise ValueError("Evaluation provenance mismatch")
        for name, digest in lock["artifacts"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"Evaluation artifact changed: {cell}/{name}")
        return read_json(directory / "evaluation.json")
    spec = result["spec"]
    shared.heartbeat(
        "finite_shot_evaluation", cell=cell, control_mode=spec["control_mode"]
    )
    model = shared.make_model(load_checkpoint(directory / "endpoint.pt")["model"])
    metrics, raw = evaluate(
        model,
        shared.data["val"],
        shared.config.eval_batch_size,
        spec["control_mode"],
        shots=shared.config.shots,
        seed=spec["seed"],
    )
    metrics.update(cell=cell, origin=result["origin"], **expected)
    atomic_checkpoint(directory / "predictions.pt", raw)
    atomic_json(directory / "evaluation.json", metrics)
    atomic_json(
        lock_path,
        {
            **expected,
            "artifacts": {
                name: sha256(directory / name)
                for name in ("evaluation.json", "predictions.pt")
            },
        },
    )
    return metrics


def paired_row(shared: Experiment, mode: str, seed: int) -> dict:
    names = [cell_name(mode, seed, v) for v in ("feedback", "no_feedback")]
    a, b = (evaluate_cell(shared, n) for n in names)
    ra, rb = (read_json(shared.output / n / "result.json") for n in names)
    if ra["initial_model_sha256"] != rb["initial_model_sha256"]:
        raise ValueError("Paired models have different initial tensors")
    ha, hb = (read_json(shared.output / n / "history.json") for n in names)
    if [r["order_sha256"] for r in ha] != [r["order_sha256"] for r in hb]:
        raise ValueError("Paired sample order mismatch")
    pa, pb = (load_checkpoint(shared.output / n / "predictions.pt") for n in names)
    if not torch.equal(pa["source_index"], pb["source_index"]):
        raise ValueError("Paired validation rows differ")
    error = float(
        (pa["concept_probabilities"] - pb["concept_probabilities"]).abs().max()
    )
    if mode == "sequential" and (
        ra["frontend_sha256"] != rb["frontend_sha256"] or error > 2e-6
    ):
        raise ValueError("Sequential pair changed its shared concept predictor")
    truth = pa["labels"].bool()
    ac, bc = pa["label_probabilities"] >= 0.5, pb["label_probabilities"] >= 0.5
    return {
        "mode": mode,
        "seed": seed,
        "feedback_accuracy": a["label"]["accuracy"],
        "no_feedback_accuracy": b["label"]["accuracy"],
        "delta_accuracy_pp": 100 * (a["label"]["accuracy"] - b["label"]["accuracy"]),
        "feedback_bce": a["label"]["bce"],
        "no_feedback_bce": b["label"]["bce"],
        "feedback_balanced_accuracy": a["label"]["balanced_accuracy"],
        "no_feedback_balanced_accuracy": b["label"]["balanced_accuracy"],
        "feedback_shot_accuracy": a["finite_shots"]["label"]["accuracy"],
        "no_feedback_shot_accuracy": b["finite_shots"]["label"]["accuracy"],
        "delta_shot_accuracy_pp": 100
        * (
            a["finite_shots"]["label"]["accuracy"]
            - b["finite_shots"]["label"]["accuracy"]
        ),
        "feedback_concept_group_exact": a["concept"]["group_argmax_exact_accuracy"],
        "no_feedback_concept_group_exact": b["concept"]["group_argmax_exact_accuracy"],
        "max_concept_probability_difference": error,
        "wrong_to_right_count": int(((bc != truth) & (ac == truth)).sum()),
        "right_to_wrong_count": int(((bc == truth) & (ac != truth)).sum()),
        "initialization_equal": True,
        "sample_orders_equal": True,
        "frontend_equal": ra["frontend_sha256"] == rb["frontend_sha256"],
    }


def statistics_for(rows: list[dict]) -> dict:
    """Training seeds, not validation images, are the repeat units."""
    result = {"n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
    for key in (
        "feedback_accuracy",
        "no_feedback_accuracy",
        "delta_accuracy_pp",
        "feedback_shot_accuracy",
        "no_feedback_shot_accuracy",
        "delta_shot_accuracy_pp",
    ):
        values = [r[key] for r in rows]
        result[key] = {
            "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        }
    return result


def summarize(shared: Experiment) -> dict:
    rows, locks = [], {}
    pairs = [
        ("joint", shared.config.joint_seed),
        *(("sequential", seed) for seed in shared.config.seed_list()),
    ]
    for mode, seed in pairs:
        names = [cell_name(mode, seed, v) for v in ("feedback", "no_feedback")]
        if not all((shared.output / n / "result.json").exists() for n in names):
            continue
        rows.append(paired_row(shared, mode, seed))
    for cell in cells(shared.config):
        for name in ("result.json", "evaluation_lock.json"):
            path = shared.output / cell / name
            if path.exists():
                locks[str(path.relative_to(shared.output))] = sha256(path)
    complete = all(
        (shared.output / c / "result.json").exists() for c in cells(shared.config)
    )
    sequential = [r for r in rows if r["mode"] == "sequential"]
    result = {
        "status": "complete" if complete else "partial",
        "test_evaluated": False,
        "engineering_subset": bool(
            shared.config.train_limit or shared.config.val_limit
        ),
        "train_samples": len(shared.data["train"]["angles"]),
        "validation_samples": len(shared.data["val"]["angles"]),
        "manifest_sha256": shared.manifest_hash,
        "paired_results": rows,
        "sequential_statistics": statistics_for(sequential),
        "joint_pilot": [r for r in rows if r["mode"] == "joint"],
        "planned_sequential_seeds": shared.config.seed_list(),
        "missing_sequential_seeds": [
            s
            for s in shared.config.seed_list()
            if s not in [r["seed"] for r in sequential]
        ],
        "interpretation": (
            "Joint is a single-seed development pilot, never pooled with Sequential"
        ),
        "evidence_role": "validation only; no hardware/noisy-training or test claim",
        "artifacts": locks,
    }
    atomic_json(shared.output / "summary.json", result)
    if rows:
        with (shared.output / "paired_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# 测量结果反馈消融：验证集结果",
        "",
        f"训练 {result['train_samples']} 张，验证 {result['validation_samples']} 张。",
        "工程子集检查，不作为论文结果。"
        if result["engineering_subset"]
        else "使用完整的预定训练／验证划分；尚未评价测试集。",
        "",
        "Joint 是单种子探索，Sequential 按预设种子配对；两种模式分别报告。",
        "统计单位为训练种子；sample_std 为样本标准差，单种子不报告标准差。",
        "训练采用精确概率；有限 shots 只用于评估，不代表带采样噪声训练。",
        "",
        "| 模式 | seed | 有反馈 | 无反馈 | 差值（百分点） | 有限 shots 差值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['seed']} | {row['feedback_accuracy']:.2%} | "
            f"{row['no_feedback_accuracy']:.2%} | {row['delta_accuracy_pp']:+.2f} | "
            f"{row['delta_shot_accuracy_pp']:+.2f} |"
        )
    stats = result["sequential_statistics"]
    lines += [
        "",
        f"Sequential 完成 {stats['n_seeds']}/{len(shared.config.seed_list())} 个种子。",
    ]
    if sequential:
        for key, label, factor in (
            ("feedback_accuracy", "有反馈准确率（%）", 100),
            ("no_feedback_accuracy", "无反馈准确率（%）", 100),
            ("delta_accuracy_pp", "配对差值（百分点）", 1),
        ):
            value = stats[key]
            deviation = (
                "未估计"
                if value["sample_std"] is None
                else f"{value['sample_std'] * factor:.2f}"
            )
            lines.append(
                f"- {label}：均值 {value['mean'] * factor:.2f}，"
                f"样本标准差 {deviation}。"
            )
    lines += [
        "",
        "所有预设种子均保留，包括负差值。当前不选择最终论文训练模式，不访问测试集。",
    ]
    (shared.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = ["summary.json", "summary.md"] + (
        ["paired_results.csv"] if rows else []
    )
    atomic_json(
        shared.output / "result_lock.json",
        {
            "manifest_sha256": shared.manifest_hash,
            "artifacts": {n: sha256(shared.output / n) for n in artifacts},
        },
    )
    return result
