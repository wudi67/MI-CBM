"""Resumable checkpoint diagnostics and separate mode/control/head tables."""

from __future__ import annotations

import csv
import io
import os
from collections.abc import Callable
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    atomic_checkpoint,
    atomic_json,
    sha256,
    utc_now,
)
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .diagnostics import MODES, diagnose
from .protocol import Experiment, read_json
from .runner import verify_complete


def checkpoint_paths(shared: Experiment) -> dict[str, Path]:
    return {
        "joint": shared.output / "joint_final.pt",
        "sequential": shared.output / "sequential_final.pt",
        **{route: shared.output / route / "endpoint.pt" for route in shared.routes},
    }


def verify_diagnostic(shared: Experiment, name: str, checkpoint: Path) -> dict:
    stem = shared.output / "diagnostics" / name
    record = read_json(stem.with_suffix(".json"))
    if (
        record["manifest_sha256"] != shared.manifest_hash
        or record["checkpoint_sha256"] != sha256(checkpoint)
        or record["raw_sha256"] != sha256(stem.with_suffix(".pt"))
        or record["pairing"] != shared.pairing
    ):
        raise ValueError(f"Diagnostic artifact changed: {name}")
    return record


def evaluate_models(shared: Experiment, stop: Callable[[], bool]) -> dict:
    records = {}
    for name, path in checkpoint_paths(shared).items():
        if stop():
            raise InterruptedError("Paused before diagnostics")
        if name in shared.routes:
            verify_complete(shared, name)
        stem = shared.output / "diagnostics" / name
        if stem.with_suffix(".json").exists():
            records[name] = verify_diagnostic(shared, name, path)
            continue
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        model = shared.make_model(checkpoint["model"])
        model_hash = state_hash(model.state_dict())

        def progress(done: int, total: int, model_name: str = name) -> None:
            atomic_json(
                shared.output / "heartbeat.json",
                {
                    "updated_at": utc_now(),
                    "pid": os.getpid(),
                    "status": "diagnostics",
                    "cell": model_name,
                    "offset": done,
                    "evaluation_rows": total,
                    "epoch_completed": 0,
                    "epochs_total": 0,
                    "global_step": 0,
                    "device": shared.runtime["device"],
                    "completed_cells": len(shared.routes),
                    "total_cells": len(shared.routes),
                },
            )
            if stop():
                raise InterruptedError(
                    "Paused during diagnostics; restart current model on resume"
                )

        progress(0, len(shared.data["val"]["angles"]))
        diagnostics, raw = diagnose(
            model,
            shared.data["val"],
            shared.config.eval_batch_size,
            shared.config.shots,
            shared.config.seed,
            progress,
        )
        if state_hash(model.state_dict()) != model_hash:
            raise RuntimeError("Diagnostics changed model tensors")
        atomic_checkpoint(stem.with_suffix(".pt"), raw)
        records[name] = {
            "name": name,
            "manifest_sha256": shared.manifest_hash,
            "checkpoint_sha256": sha256(path),
            "model_sha256": model_hash,
            "raw_sha256": sha256(stem.with_suffix(".pt")),
            "pairing": shared.pairing,
            "evaluation": diagnostics,
            "test_evaluated": False,
        }
        atomic_json(stem.with_suffix(".json"), records[name])
        del model, checkpoint
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def summarize(shared: Experiment) -> dict:
    training = {r: verify_complete(shared, r) for r in shared.routes}
    records = {
        name: verify_diagnostic(shared, name, path)
        for name, path in checkpoint_paths(shared).items()
    }
    if shared.reuse_measured_head:
        records["head_measured"] = records["sequential"]
    modes, controls, head_matrix, groups = [], [], [], []
    for name in ("standard", "joint", "sequential"):
        evaluation = records[name]["evaluation"]
        modes.append(
            {
                "model": name,
                "standard_joint_paired": shared.standard_paired,
                **evaluation["concept"],
                **{
                    f"label_{k}": v
                    for k, v in evaluation["controls"]["measured"].items()
                },
            }
        )
    for name, record in records.items():
        evaluation = record["evaluation"]
        for mode in MODES:
            controls.append(
                {"model": name, "control": mode, **evaluation["controls"][mode]}
            )
        for group, conditions in evaluation["groups"].items():
            for mode, metrics in conditions.items():
                # Empty bins carry null metrics with a fixed CSV schema.
                template = {k: None for k in evaluation["controls"][mode]}
                groups.append(
                    {
                        "model": name,
                        "group": group,
                        "control": mode,
                        **template,
                        **metrics,
                    }
                )
    for name in ("head_measured", "head_true", "head_zero"):
        for mode in ("measured", "both", "zero"):
            head_matrix.append(
                {
                    "head_training": name,
                    "evaluation_control": mode,
                    **records[name]["evaluation"]["controls"][mode],
                }
            )
    for filename, rows in (
        ("mode_comparison.csv", modes),
        ("control_comparison.csv", controls),
        ("head_matrix.csv", head_matrix),
        ("subgroups.csv", groups),
    ):
        write_csv(shared.output / filename, rows)
    lines = [
        "# 四层 Standard 与概念控制诊断",
        "",
        "固定终点、seed 0 开发验证；未评价测试集。",
        "",
        f"Standard 与历史 Joint 完整预算配对：{shared.standard_paired}。",
        f"复用历史 Sequential 作为测量控制头：{shared.reuse_measured_head}。",
        "",
        "| 模式 | Shape | Scale | 分组同时正确 | 联合 MAP | "
        "单次正确概率 | Label | BCE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in modes:
        keys = (
            "shape_marginal_argmax_accuracy",
            "scale_marginal_argmax_accuracy",
            "group_argmax_exact_accuracy",
            "joint_map_accuracy",
            "joint_single_shot_probability",
            "label_accuracy",
        )
        numbers = " | ".join(f"{row[k]:.2%}" for k in keys)
        lines.append(f"| {row['model']} | {numbers} | {row['label_bce']:.4f} |")
    lines += [
        "",
        "| 模型 | 原控制 | 真 Shape | 真 Scale | 两组真值 | 全零 | 随机控制精确平均 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in records.items():
        numbers = " | ".join(
            f"{record['evaluation']['controls'][m]['accuracy']:.2%}" for m in MODES
        )
        lines.append(f"| {name} | {numbers} |")
    lines += [
        "",
        "head_matrix.csv 给出同一冻结前端的 3×3 训练/评价控制矩阵。",
        "control_comparison.csv 同时报告 BCE、纠正/伤害样本数、概率变化和预测翻转。",
        "subgroups.csv 包含真实 18 概念组合、Label 和干预前置信边距分组及样本数。",
        "",
        "控制替换保持原测量分支、权重和条件态。随机控制仍会破坏控制与条件态的对应关系。",
        "Standard 编码没有概念监督；真实控制训练仍使用图像相关保留态，"
        "不是原 Independent CBM。",
        "零控制结果不能换算为泄漏比例；不同模式的模块更新次数和计算时间需分开报告。",
        "缩短预算或使用子集的运行只用于工程检查；"
        "Standard 配对为 False 时不能作完整模式结论。",
        "",
    ]
    (shared.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "status": "complete",
        "manifest_sha256": shared.manifest_hash,
        "pairing": shared.pairing,
        "training": training,
        "mode_comparison": modes,
        "head_matrix": head_matrix,
        "test_evaluated": False,
        "evidence_role": "fixed-endpoint development validation",
        "diagnostic_records_sha256": {
            name: sha256(shared.output / "diagnostics" / f"{name}.json")
            for name in checkpoint_paths(shared)
        },
    }
    atomic_json(shared.output / "summary.json", summary)
    return summary
