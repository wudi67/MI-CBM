"""Verify pairing and summarize fixed endpoints without evaluating test data."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256

from .protocol import ROUTES, SharedExperiment


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_complete(shared: SharedExperiment, route: str) -> dict:
    directory = shared.output / route
    result = read_json(directory / "result.json")
    if (
        result["status"] != "complete"
        or result["epoch"] != shared.config.epochs
        or result["manifest_sha256"] != shared.manifest_hash
        or result["resume_sha256"] != sha256(directory / "resume.pt")
        or result["history_sha256"] != sha256(directory / "history.json")
    ):
        raise ValueError(f"Completed route failed provenance check: {route}")
    for epoch in (shared.config.concept_epochs, shared.config.epochs):
        stem = directory / "endpoints" / f"epoch_{epoch:04d}"
        record = read_json(stem.with_suffix(".json"))
        if (
            record["manifest_sha256"] != shared.manifest_hash
            or record["test_evaluated"]
            or record["checkpoint_sha256"] != sha256(stem.with_suffix(".pt"))
            or result["endpoint_records_sha256"][str(epoch)]
            != sha256(stem.with_suffix(".json"))
        ):
            raise ValueError(f"Endpoint failed provenance check: {stem}")
        if epoch == shared.config.epochs and any(
            result[key] != value for key, value in record.items()
        ):
            raise ValueError("Final result differs from the fixed endpoint")
    return result


def summarize(shared: SharedExperiment) -> dict:
    results = {route: verify_complete(shared, route) for route in ROUTES}
    histories = {
        route: read_json(shared.output / route / "history.json") for route in ROUTES
    }
    initial = {
        route: read_json(shared.output / route / "initialization.json")
        for route in ROUTES
    }
    if any(
        item["initial_model_sha256"] != shared.manifest["initial_model_sha256"]
        for item in initial.values()
    ):
        raise ValueError("Routes did not start from the shared initialization")
    if any(len(history) != shared.config.epochs for history in histories.values()):
        raise ValueError("Incomplete epoch history")
    if [row["order_sha256"] for row in histories["joint"]] != [
        row["order_sha256"] for row in histories["sequential"]
    ]:
        raise ValueError("Paired routes used different sample permutations")
    middle = {
        route: read_json(
            shared.output
            / route
            / "endpoints"
            / f"epoch_{shared.config.concept_epochs:04d}.json"
        )
        for route in ROUTES
    }
    sequential = results["sequential"]
    if (
        middle["sequential"]["frontend_sha256"] != sequential["frontend_sha256"]
        or middle["sequential"]["head_sha256"]
        != initial["sequential"]["initial_head_sha256"]
        or middle["sequential"]["validation"]["concept"]
        != sequential["validation"]["concept"]
    ):
        raise ValueError("Sequential freeze/unchanged concept prediction check failed")
    batches = math.ceil(len(shared.data["train"]["angles"]) / shared.config.batch_size)
    expected_steps = {
        "joint": {
            "frontend": batches * shared.config.epochs,
            "label_head": batches * shared.config.epochs,
        },
        "sequential": {
            "frontend": batches * shared.config.concept_epochs,
            "label_head": batches
            * (shared.config.epochs - shared.config.concept_epochs),
        },
    }
    if any(results[route]["module_steps"] != expected_steps[route] for route in ROUTES):
        raise ValueError("Parameter update budgets do not match the declared protocol")
    endpoints = [middle["joint"], middle["sequential"], results["joint"], sequential]
    rows = []
    for endpoint in endpoints:
        validation = endpoint["validation"]
        row = {
            "route": endpoint["route"],
            "phase": endpoint["phase"],
            "epoch": endpoint["epoch"],
            "label_head_trained": endpoint["label_head_trained"],
            **validation["concept"],
            "label_accuracy": validation["label"]["accuracy"],
            "label_balanced_accuracy": validation["label"]["balanced_accuracy"],
        }
        for control, metrics in validation["control_record_interventions"].items():
            row[f"control_{control}_accuracy"] = metrics["accuracy"]
            row[f"control_{control}_delta"] = (
                metrics["accuracy"] - validation["label"]["accuracy"]
            )
        rows.append(row)
    summary = {
        "status": "complete",
        "manifest_sha256": shared.manifest_hash,
        "evidence_role": "paired seed development validation; no test evaluation",
        "test_evaluated": False,
        "pairing_checks": {
            "same_initial_weights": True,
            "same_cached_inputs": True,
            "same_epoch_sample_order": True,
            "concept_stage_head_unchanged": True,
            "label_stage_frontend_unchanged": True,
            "update_budgets_match": True,
        },
        "rows": rows,
        "deltas": {
            "concept_only_minus_joint_at_concept_epoch": {
                key: middle["sequential"]["validation"]["concept"][key]
                - middle["joint"]["validation"]["concept"][key]
                for key in (
                    "joint_map_accuracy",
                    "joint_single_shot_probability",
                    "group_argmax_exact_accuracy",
                )
            },
            "sequential_minus_joint_final_label_accuracy": sequential["validation"][
                "label"
            ]["accuracy"]
            - results["joint"]["validation"]["label"]["accuracy"],
        },
        "endpoints": endpoints,
        "interpretation": [
            "Concept-only endpoint has an untrained label head; "
            "its label metrics are diagnostics only.",
            "Control interventions replace X-control records, "
            "not conditional quantum states.",
            "Sequential keeps the input-dependent residual pathway; "
            "this is not original Independent CBM.",
            "Equal total batch updates do not imply equal runtime "
            "or equal frontend/head updates.",
        ],
    }
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (shared.output / "summary.csv").write_text(stream.getvalue(), encoding="utf-8")
    lines = [
        "# 固定电路训练方式对照（开发验证）",
        "",
        "| 路线 | Epoch | 概念联合 MAP | 概念单次测量正确概率 | Label 准确率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = (
            f"{row['label_accuracy']:.2%}"
            if row["label_head_trained"]
            else "未训练分类头"
        )
        lines.append(
            f"| {row['route']}/{row['phase']} | {row['epoch']} | "
            f"{row['joint_map_accuracy']:.2%} | "
            f"{row['joint_single_shot_probability']:.2%} | {label} |"
        )
    lines += [
        "",
        "权重初始化、样本顺序、阶段冻结及参数更新预算核验通过。",
        "这是开发验证结果；控制记录纠正不等于量子状态纠正，保留残余图像通道。",
        "完整控制干预、有限 shots、训练时间及资源记录见 summary.json；"
        "逐轮曲线见各路线 history.json。",
        "",
    ]
    (shared.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(shared.output / "summary.json", summary)
    return summary
