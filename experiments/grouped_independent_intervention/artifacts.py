"""Completed weights, optimizer budgets and read-only historical import."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from experiments.grouped_control_diagnostics.protocol import module_hash
from experiments.grouped_dynamic_vqc.runtime import array_hash, atomic_json, sha256
from experiments.grouped_feedback_ablation.protocol import (
    cell_name,
    load_checkpoint,
    read_json,
)
from experiments.grouped_sequential_intervention.protocol import verify_artifacts
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .protocol import Experiment


def verify_complete(shared: Experiment, seed: int) -> dict:
    directory = shared.output / f"independent/seed{seed}"
    result = read_json(directory / "result.json")
    spec = shared.spec(seed)
    if (
        result["status"] != "complete"
        or result["spec"] != spec
        or result["manifest_sha256"] != shared.manifest_hash
        or result["test_evaluated"]
    ):
        raise ValueError("Completed Independent protocol mismatch")
    verify_artifacts(directory, result["artifacts"])
    checkpoint = load_checkpoint(directory / "endpoint.pt")
    initial = shared.initial_for(seed)
    steps = (
        math.ceil(len(shared.data["train"]["labels"]) / shared.config.batch_size)
        * spec["epochs"]
    )
    history = read_json(directory / "history.json")
    if (
        state_hash(checkpoint["model"]) != result["model_sha256"]
        or result["initial_model_sha256"] != state_hash(initial["model"])
        or module_hash(checkpoint["model"], "frontend")
        != module_hash(initial["model"], "frontend")
        or result["frontend_sha256"] != module_hash(initial["model"], "frontend")
        or result["head_sha256"] != module_hash(checkpoint["model"], "label_head")
        or result["global_step"] != steps
    ):
        raise ValueError(
            "Completed model/initialization/frozen frontend/budget mismatch"
        )
    p = checkpoint["progress"]
    if (
        len(history) != spec["epochs"]
        or p["history"] != history
        or p["global_step"] != steps
        or p["completed_epoch"] != spec["epochs"]
        or p["order"] is not None
        or p["offset"] != 0
    ):
        raise ValueError("Completed optimizer/history boundary mismatch")
    for epoch, row in enumerate(history, 1):
        order = epoch_order(
            len(shared.data["train"]["labels"]), seed, spec["offset"] + epoch
        )
        if row["epoch"] != epoch or row["order_sha256"] != array_hash(order.numpy()):
            raise ValueError("Completed sample order mismatch")
    optimizer = checkpoint["optimizer"]
    if len(optimizer["state"]) != 5 or len(optimizer["param_groups"]) != 1:
        raise ValueError(
            "Expected five quantum classification parameter tensors in Adam"
        )
    if any(int(state["step"]) != steps for state in optimizer["state"].values()):
        raise ValueError("Adam parameter update count differs from protocol")
    if result["origin"] == "historical":
        if (
            shared.true_reference["reuse_seed"] != seed
            or result["source"] != shared.true_reference["endpoint"]
            or sha256(directory / "endpoint.pt")
            != shared.true_reference["artifacts"][result["source"]]
        ):
            raise ValueError("Historical Independent endpoint mismatch")
    elif (
        checkpoint["manifest_sha256"] != shared.manifest_hash
        or checkpoint["spec"] != spec
        or checkpoint["cell"] != f"independent/seed{seed}"
    ):
        raise ValueError("Trained endpoint metadata mismatch")
    if shared.paired_training:
        old = read_json(
            shared.source.output
            / cell_name("sequential", seed, "feedback")
            / "history.json"
        )
        if [r["order_sha256"] for r in history] != [r["order_sha256"] for r in old]:
            raise ValueError("Independent and Sequential sample orders differ")
    return result


def import_historical(shared: Experiment, seed: int) -> dict:
    if shared.true_reference["reuse_seed"] != seed:
        raise ValueError("Seed has no compatible historical Independent model")
    source = Path(shared.true_reference["endpoint"])
    verify_artifacts(Path("/"), shared.true_reference["artifacts"])
    checkpoint = load_checkpoint(source)
    old = read_json(source.parent / "result.json")
    directory = shared.output / f"independent/seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, directory / "endpoint.pt")
    atomic_json(directory / "history.json", checkpoint["progress"]["history"])
    result = {
        "status": "complete",
        "origin": "historical",
        "source": str(source),
        "cell": f"independent/seed{seed}",
        "spec": shared.spec(seed),
        "manifest_sha256": shared.manifest_hash,
        "test_evaluated": False,
        **{
            k: old[k]
            for k in (
                "model_sha256",
                "initial_model_sha256",
                "frontend_sha256",
                "head_sha256",
                "global_step",
            )
        },
        "artifacts": {
            n: sha256(directory / n) for n in ("endpoint.pt", "history.json")
        },
    }
    atomic_json(directory / "result.json", result)
    return verify_complete(shared, seed)
