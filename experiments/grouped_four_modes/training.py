"""Adapters reuse the original Standard and Joint optimizer implementations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from experiments.grouped_control_diagnostics.protocol import Config as StandardConfig
from experiments.grouped_control_diagnostics.runner import RouteRun as StandardRun
from experiments.grouped_control_diagnostics.runner import (
    verify_complete as verify_standard,
)
from experiments.grouped_dynamic_vqc.runtime import atomic_json, report, sha256
from experiments.grouped_feedback_ablation.protocol import load_checkpoint, read_json
from experiments.grouped_feedback_ablation.runner import RouteRun as JointRun
from experiments.grouped_feedback_ablation.runner import verify_complete as verify_joint
from experiments.grouped_sequential_intervention.protocol import verify_artifacts
from experiments.grouped_shots_final.data import data_hashes, subset

from .protocol import TRAININGS


class Adapter:
    """Supply the historical runners with isolated paths and one outer heartbeat."""

    def __init__(self, outer: Any, seed: int, training: str, data: dict) -> None:
        self.outer = outer
        cfg = outer.config
        self.data = data
        self.runtime = outer.runtime
        self.manifest_hash = outer.manifest_hash
        self.initial = outer.sources.initial(seed)
        self.make_model = outer.sources.baseline.make_model
        self.routes = ("standard",)
        if training == "standard":
            self.output = outer.output / f"training/seed{seed}"
            self.config = StandardConfig(
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                eval_batch_size=cfg.eval_batch_size,
                learning_rate=cfg.learning_rate,
                grad_clip=cfg.grad_clip,
                seed=seed,
                checkpoint_steps=cfg.checkpoint_steps,
            )
        else:
            self.output = outer.output / "training"
            self.config = replace(
                outer.sources.baseline.config,
                joint_seed=seed,
                joint_epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                eval_batch_size=cfg.eval_batch_size,
                learning_rate=cfg.learning_rate,
                grad_clip=cfg.grad_clip,
                checkpoint_steps=cfg.checkpoint_steps,
            )

    @property
    def stop_requested(self) -> bool:
        return self.outer.stop_requested

    def initial_for(self, _cell: str) -> dict:
        return self.initial

    def heartbeat(self, status: str, **details: object) -> None:
        self.outer.heartbeat(
            "between_cells" if status == "complete" else status, **details
        )


class Standard(StandardRun):
    """Only redirect monitoring; inherit the exact original training calculation."""

    def heartbeat(self, status: str, **details: object) -> None:
        adapter: Any = self.shared
        self.stop_requested = adapter.stop_requested
        adapter.heartbeat(
            status,
            epoch_completed=self.progress["completed_epoch"],
            epochs_total=self.epochs,
            global_step=self.progress["global_step"],
            offset=self.progress["offset"],
            training_seconds=self.training_seconds,
            **details,
        )


def train(shared: Any, max_steps: int | None = None) -> None:
    """max_steps counts NEW Adam updates across cells in this invocation."""
    shared.stage = "training"
    sources, config = shared.sources, shared.config
    data = {
        role: {
            k: v.cuda() for k, v in subset(sources.baseline.data[role], limit).items()
        }
        for role, limit in (("train", config.train_limit), ("val", config.val_limit))
    }
    data_lock = {
        "manifest_sha256": shared.manifest_hash,
        "data_hashes": {role: data_hashes(value) for role, value in data.items()},
    }
    path = shared.output / "training_data_lock.json"
    if path.exists():
        if read_json(path) != data_lock:
            raise ValueError("Training input/order source changed")
    else:
        atomic_json(path, data_lock)
    lock_path = shared.output / "training_lock.json"
    existing = read_json(lock_path) if lock_path.exists() else None
    if existing is not None:
        if existing["manifest_sha256"] != shared.manifest_hash:
            raise ValueError("Training lock manifest mismatch")
        verify_artifacts(shared.output, existing["artifacts"])
    cells, artifacts, used_steps = [], {"training_data_lock.json": sha256(path)}, 0
    for seed in config.seed_list():
        for training in TRAININGS:
            shared.cell = f"seed{seed}/{training}"
            shared.tick("verifying_training", completed_cells=len(cells))
            endpoint = sources.model_path(seed, training)
            if sources.reuse_seed(seed):
                result = read_json(endpoint.parent / "result.json")
                origin = "historical_seed0"
            else:
                adapter: Any = Adapter(shared, seed, training, data)
                cell = str(endpoint.parent.relative_to(adapter.output))
                verify = verify_standard if training == "standard" else verify_joint
                if not (endpoint.parent / "result.json").exists():
                    if existing is not None:
                        raise ValueError("Locked training cell is missing")
                    if max_steps is not None and used_steps >= max_steps:
                        raise InterruptedError(
                            "Configured new training-step limit reached"
                        )
                    route = (
                        Standard(
                            adapter,
                            "standard",
                            resume=(endpoint.parent / "resume.pt").exists(),
                        )
                        if training == "standard"
                        else JointRun(adapter, cell)
                    )
                    before = route.progress["global_step"]
                    limit = (
                        None if max_steps is None else before + max_steps - used_steps
                    )
                    outcome = route.run(limit)
                    used_steps += route.progress["global_step"] - before
                    del route
                    if outcome["status"] != "complete":
                        raise InterruptedError(
                            "Training paused with complete Adam/RNG state"
                        )
                result = verify(adapter, cell)
                origin = "trained"
                for name in ("result.json", *result["artifacts"]):
                    item = endpoint.parent / name
                    artifacts[str(item.relative_to(shared.output))] = sha256(item)
            cells.append(
                {
                    "seed": seed,
                    "training": training,
                    "origin": origin,
                    "checkpoint": str(endpoint),
                    "checkpoint_sha256": sha256(endpoint),
                    "model_sha256": result["model_sha256"],
                    "initial_model_sha256": result["initial_model_sha256"],
                    "global_step": result["global_step"],
                }
            )
            shared.heartbeat("between_cells", completed_cells=len(cells))
    value = {
        "manifest_sha256": shared.manifest_hash,
        "cells": cells,
        "artifacts": artifacts,
    }
    if existing is not None and existing != value:
        raise ValueError("Completed training registry changed")
    if existing is None:
        atomic_json(lock_path, value)
    # Validate the shared initialization, including reused seed 0, explicitly.
    for seed in config.seed_list():
        initials = {row["initial_model_sha256"] for row in cells if row["seed"] == seed}
        if len(initials) != 1:
            raise ValueError("Routes do not share initial weights")
    report(
        f"[green]Training verified:[/green] {len(cells)} cells; "
        f"final epoch {config.epochs}"
    )


def verified_frontend(shared: Any, seed: int, training: str) -> str:
    """Every newly optimized mode has its own frontend; no cross-mode state cache."""
    from experiments.grouped_control_diagnostics.protocol import (
        module_hash,  # pylint: disable=import-outside-toplevel
    )

    return module_hash(
        load_checkpoint(shared.sources.model_path(seed, training))["model"], "frontend"
    )
