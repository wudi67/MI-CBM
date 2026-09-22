"""Two paired label continuations with the 300-epoch frontend fixed throughout."""

import os
from pathlib import Path
from typing import Any

from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    atomic_json,
    cuda_runtime,
    report,
    sha256,
    utc_now,
)
from experiments.grouped_robot_continuation.runner import (
    Experiment as PreviousExperiment,
)
from experiments.grouped_robot_continuation.runner import (
    evaluate_arm,
    verify_pair,
)
from experiments.grouped_robot_continuation.training import Job, verify_job
from experiments.grouped_robot_pilot.protocol import read_json, verify_files

from .protocol import CELLS, Config, check_output, source_hashes
from .reference import Reference
from .training import Route


class Experiment(PreviousExperiment):
    """Reuse cached CUDA evaluation, raw-metric verification and pause semantics."""

    # The parent constructor selects the earlier pilot, which is not our source.
    def __init__(
        self,
        config: Config,
        output: Path,
        resume: bool = False,
        control: dict | None = None,
    ):  # pylint: disable=super-init-not-called
        self.config: Any = config
        self.output = output.resolve()
        self.control = {"stop": False} if control is None else control
        self.stage, self.cell = "preparation", "long_concept_reference"
        self.state_cache = {}
        self.cached_frontend = None
        self.new_conditions = 0
        _, original = config.sources()
        check_output(config, self.output)
        self.runtime = cuda_runtime(original.seed)
        source = Path(config.reference)
        identity = {
            "config": config.to_dict(),
            "runtime": self.runtime,
            "sources": source_hashes(),
            "reference_manifest_sha256": sha256(source / "manifest.json"),
            "reference_result_lock_sha256": sha256(source / "result_lock.json"),
        }
        self.output.mkdir(parents=True, exist_ok=True)
        if resume:
            self.manifest = read_json(self.output / "manifest.json")
            if any(self.manifest.get(k) != v for k, v in identity.items()):
                raise ValueError(
                    "Resume requires unchanged source, config and CUDA runtime"
                )
            verify_files(self.output, self.manifest["artifacts"])
        else:
            if (self.output / "manifest.json").exists() or (
                self.output / "config.json"
            ).exists():
                raise FileExistsError("Output exists; use --resume or a fresh --out")
            atomic_json(self.output / "config.json", config.to_dict())
            self.manifest = {
                "schema": "grouped_robot_label_continuation.v1",
                "created_at": utc_now(),
                **identity,
                "source_arm": "long_concept",
                "selection": "fixed final budget; no best-epoch selection",
                "training": (
                    "freeze long-concept frontend; continue both label circuits "
                    "with full Adam"
                ),
                "sample_order": "preserve the original label epoch offset",
                "test_read": False,
                "test_evaluated": False,
                "artifacts": {"config.json": sha256(self.output / "config.json")},
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        self.tick("verifying_reference")
        self.reference: Any = Reference(source, self.runtime, self.tick)
        self.data = self.reference.data
        reference_lock = {
            "source_root": str(source.resolve()),
            "selected_arm": "long_concept",
            "artifacts": self.reference.pins,
            "actual_concept_epochs": self.reference.actual_concept_epochs,
            "label_start_epoch": original.head_epochs,
            "label_order_offset": original.concept_epochs,
            "test_read": False,
        }
        path = self.output / "reference_lock.json"
        if path.exists():
            if read_json(path) != reference_lock:
                raise ValueError("Reference lock changed")
        else:
            atomic_json(path, reference_lock)
        self.verify_result_lock()
        report(
            f"[green]CUDA:[/green] {self.runtime['device']}; "
            f"batch={original.batch_size}; "
            f"frozen concept epochs={self.reference.actual_concept_epochs}; "
            f"label {original.head_epochs}->{config.head_epochs}; "
            f"source arm=long_concept"
        )

    def heartbeat(self, status: str, **details) -> None:
        atomic_json(
            self.output / "heartbeat.json",
            {
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "status": status,
                "stage": self.stage,
                "cell": self.cell,
                "device": self.runtime["device"],
                "completed_cells": sum(
                    (
                        self.output
                        / "long_label/training"
                        / cell
                        / "completion_lock.json"
                    ).exists()
                    for cell in CELLS
                ),
                "total_cells": 2,
                "completed_conditions": len(
                    list(self.output.glob("*/*/*/*/evaluation_lock.json"))
                ),
                "total_conditions": 22,
                "test_evaluated": False,
                **details,
            },
        )


def run_experiment(
    shared: Experiment,
    max_steps: int | None = None,
    max_conditions: int | None = None,
    preflight_only: bool = False,
) -> None:
    from .results import summarize  # pylint: disable=import-outside-toplevel

    if preflight_only:
        shared.heartbeat("paused", reason="source verified; use --resume to continue")
        return
    existing = shared.verify_result_lock() is not None
    evaluations = evaluate_arm(shared, "baseline", max_conditions, existing)
    training, used_steps = {}, 0
    for cell in CELLS:
        shared.stage, shared.cell = "training", f"long_label/{cell}"
        shared.tick("verifying_training")
        job = Job(shared, "long_label", cell)
        if not (job.output / "training" / cell / "completion_lock.json").exists():
            if existing:
                raise ValueError("Locked result is missing a completed training phase")
            if max_steps is not None and used_steps >= max_steps:
                raise InterruptedError("New Adam update limit reached")
            route = Route(job)
            start = route.progress["global_step"]
            route.run(None if max_steps is None else start + max_steps - used_steps)
            used_steps += route.progress["global_step"] - start
            del route
        training[f"long_label/{cell}"] = verify_job(job)
    verify_pair(shared, "long_label", training)
    evaluations += evaluate_arm(shared, "long_label", max_conditions, existing)
    history = read_json(shared.output / "long_label/training/independent/history.json")
    for row in history:
        epoch = row["epoch"]
        required = epoch > shared.reference.config.head_epochs and (
            epoch % shared.config.diagnostic_every == 0
            or epoch == shared.config.head_epochs
        )
        if required and "validation_true" not in row:
            raise ValueError("A scheduled true-control diagnostic is missing")
    final_true = next(
        r["metrics"]
        for r in evaluations
        if (r["arm"], r["role"], r["training"], r["condition"])
        == ("long_label", "validation", "independent", "correct_all_five")
    )
    if history[-1]["validation_true"] != final_true:
        raise ValueError("Final true-control diagnostic differs from raw evaluation")
    shared.reference.verify_unchanged()
    verify_files(ROOT, shared.manifest["sources"])
    if not existing:
        summarize(shared, training, evaluations)
    shared.heartbeat("complete")
