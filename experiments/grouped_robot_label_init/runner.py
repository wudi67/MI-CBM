"""Nine initialization cells on one frozen frontend, with safe reuse and resume."""

import os
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    atomic_checkpoint,
    atomic_json,
    cuda_runtime,
    report,
    sha256,
    utc_now,
)
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .diagnostics import ensure_diagnostics
from .model import initializations, verify_initializations
from .protocol import (
    AUTHOR_COMMIT,
    CELLS,
    PAPER,
    Config,
    cell_name,
    check_output,
    source_hashes,
)
from .reference import Reference
from .training import Job, Route, lock_training, verify_job


class Experiment:
    def __init__(
        self, config: Config, output: Path, resume: bool = False, control=None
    ):
        depth, _, _, pilot = config.sources()
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        self.control = {"stop": False} if control is None else control
        self.runtime = cuda_runtime(pilot.seed)
        self.stage, self.cell, self.details = "preparation", "reference", {}
        self.reuse_uniform = config.head_epochs == depth.head_epochs
        self.output.mkdir(parents=True, exist_ok=True)
        root = Path(config.reference)
        identity = {
            "config": config.to_dict(),
            "runtime": self.runtime,
            "sources": source_hashes(),
            "reference_manifest_sha256": sha256(root / "manifest.json"),
            "reference_result_lock_sha256": sha256(root / "result_lock.json"),
        }
        self.manifest: dict
        if resume:
            self.manifest = read_json(self.output / "manifest.json")
            if any(self.manifest.get(k) != v for k, v in identity.items()):
                raise ValueError(
                    "Resume requires unchanged config, sources, reference and runtime"
                )
            verify_files(self.output, self.manifest["artifacts"])
        else:
            if any(
                (self.output / name).exists()
                for name in ("manifest.json", "config.json")
            ):
                raise FileExistsError("Output exists; use --resume or a fresh --out")
            atomic_json(self.output / "config.json", config.to_dict())
            self.manifest = {
                "schema": "grouped_robot_label_init.v1",
                "created_at": utc_now(),
                **identity,
                "artifacts": {"config.json": sha256(self.output / "config.json")},
                "cells": [cell_name(*cell) for cell in CELLS],
                "reuse_uniform": self.reuse_uniform,
                "head_layers": 5,
                "label_parameters": 112,
                "initialization_sigma": config.sigma,
                "scope": (
                    "Only label gate initialization changes; "
                    "same Independent measurement/X circuit"
                ),
                "selection": (
                    "Fixed final epoch, all initialization indices; no test evaluation"
                ),
                "paper": PAPER,
                "author_code_commit": AUTHOR_COMMIT,
                "attribution": (
                    "H-EFT-VA-inspired scale on our circuit; "
                    "shifted readout is our adaptation"
                ),
                "test_read": False,
                "test_evaluated": False,
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        self.tick("verifying_reference")
        self.reference = Reference(root, self.runtime, self.tick)
        self.data, self.state_cache = self.reference.data, self.reference.state_cache
        self.save_once(
            "reference_lock.json",
            {
                "manifest_sha256": self.manifest_hash,
                "artifacts": self.reference.pins,
                "frontend_sha256": self.reference.frontend_hash,
            },
        )
        self.save_once(
            "data_reference.json",
            {
                "manifest_sha256": self.manifest_hash,
                "reference_data_lock_sha256": self.reference.data_hash,
                "states_path": str(root / "frontend_states.pt"),
                "test_read": False,
            },
        )
        self.data_hash = sha256(self.output / "data_reference.json")
        self.prepare_initializations()
        self.verify_result_lock()
        ensure_diagnostics(self)
        report(
            f"[green]CUDA:[/green] {self.runtime['device']}; 5 layers/112 params; "
            f"sigma={config.sigma:.7f}; reuse uniform={self.reuse_uniform}; "
            f"{config.head_epochs} epochs; batch={pilot.batch_size}."
        )

    def save_once(self, name: str, value: dict) -> None:
        path = self.output / name
        if path.exists():
            if read_json(path) != value:
                raise ValueError(f"Immutable metadata changed: {name}")
        else:
            atomic_json(path, value)

    def prepare_initializations(self) -> None:
        path = self.output / "initialization_lock.json"
        if not path.exists():
            values = initializations(self.reference, self.config)
            atomic_checkpoint(self.output / "initialization.pt", values)
            atomic_json(
                path,
                {
                    "manifest_sha256": self.manifest_hash,
                    "artifacts": {
                        "initialization.pt": sha256(self.output / "initialization.pt")
                    },
                    "model_hashes": {
                        k: state_hash(v["model"]) for k, v in values.items()
                    },
                },
            )
        lock = read_json(path)
        if lock["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Initialization manifest changed")
        verify_files(self.output, lock["artifacts"])
        self.initial = load(self.output / "initialization.pt")
        verify_initializations(self.initial, self.reference, self.config)
        if lock["model_hashes"] != {
            k: state_hash(v["model"]) for k, v in self.initial.items()
        }:
            raise ValueError("Initialization model fingerprint changed")
        self.initial_hash = sha256(path)

    def activate(self, stage: str, cell: str) -> None:
        self.stage, self.cell, self.details = stage, cell, {}

    def heartbeat(self, status: str, **details) -> None:
        self.details.update(details)
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
                    (self.output / cell_name(*c) / "training_lock.json").exists()
                    for c in CELLS
                ),
                "total_cells": 9,
                "epochs_total": self.config.head_epochs,
                "completed_conditions": len(
                    list(self.output.glob("*/*/*/*/evaluation_lock.json"))
                ),
                "total_conditions": 36,
                "test_evaluated": False,
                **self.details,
            },
        )

    def tick(self, status: str, **details) -> None:
        self.heartbeat(status, **details)
        if self.control["stop"]:
            raise InterruptedError("Pause requested; resume to continue")

    def verify_result_lock(self) -> dict | None:
        path = self.output / "result_lock.json"
        if not path.exists():
            return None
        lock = read_json(path)
        if lock["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Result manifest changed")
        verify_files(self.output, lock["artifacts"])
        return lock

    def verify_unchanged(self) -> None:
        self.reference.verify_unchanged()
        verify_files(ROOT, self.manifest["sources"])


def run_experiment(
    shared: Experiment, max_steps: int | None = None, preflight_only: bool = False
) -> None:
    from .evaluation import evaluate_job
    from .results import summarize

    if preflight_only:
        shared.heartbeat(
            "paused",
            reason="Reference and initial CUDA diagnostics verified; use --resume",
        )
        return
    existing = shared.verify_result_lock() is not None
    used_steps, training, evaluations = 0, [], []
    for method, index in CELLS:
        job = Job(shared, method, index)
        shared.activate("training", job.name)
        shared.tick("verifying_training")
        if not (job.output / "training_lock.json").exists():
            if existing:
                raise ValueError("Locked result is missing a training cell")
            if not job.reused:
                if max_steps is not None and used_steps >= max_steps:
                    raise InterruptedError("New optimizer update budget reached")
                route = Route(job)
                start = route.progress["global_step"]
                route.run(None if max_steps is None else start + max_steps - used_steps)
                used_steps += route.progress["global_step"] - start
                del route
            lock_training(job)
        training.append(verify_job(job))
        shared.activate("evaluation", job.name)
        evaluations.extend(evaluate_job(job, existing))
    shared.verify_unchanged()
    if not existing:
        summarize(shared, training, evaluations)
    shared.activate("complete", "all")
    shared.heartbeat("complete", epoch_completed=shared.config.head_epochs)
    report(
        "[green]Initialization comparison complete:[/green] "
        f"{shared.output / 'summary.md'}"
    )
