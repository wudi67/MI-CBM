"""Train six concept circuits from scratch, using immutable cached Robot inputs."""

import os
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    array_hash,
    atomic_checkpoint,
    atomic_json,
    cuda_runtime,
    report,
    sha256,
    utc_now,
)
from experiments.grouped_robot_continuation.reference import Reference
from experiments.grouped_robot_pilot.data import selected_indices
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .evaluation import evaluate_job
from .model import initial_diagnostics, initializations, verify_initializations
from .protocol import (
    AUTHOR_COMMIT,
    CELLS,
    PAPER,
    Config,
    cell_name,
    check_output,
    source_hashes,
)
from .training import Job, Route, verify_job


class Experiment:
    def __init__(
        self, config: Config, output: Path, resume: bool = False, control=None
    ):
        pilot = config.source()
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        self.control = {"stop": False} if control is None else control
        self.runtime = cuda_runtime(pilot.seed)
        self.stage, self.cell, self.details = "preparation", "reference", {}
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
                    "Resume requires unchanged config, source, reference and runtime"
                )
            verify_files(self.output, self.manifest["artifacts"])
        else:
            if any(
                (self.output / n).exists() for n in ("manifest.json", "config.json")
            ):
                raise FileExistsError("Output exists; use --resume or a fresh --out")
            atomic_json(self.output / "config.json", config.to_dict())
            self.manifest = {
                "schema": "grouped_robot_concept_init.v1",
                "created_at": utc_now(),
                **identity,
                "artifacts": {"config.json": sha256(self.output / "config.json")},
                "cells": [cell_name(*cell) for cell in CELLS],
                "frontend_layers": 4,
                "frontend_qubits": 10,
                "frontend_parameters": 240,
                "initialization_sigma": config.sigma,
                "uniform_range": [0, 3.141592653589793],
                "scope": (
                    "Only U3/CU3 initialization changes; "
                    "original four data re-uploading layers"
                ),
                "training": (
                    "Fresh Adam for every cell; joint true-concept NLL; "
                    "fixed sample-order seed zero"
                ),
                "selection": (
                    "All three initializations, fixed final epoch; "
                    "no best-epoch selection"
                ),
                "primary_metric": (
                    "all_concepts_accuracy: "
                    "five marginal threshold predictions all correct"
                ),
                "paper": PAPER,
                "author_code_commit": AUTHOR_COMMIT,
                "attribution": (
                    "H-EFT-VA-inspired scale; our Fusion circuit, "
                    "no transferred barren-plateau guarantee"
                ),
                "test_read": False,
                "test_evaluated": False,
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        self.tick("verifying_reference")
        self.reference = Reference(root, self.runtime, self.tick)
        self.save_once(
            "reference_lock.json",
            {"manifest_sha256": self.manifest_hash, "artifacts": self.reference.pins},
        )
        self.data = {}
        for role, limit in (
            ("train", config.train_limit),
            ("validation", config.val_limit),
        ):
            source = self.reference.data[role]
            index = selected_indices(source["concepts"].cpu(), limit, pilot.seed).cuda()
            self.data[role] = {k: v[index] for k, v in source.items()}
        self.save_once(
            "data_reference.json",
            {
                "manifest_sha256": self.manifest_hash,
                "reference_data_lock_sha256": self.reference.data_hash,
                "prepared_data_path": str(root / "data.pt"),
                "data_hashes": {
                    r: {k: array_hash(v.cpu().numpy()) for k, v in d.items()}
                    for r, d in self.data.items()
                },
                "test_read": False,
            },
        )
        self.data_hash = sha256(self.output / "data_reference.json")
        path = self.output / "initialization.pt"
        if not path.exists():
            atomic_checkpoint(path, initializations(self.reference.initial, config))
        self.initial = load(path)
        verify_initializations(self.initial, self.reference.initial, config)
        self.save_once(
            "initialization_lock.json",
            {
                "manifest_sha256": self.manifest_hash,
                "artifacts": {"initialization.pt": sha256(path)},
                "model_hashes": {
                    k: state_hash(v["model"]) for k, v in self.initial.items()
                },
            },
        )
        lock_path = self.output / "initial_diagnostics_lock.json"
        if lock_path.exists():
            lock = read_json(lock_path)
            if lock["manifest_sha256"] != self.manifest_hash:
                raise ValueError("Initial diagnostic manifest changed")
            verify_files(self.output, lock["artifacts"])
        else:
            self.tick("initial_diagnostics")
            atomic_json(
                self.output / "initial_diagnostics.json",
                initial_diagnostics(self.initial, self.data["train"], pilot.batch_size),
            )
            self.save_once(
                "initial_diagnostics_lock.json",
                {
                    "manifest_sha256": self.manifest_hash,
                    "artifacts": {
                        "initial_diagnostics.json": sha256(
                            self.output / "initial_diagnostics.json"
                        )
                    },
                },
            )
        self.verify_result_lock()
        report(
            f"[green]CUDA ready:[/green] {self.runtime['device']}; "
            f"six concept runs, 4 layers / 240 angles, sigma={config.sigma:.6f}"
        )

    def save_once(self, name: str, value: dict) -> None:
        path = self.output / name
        if path.exists():
            if read_json(path) != value:
                raise ValueError(f"Existing artifact changed: {name}")
        else:
            atomic_json(path, value)

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
                    (
                        self.output / cell_name(*cell) / "training/concept/result.json"
                    ).exists()
                    for cell in CELLS
                ),
                "total_cells": len(CELLS),
                "epochs_total": self.config.concept_epochs,
                "completed_conditions": len(
                    list(self.output.glob("*/init_*/*/evaluation_lock.json"))
                ),
                "total_conditions": 2 * len(CELLS),
                "test_evaluated": False,
                **self.details,
            },
        )

    def tick(self, status: str, **details) -> None:
        self.heartbeat(status, **details)
        if self.control["stop"]:
            raise InterruptedError("Pause requested")

    def verify_result_lock(self) -> dict | None:
        path = self.output / "result_lock.json"
        if not path.exists():
            return None
        value = read_json(path)
        if value["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Result lock manifest changed")
        verify_files(self.output, value["artifacts"])
        return value

    def verify_unchanged(self) -> None:
        self.reference.verify_unchanged()
        verify_files(ROOT, self.manifest["sources"])


def run_experiment(
    shared: Experiment, max_steps: int | None = None, preflight_only: bool = False
) -> None:
    if preflight_only:
        shared.heartbeat("paused", reason="Preflight complete; resume to train")
        return
    existing = shared.verify_result_lock() is not None
    evaluations, training = [], []
    remaining = max_steps
    for method, index in CELLS:
        shared.cell, shared.stage, shared.details = (
            cell_name(method, index),
            "training",
            {},
        )
        shared.tick("preparing_cell")
        job = Job(shared, method, index)
        if not (job.output / "training/concept/result.json").exists():
            if existing:
                raise ValueError("Locked result is missing completed training")
            if remaining is not None and remaining <= 0:
                raise InterruptedError("Requested new-update budget reached")
            route = Route(job)
            before = route.progress["global_step"]
            route.run(None if remaining is None else before + remaining)
            if remaining is not None:
                remaining -= route.progress["global_step"] - before
            del route
        training.append({"cell_name": job.name, **verify_job(job)})
        shared.stage = "evaluation"
        shared.tick("evaluating")
        evaluations.extend(evaluate_job(job, existing))
    shared.stage, shared.cell, shared.details = "summary", "all", {}
    shared.tick("verifying_sources")
    shared.verify_unchanged()
    if not existing:
        from .results import write_results  # pylint: disable=import-outside-toplevel

        write_results(shared, evaluations, training)
    shared.verify_result_lock()
    shared.stage = "complete"
    shared.heartbeat("complete", epoch_completed=shared.config.concept_epochs)
    report(
        "[green]Concept initialization comparison complete:[/green] "
        f"{shared.output / 'summary.md'}"
    )
