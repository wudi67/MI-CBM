"""Prepare a frozen concept cache and immutable MLP initialization on CUDA."""

import os
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    atomic_checkpoint,
    atomic_json,
    cuda_runtime,
    report,
    rng_state,
    sha256,
    utc_now,
)
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .model import ConceptMLP
from .protocol import Config, check_output, source_hashes
from .reference import Reference


class Experiment:
    def __init__(
        self, config: Config, output: Path, resume: bool = False, control=None
    ):
        config.validate()
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        self.control = {"stop": False} if control is None else control
        self.runtime = cuda_runtime(config.seed)
        self.stage = "preparation"
        self.details: dict = {}
        self.output.mkdir(parents=True, exist_ok=True)
        source = Path(config.reference)
        identity = {
            "config": config.to_dict(),
            "runtime": self.runtime,
            "sources": source_hashes(),
            "reference_result_lock_sha256": sha256(source / "result_lock.json"),
            "reference_manifest_sha256": sha256(source / "manifest.json"),
        }
        self.manifest: dict
        if resume:
            self.manifest = read_json(self.output / "manifest.json")
            if any(self.manifest.get(k) != v for k, v in identity.items()):
                raise ValueError(
                    "Resume requires unchanged configuration, source and CUDA runtime"
                )
            verify_files(self.output, self.manifest["artifacts"])
        else:
            if any(
                (self.output / n).exists() for n in ("config.json", "manifest.json")
            ):
                raise FileExistsError("Output exists; use --resume or a fresh --out")
            atomic_json(self.output / "config.json", config.to_dict())
            self.manifest = {
                "schema": "grouped_robot_mlp_diagnostic.v1",
                "created_at": utc_now(),
                **identity,
                "artifacts": {"config.json": sha256(self.output / "config.json")},
                "architecture": "5 concepts -> Linear(5,16) -> Tanh -> Linear(16,1)",
                "parameters": 113,
                "training": "Independent: only training-set true concepts and labels",
                "evaluation": (
                    "sum over 32 original Born-weighted label probabilities; "
                    "threshold 0.5"
                ),
                "selection": "fixed final epoch, no LR or best-epoch selection",
                "evidence_role": (
                    "single-seed development diagnostic; different downstream interface"
                ),
                "test_read": False,
                "test_evaluated": False,
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        self.tick("verifying_reference")
        self.reference = Reference(source, self.runtime, self.tick)
        self.order_seed = self.reference.reference.config.seed
        self.order_offset = self.reference.reference.config.concept_epochs
        self.save_once(
            "reference_lock.json",
            {
                "manifest_sha256": self.manifest_hash,
                "artifacts": self.reference.pins,
                "checkpoint": str(self.reference.checkpoint),
                "frontend_sha256": self.reference.frontend_hash,
                "order_seed": self.order_seed,
                "order_epoch_offset": self.order_offset,
                "test_read": False,
            },
        )
        self.prepare_cache()
        self.prepare_initialization()
        self.verify_result_lock()
        report(
            f"[green]CUDA:[/green] {self.runtime['device']}; "
            "MLP 5->16->1, 113 parameters; "
            f"batch={config.batch_size}; epochs={config.epochs}. "
            "Frozen VQC; train and validation only."
        )

    def save_once(self, name: str, value: dict) -> None:
        path = self.output / name
        if path.exists():
            if read_json(path) != value:
                raise ValueError(f"Immutable artifact changed: {name}")
        else:
            atomic_json(path, value)

    def prepare_cache(self) -> None:
        path = self.output / "data_lock.json"
        if path.exists():
            lock = read_json(path)
            if lock["manifest_sha256"] != self.manifest_hash:
                raise ValueError("Concept cache manifest changed")
            verify_files(self.output, lock["artifacts"])
        else:
            packed, audit = self.reference.export(self.tick)
            atomic_checkpoint(self.output / "concept_cache.pt", packed)
            atomic_json(self.output / "frontend_cuda_check.json", audit)
            atomic_json(
                path,
                {
                    "manifest_sha256": self.manifest_hash,
                    "artifacts": {
                        n: sha256(self.output / n)
                        for n in ("concept_cache.pt", "frontend_cuda_check.json")
                    },
                    "test_read": False,
                },
            )
        self.data_hash = sha256(path)
        self.cpu_data = load(self.output / "concept_cache.pt")
        if set(self.cpu_data) != {"train", "validation"}:
            raise ValueError("Only train and validation may be cached")
        self.data = {}
        for role, value in self.cpu_data.items():
            raw = self.reference.quantum_raw[role]["measured"]
            for key in (
                "concepts",
                "labels",
                "source_index",
                "robot_ids",
                "concept_probabilities",
            ):
                if not torch.equal(value[key], raw[key]):
                    raise ValueError(
                        f"Cache changed the source rows or predictions: {role}/{key}"
                    )
            for name, probability in value["quantum_label_probabilities"].items():
                original = self.reference.quantum_raw[role][name][
                    "branch_label_mass"
                ].sum(1)
                if not torch.equal(probability, original):
                    raise ValueError("Cached quantum label probabilities changed")
            self.data[role] = {
                k: value[k].cuda()
                for k in ("concepts", "labels", "concept_probabilities")
            }

    def prepare_initialization(self) -> None:
        path = self.output / "initialization_lock.json"
        if not path.exists():
            # Source audits may construct quantum models; MLP initialization must
            # not depend on that work or on whether a concept cache was reused.
            cuda_runtime(self.config.seed)
            model = ConceptMLP().cuda()
            atomic_checkpoint(
                self.output / "initialization.pt",
                {
                    "model": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "rng": rng_state(),
                },
            )
            atomic_json(
                path,
                {
                    "manifest_sha256": self.manifest_hash,
                    "model_sha256": state_hash(model.state_dict()),
                    "artifacts": {
                        "initialization.pt": sha256(self.output / "initialization.pt")
                    },
                },
            )
        lock = read_json(path)
        verify_files(self.output, lock["artifacts"])
        self.initial = load(self.output / "initialization.pt")
        self.initial_hash = state_hash(self.initial["model"])
        if (
            lock["manifest_sha256"] != self.manifest_hash
            or lock["model_sha256"] != self.initial_hash
        ):
            raise ValueError("MLP initialization changed")

    def heartbeat(self, status: str, **details) -> None:
        self.details.update(details)
        atomic_json(
            self.output / "heartbeat.json",
            {
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "status": status,
                "stage": self.stage,
                "cell": "independent_mlp",
                "device": self.runtime["device"],
                "completed_cells": int(
                    (self.output / "training/completion_lock.json").exists()
                ),
                "total_cells": 1,
                "epochs_total": self.config.epochs,
                "completed_conditions": 3
                * sum(
                    (self.output / role / "evaluation_lock.json").exists()
                    for role in ("train", "validation")
                ),
                "total_conditions": 6,
                "test_evaluated": False,
                **self.details,
            },
        )

    def tick(self, status: str, **details) -> None:
        self.heartbeat(status, **details)
        if self.control["stop"]:
            raise InterruptedError("Pause requested; historical files remain unchanged")

    def verify_result_lock(self) -> dict | None:
        path = self.output / "result_lock.json"
        if not path.exists():
            return None
        value = read_json(path)
        if value["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Result manifest changed")
        verify_files(self.output, value["artifacts"])
        return value

    def verify_unchanged(self) -> None:
        self.reference.verify_unchanged()
        verify_files(ROOT, self.manifest["sources"])


def run_experiment(
    shared: Experiment, max_steps: int | None = None, preflight_only: bool = False
) -> None:
    from .results import finalize
    from .training import Route, verify_complete

    if preflight_only:
        shared.heartbeat(
            "paused", reason="Source and CUDA concept cache verified; use --resume"
        )
        return
    existing = shared.verify_result_lock() is not None
    shared.stage = "training"
    if not (shared.output / "training/completion_lock.json").exists():
        if existing:
            raise ValueError("Locked result is missing its training endpoint")
        Route(shared).run(max_steps)
    verify_complete(shared)
    shared.stage = "evaluation"
    finalize(shared, existing)
    shared.verify_unchanged()
    shared.heartbeat("complete", epoch_completed=shared.config.epochs)
    report(f"[green]MLP diagnostic complete:[/green] {shared.output / 'summary.md'}")
