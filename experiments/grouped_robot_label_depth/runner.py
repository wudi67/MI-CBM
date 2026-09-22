"""One frozen quantum frontend, six label jobs, and resumable A/B evaluation."""

import os
from pathlib import Path

import torch

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
from experiments.grouped_robot_mlp_diagnostic.reference import Reference
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .model import initializations, make_model, verify_initializations
from .protocol import CELLS, Config, cell_name, check_output, source_hashes
from .training import Job, Route, lock_training, verify_job


class Experiment:
    def __init__(
        self, config: Config, output: Path, resume: bool = False, control=None
    ):
        source_config, _, pilot = config.sources()
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        self.control = {"stop": False} if control is None else control
        self.runtime = cuda_runtime(pilot.seed)
        self.stage, self.cell, self.details = "preparation", "reference", {}
        self.reuse_a0 = config.head_epochs == source_config.head_epochs
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
                    "Resume requires unchanged config, sources, reference "
                    "and CUDA runtime"
                )
            verify_files(self.output, self.manifest["artifacts"])
        else:
            if any(
                (self.output / n).exists() for n in ("manifest.json", "config.json")
            ):
                raise FileExistsError("Output exists; use --resume or a fresh --out")
            atomic_json(self.output / "config.json", config.to_dict())
            self.manifest = {
                "schema": "grouped_robot_label_depth.v1",
                "created_at": utc_now(),
                **identity,
                "artifacts": {"config.json": sha256(self.output / "config.json")},
                "cells": [cell_name(*cell) for cell in CELLS],
                "reuse_a0": self.reuse_a0,
                "scope": (
                    "A/B only: same retained quantum states, "
                    "measurement and conditional X"
                ),
                "training": (
                    "Independent, fixed frontend; same samples, "
                    "orders, Adam and epoch budget"
                ),
                "initialization": (
                    "paired first layer and final readout; "
                    "B extra four layers uniform [-pi,pi]"
                ),
                "selection": "fixed final epoch; all three label initializations",
                "test_read": False,
                "test_evaluated": False,
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        self.tick("verifying_reference")
        self.reference = Reference(root, self.runtime, self.tick)
        self.data = self.reference.data
        self.save_once(
            "reference_lock.json",
            {
                "manifest_sha256": self.manifest_hash,
                "artifacts": self.reference.pins,
                "checkpoint": str(self.reference.checkpoint),
                "frontend_sha256": self.reference.frontend_hash,
                "test_read": False,
            },
        )
        self.prepare_initializations()
        self.prepare_states()
        self.verify_result_lock()
        report(
            f"[green]CUDA:[/green] {self.runtime['device']}; "
            f"A=L1/24 params, B=L5/112 params; {config.head_epochs} epochs; "
            f"batch={pilot.batch_size}; lr={pilot.learning_rate}; "
            f"reuse historical A0={self.reuse_a0}."
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
            values = initializations(self.reference)
            verify_initializations(values, self.reference.frontend_hash)
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
                    "extra_layer_seeds": {
                        k: v["extra_layers_seed"] for k, v in values.items()
                    },
                },
            )
        lock = read_json(path)
        if lock["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Initialization manifest changed")
        verify_files(self.output, lock["artifacts"])
        self.initial = load(self.output / "initialization.pt")
        if set(self.initial) != {cell_name(*cell) for cell in CELLS}:
            raise ValueError("Initialization cells changed")
        verify_initializations(self.initial, self.reference.frontend_hash)
        if lock["model_hashes"] != {
            k: state_hash(v["model"]) for k, v in self.initial.items()
        }:
            raise ValueError("Initial model fingerprint changed")

    @torch.no_grad()
    def prepare_states(self) -> None:
        path = self.output / "state_cache_lock.json"
        if not path.exists():
            model = make_model(load(self.reference.checkpoint)["model"])
            model.eval().requires_grad_(False)
            before = state_hash(model.state_dict())
            states, audit = {}, {}
            batch = self.reference.reference.config.eval_batch_size
            for role, data in self.data.items():
                chunks = []
                for start in range(0, len(data["angles"]), batch):
                    chunks.append(
                        model.frontend(data["angles"][start : start + batch])
                        .detach()
                        .cpu()
                    )
                    self.tick(
                        "caching_frozen_frontend",
                        role=role,
                        offset=start + len(chunks[-1]),
                        stage_samples=len(data["angles"]),
                    )
                states[role] = torch.cat(chunks)
                p = states[role].reshape(-1, 32, 32).abs().square().sum(-1)
                original = self.reference.quantum_raw[role]["measured"][
                    "concept_probabilities"
                ]
                torch.testing.assert_close(p, original, atol=2e-6, rtol=2e-6)
                audit[role] = {
                    "n_samples": len(p),
                    "max_born_difference": float((p - original).abs().max()),
                    "source_index_sha256": array_hash(
                        data["source_index"].cpu().numpy()
                    ),
                    "states_sha256": array_hash(states[role].numpy()),
                }
            if before != state_hash(model.state_dict()):
                raise ValueError("Caching changed the frozen frontend")
            atomic_checkpoint(self.output / "frontend_states.pt", states)
            atomic_json(
                self.output / "frontend_cuda_check.json",
                {
                    "device": "cuda:0",
                    "frontend_sha256": self.reference.frontend_hash,
                    "roles": audit,
                    "test_read": False,
                },
            )
            atomic_json(
                path,
                {
                    "manifest_sha256": self.manifest_hash,
                    "reference_data_lock_sha256": self.reference.reference.data_hash,
                    "artifacts": {
                        n: sha256(self.output / n)
                        for n in ("frontend_states.pt", "frontend_cuda_check.json")
                    },
                },
            )
        lock = read_json(path)
        if (
            lock["manifest_sha256"] != self.manifest_hash
            or lock["reference_data_lock_sha256"] != self.reference.reference.data_hash
        ):
            raise ValueError("State cache provenance changed")
        verify_files(self.output, lock["artifacts"])
        self.data_hash = sha256(path)
        states = load(self.output / "frontend_states.pt")
        if set(states) != {"train", "validation"}:
            raise ValueError("Only train/validation states may be cached")
        self.state_cache = {k: v.cuda() for k, v in states.items()}
        for role, value in self.state_cache.items():
            if value.shape != (len(self.data[role]["labels"]), 1024):
                raise ValueError("Cached state dimensions changed")
            original = self.reference.quantum_raw[role]["measured"][
                "concept_probabilities"
            ]
            torch.testing.assert_close(
                value.reshape(-1, 32, 32).abs().square().sum(-1).cpu(),
                original,
                atol=2e-6,
                rtol=2e-6,
            )

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
                    (self.output / cell_name(*cell) / "training_lock.json").exists()
                    for cell in CELLS
                ),
                "total_cells": 6,
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
            raise InterruptedError(
                "Pause requested; historical outputs remain unchanged"
            )

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
    from .evaluation import evaluate_job
    from .results import summarize

    if preflight_only:
        shared.heartbeat(
            "paused",
            reason="Sources, initializations and CUDA states verified; use --resume",
        )
        return
    existing = shared.verify_result_lock() is not None
    used_steps, training, evaluations = 0, [], []
    for depth, index in CELLS:
        job = Job(shared, depth, index)
        shared.activate("training", job.name)
        shared.tick("verifying_training")
        if not (job.output / "training_lock.json").exists():
            if existing:
                raise ValueError("Locked result is missing a training unit")
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
    report(f"[green]All A/B jobs verified:[/green] {shared.output / 'summary.md'}")
