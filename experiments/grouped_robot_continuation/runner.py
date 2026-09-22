"""Run both independent continuations and evaluate matched control conditions."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    atomic_checkpoint,
    atomic_json,
    cuda_runtime,
    report,
    sha256,
    utc_now,
)
from experiments.grouped_robot_pilot.evaluation import evaluate, metrics
from experiments.grouped_robot_pilot.protocol import (
    CONDITIONS,
    read_json,
    verify_files,
)
from experiments.grouped_robot_pilot.protocol import (
    Config as PilotConfig,
)
from experiments.grouped_robot_pilot.runner import Experiment as PilotExperiment
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .protocol import ARMS, JOBS, Config, check_output, source_hashes
from .reference import Reference
from .training import Job, Route, verify_job

TRAIN_CONDITIONS = (
    ("independent", "measured", 0),
    ("independent", "correct_all_five", 31),
    ("no_feedback", "zero", 0),
)


class Experiment:
    def __init__(
        self,
        config: Config,
        output: Path,
        resume: bool = False,
        control: dict | None = None,
    ):
        self.config, self.output = config, output.resolve()
        self.control = {"stop": False} if control is None else control
        self.stage, self.cell = "preparation", "reference"
        self.state_cache: dict = {}
        self.cached_frontend: str | None = None
        self.new_conditions = 0
        reference = Path(config.reference).resolve()
        original = PilotConfig(**read_json(reference / "config.json"))
        config.validate(original)
        check_output(config, self.output, original)
        self.runtime = cuda_runtime(original.seed)
        identity = {
            "config": config.to_dict(),
            "runtime": self.runtime,
            "sources": source_hashes(),
            "reference_manifest_sha256": sha256(reference / "manifest.json"),
            "reference_result_lock_sha256": sha256(reference / "result_lock.json"),
        }
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest: dict
        if resume:
            self.manifest = read_json(self.output / "manifest.json")
            if any(self.manifest.get(k) != v for k, v in identity.items()):
                raise ValueError(
                    "Resume requires unchanged source, reference, config "
                    "and CUDA runtime"
                )
            verify_files(self.output, self.manifest["artifacts"])
        else:
            if (self.output / "manifest.json").exists() or (
                self.output / "config.json"
            ).exists():
                raise FileExistsError(
                    "Continuation exists; use --resume or a fresh --out"
                )
            atomic_json(self.output / "config.json", config.to_dict())
            self.manifest = {
                "schema": "grouped_robot_continuation.v1",
                "created_at": utc_now(),
                **identity,
                "artifacts": {"config.json": sha256(self.output / "config.json")},
                "selection": (
                    "fixed final endpoints; no best-epoch or result-dependent selection"
                ),
                "long_concept": (
                    "continue concept Adam; reset both label circuits to original "
                    "initialization, new Adam and original label sample orders"
                ),
                "long_label": (
                    "original frozen concept endpoint; continue each original "
                    "label model and full Adam separately"
                ),
                "evaluation": (
                    "baseline plus both arms; 8 validation conditions and "
                    "3 matched-control training diagnostics each"
                ),
                "test_evaluated": False,
                "test_read": False,
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        self.tick("verifying_reference")
        self.reference = Reference(reference, self.runtime, self.tick)
        self.data = self.reference.data
        reference_lock = {
            "reference": str(reference),
            "artifacts": self.reference.pins,
            "source_config": original.to_dict(),
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
            f"baseline={original.concept_epochs}+{original.head_epochs}; "
            f"long concept={config.concept_epochs}+{original.head_epochs}; "
            f"long label={original.concept_epochs}+{config.head_epochs}"
        )

    @property
    def stop_requested(self) -> bool:
        return self.control["stop"]

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
                        self.output / arm / "training" / cell / "completion_lock.json"
                    ).exists()
                    for arm, cell in JOBS
                ),
                "total_cells": len(JOBS),
                "completed_conditions": len(
                    list(self.output.glob("*/*/*/*/evaluation_lock.json"))
                ),
                "total_conditions": 33,
                "test_evaluated": False,
                **details,
            },
        )

    def tick(self, status: str, **details) -> None:
        self.heartbeat(status, **details)
        if self.stop_requested:
            raise InterruptedError("Pause requested; original pilot remains unchanged")

    @torch.no_grad()
    def prepare_states(self, model) -> None:
        fingerprint = state_hash(model.frontend.state_dict())
        if fingerprint == self.cached_frontend:
            return
        self.state_cache.clear()
        batch = self.reference.config.eval_batch_size
        for role, values in self.data.items():
            chunks = []
            for start in range(0, len(values["angles"]), batch):
                chunks.append(
                    model.frontend(values["angles"][start : start + batch]).detach()
                )
                self.tick(
                    "caching_frontend",
                    offset=min(start + batch, len(values["angles"])),
                    stage_samples=len(values["angles"]),
                )
            self.state_cache[role] = torch.cat(chunks)
        self.cached_frontend = fingerprint

    def checkpoint_path(self, arm: str, cell: str) -> Path:
        root = self.reference.output if arm == "baseline" else self.output / arm
        return root / "training" / cell / "endpoint.pt"

    def expected(self, arm: str, role: str, cell: str, name: str, mask: int) -> dict:
        return {
            "manifest_sha256": self.manifest_hash,
            "data_lock_sha256": self.reference.data_hash,
            "checkpoint_sha256": sha256(self.checkpoint_path(arm, cell)),
            "arm": arm,
            "role": role,
            "training": cell,
            "condition": name,
            "correction_mask": mask,
            "zero_feedback": cell == "no_feedback",
            "engineering_subset": self.config.development,
            "test_evaluated": False,
        }

    def verify_condition(
        self, arm: str, role: str, cell: str, name: str, mask: int
    ) -> dict | None:
        directory = self.output / arm / role / cell / name
        path = directory / "evaluation_lock.json"
        if not path.exists():
            return None
        locked = read_json(path)
        value = read_json(directory / "evaluation.json")
        expected = self.expected(arm, role, cell, name, mask)
        if any(locked.get(k) != v or value.get(k) != v for k, v in expected.items()):
            raise ValueError("Evaluation provenance changed")
        verify_files(directory, locked["artifacts"])
        raw = load(directory / "predictions.pt")
        for key in ("source_index", "labels", "concepts", "robot_ids"):
            if not torch.equal(raw[key], self.data[role][key].cpu()):
                raise ValueError("Evaluation identities/targets changed")
        if value["metrics"] != metrics(raw):
            raise ValueError("Evaluation metrics do not reproduce")
        if arm == "long_label":
            baseline = load(
                self.output / "baseline" / role / cell / name / "predictions.pt"
            )
            torch.testing.assert_close(
                raw["concept_probabilities"],
                baseline["concept_probabilities"],
                atol=2e-6,
                rtol=2e-6,
            )
        return value

    def evaluate_condition(
        self, arm: str, role: str, cell: str, name: str, mask: int
    ) -> None:
        directory = self.output / arm / role / cell / name
        reuse = arm == "baseline" and role == "validation"
        if reuse:
            raw = load(
                self.reference.output / "validation" / cell / name / "predictions.pt"
            )
            metric = metrics(raw)
        else:
            model = PilotExperiment.make_model(
                load(self.checkpoint_path(arm, cell))["model"]
            )
            model.eval().requires_grad_(False)
            before = state_hash(model.state_dict())
            self.prepare_states(model)
            metric, raw = evaluate(
                model,
                self.data[role],
                self.reference.config.eval_batch_size,
                zero=cell == "no_feedback",
                mask=mask,
                states=self.state_cache[role],
                tick=self.tick,
            )
            if before != state_hash(model.state_dict()):
                raise ValueError("Inference changed model parameters")
        expected = self.expected(arm, role, cell, name, mask)
        atomic_checkpoint(directory / "predictions.pt", raw)
        atomic_json(
            directory / "evaluation.json",
            {
                **expected,
                "n_samples": len(raw["labels"]),
                "metrics": metric,
                "reused_baseline_validation": reuse,
            },
        )
        atomic_json(
            directory / "evaluation_lock.json",
            {
                **expected,
                "artifacts": {
                    n: sha256(directory / n)
                    for n in ("predictions.pt", "evaluation.json")
                },
            },
        )
        self.new_conditions += 1
        report(f"{arm}/{role}/{cell}/{name}: label={metric['label']['accuracy']:.2%}")

    def verify_result_lock(self) -> dict | None:
        path = self.output / "result_lock.json"
        if not path.exists():
            return None
        value = read_json(path)
        if value["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Result manifest changed")
        verify_files(self.output, value["artifacts"])
        return value


def evaluate_arm(
    shared: Experiment, arm: str, limit: int | None, existing: bool
) -> list[dict]:
    shared.stage = "evaluation"
    values = []
    for role, conditions in (("validation", CONDITIONS), ("train", TRAIN_CONDITIONS)):
        for cell, name, mask in conditions:
            shared.cell = f"{arm}/{role}/{cell}/{name}"
            shared.tick("verifying_evaluation")
            value = shared.verify_condition(arm, role, cell, name, mask)
            if value is None:
                if existing:
                    raise ValueError("Locked result is missing an evaluation")
                if limit is not None and shared.new_conditions >= limit:
                    raise InterruptedError("New evaluation limit reached")
                shared.evaluate_condition(arm, role, cell, name, mask)
                value = shared.verify_condition(arm, role, cell, name, mask)
            assert value is not None
            values.append(value)
    return values


def verify_pair(shared: Experiment, arm: str, training: dict) -> None:
    a, b = (training[f"{arm}/{cell}"] for cell in ("independent", "no_feedback"))
    for key in ("initial_model_sha256", "frontend_sha256", "global_step"):
        if a[key] != b[key]:
            raise ValueError("The two label routes lost their pairing")
    if (
        arm == "long_label"
        and a["frontend_sha256"]
        != shared.reference.training["independent"]["frontend_sha256"]
    ):
        raise ValueError("Label continuation used the wrong concept endpoint")


def run_experiment(
    shared: Experiment,
    max_steps: int | None = None,
    max_conditions: int | None = None,
    preflight_only: bool = False,
) -> None:
    from .results import summarize  # pylint: disable=import-outside-toplevel

    if preflight_only:
        shared.heartbeat(
            "paused", reason="reference verified; use --resume to continue"
        )
        return
    existing = shared.verify_result_lock() is not None
    evaluations = evaluate_arm(shared, "baseline", max_conditions, existing)
    used_steps, training = 0, {}
    for arm in ARMS:
        shared.stage = "training"
        for candidate, cell in JOBS:
            if candidate != arm:
                continue
            shared.cell = f"{arm}/{cell}"
            shared.tick("verifying_training")
            job = Job(shared, arm, cell)
            if not (job.output / "training" / cell / "completion_lock.json").exists():
                if existing:
                    raise ValueError("Locked run is missing a training phase")
                if max_steps is not None and used_steps >= max_steps:
                    raise InterruptedError("New Adam update limit reached")
                route = Route(job)
                start = route.progress["global_step"]
                route.run(None if max_steps is None else start + max_steps - used_steps)
                used_steps += route.progress["global_step"] - start
                del route
            training[f"{arm}/{cell}"] = verify_job(job)
        verify_pair(shared, arm, training)
        evaluations += evaluate_arm(shared, arm, max_conditions, existing)
    shared.reference.verify_unchanged()
    verify_files(ROOT, shared.manifest["sources"])
    if not existing:
        summarize(shared, training, evaluations)
    shared.heartbeat("complete")
