"""Prepare once, train three cells, then evaluate eight fixed validation conditions."""

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
    rng_state,
    sha256,
    utc_now,
)
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .data import prepare, table_pins
from .evaluation import evaluate, metrics
from .model import RobotVQC
from .protocol import (
    CELLS,
    CONCEPTS,
    CONDITIONS,
    Config,
    check_output,
    read_json,
    source_hashes,
    verify_files,
)
from .training import Route, load, verify_complete


class Experiment:
    def __init__(
        self,
        config: Config,
        output: Path,
        resume: bool = False,
        control: dict | None = None,
    ) -> None:
        config.validate()
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        self.control = {"stop": False} if control is None else control
        self.runtime = cuda_runtime(config.seed)
        self.stage, self.cell = "preparation", "initialization"
        self.data: dict = {}
        self.state_cache: dict = {}
        self.cached_frontend: str | None = None
        self.new_conditions = 0
        self.output.mkdir(parents=True, exist_ok=True)
        pins = table_pins(Path(config.dataset))
        sources = source_hashes()
        self.manifest: dict
        if resume:
            self.manifest = read_json(self.output / "manifest.json")
            if (
                self.manifest["config"] != config.to_dict()
                or self.manifest["sources"] != sources
                or self.manifest["tables"] != pins
                or self.manifest["runtime"] != self.runtime
            ):
                raise ValueError(
                    "Resume requires unchanged config, sources, tables and CUDA runtime"
                )
            verify_files(self.output, self.manifest["artifacts"])
            self.initial = load(self.output / "initialization.pt")
        else:
            if (self.output / "manifest.json").exists() or (
                self.output / "config.json"
            ).exists():
                raise FileExistsError("Output exists; use --resume or a fresh --out")
            model = RobotVQC(front_layers=4, label_layers=1).cuda()
            if sum(p.numel() for p in model.parameters()) != 264:
                raise ValueError(
                    "Expected the unchanged 264-parameter dSprites circuit"
                )
            self.initial = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "rng": rng_state(),
            }
            del model
            atomic_checkpoint(self.output / "initialization.pt", self.initial)
            atomic_json(self.output / "config.json", config.to_dict())
            self.manifest = {
                "schema": "grouped_robot_pilot.v1",
                "created_at": utc_now(),
                "config": config.to_dict(),
                "runtime": self.runtime,
                "tables": pins,
                "sources": sources,
                "artifacts": {
                    name: sha256(self.output / name)
                    for name in ("config.json", "initialization.pt")
                },
                "architecture": (
                    "Fusion L4: A5+B5 uploaded, A5 measured, retained B5 plus "
                    "readout L1; 264 parameters"
                ),
                "concept_order": list(CONCEPTS),
                "concept_encoding": (
                    "five binary concepts, first bit most significant; all 32 "
                    "codes legal"
                ),
                "concept_objective": (
                    "negative log Born probability of the true five-bit record"
                ),
                "independent": (
                    "frozen frontend; true classical X controls during label "
                    "training; measured controls during normal inference"
                ),
                "no_feedback": (
                    "same frozen frontend and initial label circuit; "
                    "measurement retained, all X controls zero"
                ),
                "correction": (
                    "replace only chosen classical bits, retaining original m "
                    "branch weights and B states"
                ),
                "selection": (
                    "fixed final epochs; no best-epoch, seed or "
                    "accuracy-dependent selection"
                ),
                "evidence_role": (
                    "seed-zero validation pilot, not formal five-seed or test evidence"
                ),
                "engineering_subset": config.development,
                "test_read": False,
                "test_evaluated": False,
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        report("[cyan]Preparing Robot train/validation; test split is unused.[/cyan]")
        self.data, self.audit = prepare(config, self.output, self.tick)
        self.data_hash = sha256(self.output / "data_lock.json")
        self.verify_result_lock()
        report(
            f"[green]CUDA ready:[/green] {self.runtime['device']}; "
            f"train={len(self.data['train']['labels'])}, "
            f"val={len(self.data['validation']['labels'])}; "
            f"11 qubits, 264 parameters, batch={config.batch_size}"
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
                    (self.output / "training" / cell / "result.json").exists()
                    for cell in CELLS
                ),
                "total_cells": len(CELLS),
                "completed_conditions": len(
                    list((self.output / "validation").glob("*/*/evaluation_lock.json"))
                ),
                "total_conditions": len(CONDITIONS),
                "test_evaluated": False,
                **details,
            },
        )

    def tick(self, status: str, **details) -> None:
        self.heartbeat(status, **details)
        if self.stop_requested:
            raise InterruptedError("Pause requested")

    @staticmethod
    def make_model(weights: dict) -> RobotVQC:
        model = RobotVQC(front_layers=4, label_layers=1).cuda()
        model.load_state_dict(weights)
        return model

    def initial_for(self, cell: str) -> dict:
        if cell == "concept":
            return self.initial
        endpoint = load(self.output / "training/concept/endpoint.pt")
        return {"model": endpoint["model"], "rng": self.initial["rng"]}

    @torch.no_grad()
    def prepare_states(self, model: RobotVQC) -> None:
        fingerprint = state_hash(model.frontend.state_dict())
        if self.cached_frontend == fingerprint:
            return
        model.eval()
        for role, data in self.data.items():
            chunks = []
            for start in range(0, len(data["angles"]), self.config.eval_batch_size):
                chunks.append(
                    model.frontend(
                        data["angles"][start : start + self.config.eval_batch_size]
                    ).detach()
                )
                self.tick(
                    "caching_frontend",
                    offset=min(
                        start + self.config.eval_batch_size, len(data["angles"])
                    ),
                )
            self.state_cache[role] = torch.cat(chunks)
        self.cached_frontend = fingerprint

    def verify_result_lock(self) -> dict | None:
        path = self.output / "result_lock.json"
        if not path.exists():
            return None
        value = read_json(path)
        if value["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Result lock manifest changed")
        verify_files(self.output, value["artifacts"])
        return value

    def expected(self, cell: str, name: str, mask: int) -> dict:
        return {
            "manifest_sha256": self.manifest_hash,
            "data_lock_sha256": self.data_hash,
            "checkpoint_sha256": sha256(
                self.output / "training" / cell / "endpoint.pt"
            ),
            "training": cell,
            "condition": name,
            "correction_mask": mask,
            "zero_feedback": cell == "no_feedback",
            "role": "validation",
            "test_evaluated": False,
            "engineering_subset": self.config.development,
        }

    def verify_condition(self, cell: str, name: str, mask: int) -> dict | None:
        root = self.output / "validation" / cell / name
        if not (root / "evaluation_lock.json").exists():
            return None
        value = read_json(root / "evaluation.json")
        lock = read_json(root / "evaluation_lock.json")
        if any(
            value[k] != v or lock[k] != v
            for k, v in self.expected(cell, name, mask).items()
        ):
            raise ValueError("Evaluation condition provenance changed")
        verify_files(root, lock["artifacts"])
        raw = load(root / "predictions.pt")
        for key in ("source_index", "concepts", "labels", "robot_ids"):
            if not torch.equal(raw[key], self.data["validation"][key].cpu()):
                raise ValueError("Evaluation row identities/targets changed")
        if value["metrics"] != metrics(raw):
            raise ValueError("Metrics do not reproduce from saved probabilities")
        return value

    def evaluate_condition(self, cell: str, name: str, mask: int) -> dict:
        model = self.make_model(
            load(self.output / "training" / cell / "endpoint.pt")["model"]
        )
        model.requires_grad_(False)
        before = state_hash(model.state_dict())
        self.prepare_states(model)
        metric, raw = evaluate(
            model,
            self.data["validation"],
            self.config.eval_batch_size,
            zero=cell == "no_feedback",
            mask=mask,
            states=self.state_cache["validation"],
            tick=self.tick,
        )
        if state_hash(model.state_dict()) != before:
            raise ValueError("Evaluation modified model parameters")
        root = self.output / "validation" / cell / name
        expected = self.expected(cell, name, mask)
        value = {**expected, "n_samples": len(raw["labels"]), "metrics": metric}
        atomic_checkpoint(root / "predictions.pt", raw)
        atomic_json(root / "evaluation.json", value)
        atomic_json(
            root / "evaluation_lock.json",
            {
                **expected,
                "artifacts": {
                    filename: sha256(root / filename)
                    for filename in ("predictions.pt", "evaluation.json")
                },
            },
        )
        self.new_conditions += 1
        report(f"[green]{cell}/{name}:[/green] label={metric['label']['accuracy']:.2%}")
        return value


def run_experiment(
    shared: Experiment,
    max_steps: int | None = None,
    max_conditions: int | None = None,
    preflight_only: bool = False,
) -> None:
    from .results import summarize  # pylint: disable=import-outside-toplevel

    if preflight_only:
        shared.heartbeat("paused", reason="preflight complete; use --resume to train")
        return
    existing = shared.verify_result_lock()
    used_steps, training = 0, {}
    shared.stage = "training"
    for cell in CELLS:
        shared.cell = cell
        shared.tick("verifying_training")
        if not (shared.output / "training" / cell / "result.json").exists():
            if existing is not None:
                raise ValueError("Completed run is missing a training cell")
            if max_steps is not None and used_steps >= max_steps:
                raise InterruptedError("Configured new-update limit reached")
            route = Route(shared, cell)
            start = route.progress["global_step"]
            route.run(None if max_steps is None else start + max_steps - used_steps)
            used_steps += route.progress["global_step"] - start
            del route
        training[cell] = verify_complete(shared, cell)
    a, b = training["independent"], training["no_feedback"]
    for key in ("initial_model_sha256", "frontend_sha256", "global_step"):
        if a[key] != b[key]:
            raise ValueError("Independent and zero routes are not paired")
    shared.stage = "validation"
    evaluations = []
    for cell, name, mask in CONDITIONS:
        shared.cell = f"{cell}/{name}"
        shared.tick("verifying_condition")
        value = shared.verify_condition(cell, name, mask)
        if value is None:
            if existing is not None:
                raise ValueError("Completed run is missing an evaluation")
            if max_conditions is not None and shared.new_conditions >= max_conditions:
                raise InterruptedError("Configured condition limit reached")
            shared.evaluate_condition(cell, name, mask)
            value = shared.verify_condition(cell, name, mask)
        assert value is not None
        evaluations.append(value)
    verify_files(Path("/"), shared.manifest["tables"])
    verify_files(ROOT, shared.manifest["sources"])
    if existing is None:
        summarize(shared, training, evaluations)
    shared.heartbeat("complete")
