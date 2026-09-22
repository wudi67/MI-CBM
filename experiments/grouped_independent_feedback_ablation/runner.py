"""Frozen CUDA evaluation, verified prediction reuse and condition-level recovery."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.data import stratified_indices
from experiments.grouped_dynamic_vqc.evaluation import concept_metrics, label_metrics
from experiments.grouped_dynamic_vqc.runtime import (
    array_hash,
    atomic_checkpoint,
    atomic_json,
    report,
    sha256,
    utc_now,
)
from experiments.grouped_feedback_ablation.evaluation import evaluate
from experiments.grouped_feedback_ablation.protocol import load_checkpoint, read_json
from experiments.grouped_sequential_intervention.protocol import verify_artifacts
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .protocol import CONTROLS, VARIANTS, Config, Sources, check_output, source_hashes


def recompute_metrics(raw: dict) -> dict:
    """Always derive reported metrics from saved per-image predictions."""
    for key in (
        "concept_probabilities",
        "label_probabilities",
        "shot_label_probabilities",
    ):
        value = raw[key]
        if (
            not torch.isfinite(value).all()
            or value.min() < -1e-6
            or value.max() > 1 + 1e-5
        ):
            raise ValueError(f"Invalid saved probabilities: {key}")
    p = raw["concept_probabilities"]
    if p.shape != (len(raw["labels"]), 32) or (p.sum(1) - 1).abs().max() > 1e-5:
        raise ValueError("Expected all 32 normalized Born branches")
    return {
        "n_samples": len(raw["labels"]),
        "concept": concept_metrics(p, raw["concepts"]),
        "label": label_metrics(raw["label_probabilities"], raw["labels"]),
        "finite_shots": {
            "label": label_metrics(raw["shot_label_probabilities"], raw["labels"])
        },
    }


class Experiment:
    def __init__(self, config: Config, output: Path, resume: bool = False) -> None:
        config.validate()
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        self.stop_requested = False
        self.sources = Sources(config)
        self.runtime = self.sources.runtime
        data = self.sources.baseline.data["val"]
        if config.val_limit > len(data["labels"]):
            raise ValueError("Validation subset exceeds source data")
        self.indices = stratified_indices(
            data["concepts"].cpu().numpy(), config.val_limit, 0
        )
        self.data = {k: v[self.indices] for k, v in data.items()}
        self.reuse = self.sources.can_reuse()
        reference = {"artifacts": self.sources.hashes, "pairing": self.sources.pairing}
        self.manifest: dict
        self.output.mkdir(parents=True, exist_ok=True)
        if resume:
            self.manifest = read_json(self.output / "manifest.json")
            if (
                self.manifest["config"] != config.to_dict()
                or self.manifest["sources"] != source_hashes()
            ):
                raise ValueError("Resume requires identical config and source hashes")
            if self.manifest["runtime"] != self.runtime:
                raise ValueError("Resume runtime mismatch")
            verify_artifacts(self.output, self.manifest["artifacts"])
            if read_json(self.output / "reference_lock.json") != reference:
                raise ValueError("Pinned training/evaluation sources changed")
            if (self.output / "result_lock.json").exists():
                lock = read_json(self.output / "result_lock.json")
                if lock["manifest_sha256"] != sha256(self.output / "manifest.json"):
                    raise ValueError("Summary provenance mismatch")
                verify_artifacts(self.output, lock["artifacts"])
        else:
            if (self.output / "config.json").exists() or (
                self.output / "manifest.json"
            ).exists():
                raise FileExistsError("Run exists; use --resume or a fresh --out")
            atomic_json(self.output / "config.json", config.to_dict())
            atomic_json(self.output / "reference_lock.json", reference)
            self.manifest = {
                "schema": "grouped_independent_feedback_ablation.v1",
                "created_at": utc_now(),
                "config": config.to_dict(),
                "sources": source_hashes(),
                "runtime": self.runtime,
                "artifacts": {
                    n: sha256(self.output / n)
                    for n in ("config.json", "reference_lock.json")
                },
                "validation_source_indices_sha256": array_hash(
                    self.data["source_index"].cpu().numpy()
                ),
                "engineering_subset": config.development,
                "test_evaluated": False,
                "paired_training_budget": True,
                "prediction_reuse": self.reuse,
                "architecture": "Fusion L4 (240 frozen); B5 + readout L1 (24 trained)",
                "training": "none; reuse separately trained final endpoints",
                "feedback_training": "true records; actual branches and B states",
                "evaluation": "unassisted measured versus zero; all 32 branches",
                "finite_shots": "joint (m,y), own control, seed + 937; evaluation only",
                "selection": "fixed final epoch; all requested training seeds",
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        if self.manifest["validation_source_indices_sha256"] != array_hash(
            self.data["source_index"].cpu().numpy()
        ):
            raise ValueError("Evaluation data changed")

    def heartbeat(self, status: str, **details: object) -> None:
        atomic_json(
            self.output / "heartbeat.json",
            {
                "status": status,
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "device": self.runtime["device"],
                "completed_conditions": len(
                    list(self.output.glob("seed*/*/evaluation_lock.json"))
                ),
                "total_conditions": 2 * len(self.config.seed_list()),
                "validation_samples": len(self.data["labels"]),
                **details,
            },
        )

    def directory(self, seed: int, variant: str) -> Path:
        return self.output / f"seed{seed}" / variant

    def expected(self, seed: int, variant: str) -> dict:
        return {
            "manifest_sha256": self.manifest_hash,
            "checkpoint_sha256": sha256(self.sources.model_path(seed, variant)),
            "seed": seed,
            "variant": variant,
            "control_mode": CONTROLS[variant],
        }

    def verify_condition(self, seed: int, variant: str) -> dict | None:
        path = self.directory(seed, variant)
        if not (path / "evaluation_lock.json").exists():
            return None
        lock = read_json(path / "evaluation_lock.json")
        if any(lock[k] != v for k, v in self.expected(seed, variant).items()):
            raise ValueError("Evaluation provenance mismatch")
        verify_artifacts(path, lock["artifacts"])
        return read_json(path / "evaluation.json")

    def validate_rows(self, raw: dict) -> None:
        for key in ("labels", "concepts", "source_index"):
            if not torch.equal(raw[key], self.data[key].cpu()):
                raise ValueError(f"Evaluation row pairing differs: {key}")

    @torch.no_grad()
    def evaluate_condition(
        self, seed: int, variant: str, states: torch.Tensor | None
    ) -> None:
        path = self.directory(seed, variant)
        source = self.sources.evaluation_path(seed, variant)
        old = load_checkpoint(source / "predictions.pt")
        old_metrics = read_json(source / "evaluation.json")
        self.heartbeat(
            "reusing" if self.reuse else "evaluating", cell=f"seed{seed}/{variant}"
        )
        if self.reuse:
            raw = old
        else:
            weights = load_checkpoint(self.sources.model_path(seed, variant))["model"]
            model = (
                self.sources.baseline.make_model(weights).eval().requires_grad_(False)
            )
            before = state_hash(model.state_dict())
            _, raw = evaluate(
                model,
                self.data,
                self.config.eval_batch_size,
                CONTROLS[variant],
                shots=self.config.shots,
                seed=seed,
                cached_states=states,
            )
            if state_hash(model.state_dict()) != before:
                raise RuntimeError("Evaluation changed frozen model")
            for key in ("concept_probabilities", "label_probabilities"):
                if (raw[key] - old[key][self.indices]).abs().max() > 2e-6:
                    raise ValueError(f"CUDA replay differs from source: {key}")
        self.validate_rows(raw)
        metrics = recompute_metrics(raw)
        if self.reuse and any(
            (
                metrics["label"] != old_metrics["label"],
                metrics["concept"] != old_metrics["concept"],
                metrics["finite_shots"]["label"]
                != old_metrics["finite_shots"]["label"],
            )
        ):
            raise ValueError("Saved metrics disagree with per-image predictions")
        metrics.update(self.expected(seed, variant))
        metrics.update(
            {
                "test_evaluated": False,
                "origin": "verified reuse" if self.reuse else "CUDA replay",
                "source_evaluation_sha256": sha256(source / "evaluation.json"),
            }
        )
        metrics["finite_shots"].update(
            {
                "shots_per_image": self.config.shots,
                "seed": seed + 937,
                "control_mode": CONTROLS[variant],
                "sampling": "joint (m,y) under own control",
            }
        )
        path.mkdir(parents=True, exist_ok=True)
        if self.reuse:
            shutil.copyfile(source / "predictions.pt", path / "predictions.pt")
        else:
            atomic_checkpoint(path / "predictions.pt", raw)
        atomic_json(path / "evaluation.json", metrics)
        atomic_json(
            path / "evaluation_lock.json",
            {
                **self.expected(seed, variant),
                "artifacts": {
                    n: sha256(path / n) for n in ("predictions.pt", "evaluation.json")
                },
            },
        )
        report(
            f"[green]seed {seed}, {variant}:[/green] "
            f"Label {metrics['label']['accuracy']:.2%}; "
            f"{self.config.shots} shots "
            f"{metrics['finite_shots']['label']['accuracy']:.2%}"
        )

    @torch.no_grad()
    def states_for(self, seed: int) -> torch.Tensor | None:
        if self.reuse:
            return None
        model = self.sources.baseline.make_model(
            load_checkpoint(self.sources.model_path(seed, "feedback"))["model"]
        )
        model.eval().requires_grad_(False)
        chunks = []
        for start in range(0, len(self.data["labels"]), self.config.eval_batch_size):
            self.heartbeat("caching_frontend", cell=f"seed{seed}", offset=start)
            if self.stop_requested:
                raise InterruptedError("Paused while caching validation states")
            chunks.append(
                model.frontend(
                    self.data["angles"][start : start + self.config.eval_batch_size]
                )
            )
        return torch.cat(chunks)


def run_experiment(shared: Experiment, max_conditions: int | None = None) -> dict:
    from .results import summarize  # pylint: disable=import-outside-toplevel

    completed = 0
    for seed in shared.config.seed_list():
        missing = [v for v in VARIANTS if shared.verify_condition(seed, v) is None]
        states = shared.states_for(seed) if missing else None
        for variant in missing:
            if shared.stop_requested:
                raise InterruptedError("Paused between conditions")
            shared.evaluate_condition(seed, variant, states)
            completed += 1
            if max_conditions is not None and completed >= max_conditions:
                shared.sources.verify()
                result = summarize(shared)
                shared.heartbeat(
                    "complete" if result["status"] == "complete" else "paused"
                )
                return result
        summarize(shared)
    shared.sources.verify()
    result = summarize(shared)
    shared.heartbeat("complete", cell="all paired evaluations complete")
    report(f"[green]完成：[/green]{shared.output / 'summary.md'}")
    return result
