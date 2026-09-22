"""Evaluation-only worker; completed conditions are immutable resume units."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.data import stratified_indices
from experiments.grouped_dynamic_vqc.runtime import (
    array_hash,
    atomic_checkpoint,
    atomic_json,
    report,
    sha256,
    utc_now,
)
from experiments.grouped_feedback_ablation.protocol import (
    cell_name,
    load_checkpoint,
    read_json,
)
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .evaluation import evaluate
from .protocol import (
    MODES,
    Config,
    check_output,
    open_reference,
    source_hashes,
    verify_artifacts,
)


class Experiment:
    def __init__(self, config: Config, output: Path, resume: bool = False) -> None:
        config.validate()
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        self.stop_requested = False
        self.completed = 0
        self.source, reference = open_reference(config)
        self.runtime = self.source.runtime
        source_data = self.source.data["val"]
        if config.val_limit > len(source_data["labels"]):
            raise ValueError("Validation subset exceeds the source validation split")
        self.indices = stratified_indices(
            source_data["concepts"].cpu().numpy(), config.val_limit, 0
        )
        self.data = {key: value[self.indices] for key, value in source_data.items()}
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
                raise ValueError("Resume requires the original CUDA runtime")
            verify_artifacts(self.output, self.manifest["artifacts"])
            if reference != read_json(self.output / "reference_lock.json"):
                raise ValueError(
                    "Source experiment changed since this evaluation started"
                )
            result_lock_path = self.output / "result_lock.json"
            if result_lock_path.exists():
                result_lock = read_json(result_lock_path)
                if result_lock["manifest_sha256"] != sha256(
                    self.output / "manifest.json"
                ):
                    raise ValueError("Summary provenance mismatch")
                verify_artifacts(self.output, result_lock["artifacts"])
        else:
            if (self.output / "manifest.json").exists() or (
                self.output / "config.json"
            ).exists():
                raise FileExistsError(
                    "Evaluation exists; use --resume or a fresh --out"
                )
            atomic_json(self.output / "config.json", config.to_dict())
            atomic_json(self.output / "reference_lock.json", reference)
            self.manifest = {
                "schema": "grouped_sequential_intervention.v1",
                "created_at": utc_now(),
                "config": config.to_dict(),
                "sources": source_hashes(),
                "runtime": self.runtime,
                "artifacts": {
                    name: sha256(self.output / name)
                    for name in ("config.json", "reference_lock.json")
                },
                "validation_source_indices_sha256": array_hash(
                    self.data["source_index"].cpu().numpy()
                ),
                "n_samples": len(self.data["labels"]),
                "conditions": list(MODES),
                "training": "none; freeze existing final Sequential feedback endpoints",
                "intervention": (
                    "replace only classical X-control record after measurement; "
                    "preserve original Born weights and pre-X B branch states"
                ),
                "concept_units": "Shape (first 2 bits), Scale (last 3 bits)",
                "invalid_codes": (
                    "all 32 physical branches retained; no postselection "
                    "or renormalization to 18 valid codes"
                ),
                "finite_shots": (
                    "one fixed joint (m,y) sampling realization per training seed "
                    "and condition; seed + 937 + 100003 * condition_index"
                ),
                "threshold": 0.5,
                "evidence_role": (
                    "fixed-endpoint validation evaluation; "
                    "no test or noisy-training claim"
                ),
                "engineering_subset": bool(
                    config.val_limit
                    or self.source.config.train_limit
                    or self.source.config.val_limit
                ),
                "test_evaluated": False,
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        if self.manifest["validation_source_indices_sha256"] != array_hash(
            self.data["source_index"].cpu().numpy()
        ):
            raise ValueError("Evaluation row selection changed")

    def heartbeat(self, status: str, **details: object) -> None:
        atomic_json(
            self.output / "heartbeat.json",
            {
                "status": status,
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "device": self.runtime["device"],
                "completed_conditions": self.completed,
                "total_conditions": 4 * len(self.config.seed_list()),
                "validation_samples": len(self.data["labels"]),
                **details,
            },
        )

    def directory(self, seed: int, mode: str) -> Path:
        return self.output / f"seed{seed}" / mode

    def expected(self, seed: int, mode: str) -> dict:
        cell = cell_name("sequential", seed, "feedback")
        checkpoint = self.source.output / cell / "endpoint.pt"
        return {
            "manifest_sha256": self.manifest_hash,
            "checkpoint_sha256": sha256(checkpoint),
            "seed": seed,
            "control_mode": mode,
        }

    def verify_condition(self, seed: int, mode: str) -> dict | None:
        directory = self.directory(seed, mode)
        if not (directory / "evaluation_lock.json").exists():
            return None
        lock = read_json(directory / "evaluation_lock.json")
        if any(lock[k] != v for k, v in self.expected(seed, mode).items()):
            raise ValueError("Condition provenance mismatch")
        verify_artifacts(directory, lock["artifacts"])
        return read_json(directory / "evaluation.json")

    @torch.no_grad()
    def evaluate_seed(self, seed: int, modes: list[str], budget: int | None) -> int:
        cell = cell_name("sequential", seed, "feedback")
        weights = load_checkpoint(self.source.output / cell / "endpoint.pt")["model"]
        model = self.source.make_model(weights).eval().requires_grad_(False)
        before = state_hash(model.state_dict())
        chunks = []
        for start in range(0, len(self.data["labels"]), self.config.eval_batch_size):
            self.heartbeat("caching_frontend", cell=f"seed{seed}", offset=start)
            chunks.append(
                model.frontend(
                    self.data["angles"][start : start + self.config.eval_batch_size]
                )
            )
            if self.stop_requested:
                raise InterruptedError("Paused while preparing validation states")
        states = torch.cat(chunks)
        evaluated = 0
        for mode in modes:

            def progress(rows: int, active_mode: str = mode) -> None:
                self.heartbeat(
                    "evaluating", cell=f"seed{seed}/{active_mode}", offset=rows
                )
                if self.stop_requested:
                    raise InterruptedError(
                        "Current condition will be recomputed on resume"
                    )

            metrics, raw = evaluate(
                model,
                self.data,
                states,
                mode,
                batch_size=self.config.eval_batch_size,
                shots=self.config.shots,
                seed=seed,
                progress=progress,
            )
            if state_hash(model.state_dict()) != before:
                raise RuntimeError("Evaluation modified frozen model parameters")
            if mode == "measured":
                self.verify_baseline(seed, raw, metrics)
            metrics.update(self.expected(seed, mode))
            metrics["model_sha256"] = before
            directory = self.directory(seed, mode)
            atomic_checkpoint(directory / "predictions.pt", raw)
            atomic_json(directory / "evaluation.json", metrics)
            atomic_json(
                directory / "evaluation_lock.json",
                {
                    **self.expected(seed, mode),
                    "artifacts": {
                        name: sha256(directory / name)
                        for name in ("predictions.pt", "evaluation.json")
                    },
                },
            )
            self.completed += 1
            evaluated += 1
            report(
                f"[green]seed {seed}, {mode}:[/green] "
                f"Label {metrics['label']['accuracy']:.2%}; "
                f"{self.config.shots} shots "
                f"{metrics['finite_shots']['label']['accuracy']:.2%}"
            )
            if budget is not None and evaluated >= budget:
                break
        return evaluated

    def verify_baseline(self, seed: int, raw: dict, metrics: dict) -> None:
        cell = cell_name("sequential", seed, "feedback")
        old = load_checkpoint(self.source.output / cell / "predictions.pt")
        for key in ("labels", "concepts", "source_index"):
            if not torch.equal(raw[key], old[key][self.indices]):
                raise ValueError(
                    "Baseline rows or targets differ from the source evaluation"
                )
        errors = {}
        for key in ("concept_probabilities", "label_probabilities"):
            error = float((raw[key] - old[key][self.indices]).abs().max())
            errors[key] = error
            if error > 2e-6:
                raise ValueError(
                    "Baseline predictions differ from the frozen source endpoint"
                )
        same_sampling = (
            not self.config.val_limit
            and self.config.shots == self.source.config.shots
            and self.config.eval_batch_size == self.source.config.eval_batch_size
        )
        if same_sampling and not torch.equal(
            raw["shot_label_probabilities"], old["shot_label_probabilities"]
        ):
            raise ValueError("Baseline finite-shot reproduction failed")
        metrics["baseline_reproduction"] = {
            "max_errors": errors,
            "finite_shots_identical": True if same_sampling else None,
        }


def run_experiment(shared: Experiment, max_conditions: int | None = None) -> dict:
    from .results import summarize  # pylint: disable=import-outside-toplevel

    shared.completed = 0
    pending = {}
    for seed in shared.config.seed_list():
        pending[seed] = []
        for mode in MODES:
            if shared.verify_condition(seed, mode) is None:
                pending[seed].append(mode)
            else:
                shared.completed += 1
    newly_done = 0
    for seed, modes in pending.items():
        if not modes:
            continue
        budget = None if max_conditions is None else max_conditions - newly_done
        newly_done += shared.evaluate_seed(seed, modes, budget)
        result = summarize(shared)
        if shared.stop_requested or (
            max_conditions is not None and newly_done >= max_conditions
        ):
            reference = read_json(shared.output / "reference_lock.json")
            verify_artifacts(Path(reference["root"]), reference["artifacts"])
            shared.heartbeat("complete" if result["status"] == "complete" else "paused")
            return result
    reference = read_json(shared.output / "reference_lock.json")
    verify_artifacts(Path(reference["root"]), reference["artifacts"])
    result = summarize(shared)
    shared.heartbeat("complete")
    return result
