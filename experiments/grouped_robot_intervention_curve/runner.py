"""Read-only model inference with resumable commits per correction subset."""

import os
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    atomic_checkpoint,
    atomic_json,
    report,
    sha256,
    utc_now,
)
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_robot_shots_final.data import data_hashes
from experiments.grouped_robot_shots_final.evaluation import exact_metrics
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .evaluation import infer, verify_concepts
from .protocol import (
    MASKS,
    MODES,
    Config,
    check_output,
    selected_concepts,
    source_hashes,
)
from .reference import Sources


class Experiment:
    def __init__(self, config: Config, output: Path, resume: bool = False):
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.stop_requested = False
        self.cell = "source verification"
        self.data: dict = {}
        self.runtime = {"device": "initializing CUDA"}
        self.sources = Sources(config, self.tick)
        self.data, self.runtime = self.sources.data, self.sources.runtime
        self.new_conditions = 0
        self.cached_frontend = None
        self.state_cache = None
        reference = {"artifacts": self.sources.hashes}
        self.manifest: dict
        identity = {
            "config": config.to_dict(),
            "sources": source_hashes(),
            "runtime": self.runtime,
            "data_hashes": data_hashes(self.data),
        }
        if resume:
            self.manifest = read_json(self.output / "manifest.json")
            if any(self.manifest.get(k) != v for k, v in identity.items()):
                raise ValueError("Resume requires unchanged config/code/runtime/data")
            verify_files(self.output, self.manifest["artifacts"])
            if read_json(self.output / "reference_lock.json") != reference:
                raise ValueError("Historical reference changed")
        else:
            if (self.output / "config.json").exists() or (
                self.output / "manifest.json"
            ).exists():
                raise FileExistsError("Run exists; use --resume or another --out")
            atomic_json(self.output / "config.json", config.to_dict())
            atomic_json(self.output / "reference_lock.json", reference)
            self.manifest = {
                **identity,
                "schema": "grouped_robot_intervention_curve.v1",
                "created_at": utc_now(),
                "role": config.role,
                "n_samples": len(self.data["labels"]),
                "modes": list(MODES),
                "masks": list(MASKS),
                "masks_per_count": [1, 5, 10, 10, 5, 1],
                "selection": (
                    "all subsets; no chosen ordering or result-dependent selection"
                ),
                "count_definition": (
                    "number of queried concept positions, "
                    "including already correct bits"
                ),
                "intervention": (
                    "replace selected classical X controls; "
                    "retain Born branches and pre-feedback states"
                ),
                "prediction": "exact Born-weighted label probability >= 0.5",
                "aggregation": (
                    "score each mask before averaging masks of the same count "
                    "within seed; then mean/sample SD across seeds"
                ),
                "endpoints": (
                    "reuse locked zero/all correction predictions "
                    "from completed evaluation"
                ),
                "training": (
                    "none; frozen checkpoints with original architecture "
                    "and preprocessing"
                ),
                "test_scope": (
                    "completion of planned curve after earlier test results; "
                    "no tuning or retraining"
                ),
                "artifacts": {
                    p: sha256(self.output / p)
                    for p in ("config.json", "reference_lock.json")
                },
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        self.verify_result_lock()

    @property
    def total_conditions(self) -> int:
        return len(self.config.seed_list()) * len(MODES) * len(MASKS)

    def heartbeat(self, status: str, **details) -> None:
        atomic_json(
            self.output / "heartbeat.json",
            {
                "status": status,
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "stage": self.config.role,
                "cell": self.cell,
                "device": self.runtime["device"],
                "stage_samples": len(self.data.get("labels", [])),
                "completed_conditions": len(
                    list(self.output.glob("seed*/*/mask*/evaluation_lock.json"))
                ),
                "total_conditions": self.total_conditions,
                "test_read": not self.config.development and bool(self.data),
                "test_evaluated": not self.config.development
                and (self.output / "result_lock.json").exists(),
                **details,
            },
        )

    def tick(self, status: str, **details) -> None:
        self.heartbeat(status, **details)
        if self.stop_requested:
            raise InterruptedError("Pause requested; completed subsets are preserved")

    def verify_result_lock(self) -> dict | None:
        path = self.output / "result_lock.json"
        if not path.exists():
            return None
        lock = read_json(path)
        if lock["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Result manifest mismatch")
        verify_files(self.output, lock["artifacts"])
        return lock

    def directory(self, seed: int, mode: str, mask: int) -> Path:
        return self.output / f"seed{seed}" / mode / f"mask{mask:02d}"

    def expected(self, seed: int, mode: str, mask: int) -> dict:
        return {
            "manifest_sha256": self.manifest_hash,
            "checkpoint_sha256": self.sources.checkpoint_hashes[seed, mode],
            "role": self.config.role,
            "seed": seed,
            "training": mode,
            "mask": mask,
            "count": mask.bit_count(),
            "selected_concepts": selected_concepts(mask),
            "reused_endpoint": mask in (0, 31),
        }

    def verify_condition(
        self, seed: int, mode: str, mask: int, baseline: dict
    ) -> dict | None:
        root = self.directory(seed, mode, mask)
        if not (root / "evaluation_lock.json").exists():
            return None
        lock = read_json(root / "evaluation_lock.json")
        expected = self.expected(seed, mode, mask)
        if any(lock[k] != v for k, v in expected.items()):
            raise ValueError("Subset provenance changed")
        verify_files(root, lock["artifacts"])
        raw, record = load(root / "joint.pt"), read_json(root / "evaluation.json")
        verify_concepts(raw, baseline)
        if any(record[k] != v for k, v in expected.items()) or record[
            "metrics"
        ] != exact_metrics(raw):
            raise ValueError("Subset metrics do not reproduce from saved probabilities")
        if mask in (0, 31):
            historical = self.sources.endpoint(seed, mode, mask)
            if any(not torch.equal(v, historical[k]) for k, v in raw.items()):
                raise ValueError("Reused endpoint differs from the historical endpoint")
        return record

    @torch.no_grad()
    def states(self, model) -> torch.Tensor:
        fingerprint = state_hash(model.frontend.state_dict())
        if self.cached_frontend == fingerprint and self.state_cache is not None:
            return self.state_cache
        chunks, batch = [], self.config.eval_batch_size
        for start in range(0, len(self.data["labels"]), batch):
            chunks.append(
                model.frontend(self.data["angles"][start : start + batch].cuda())
            )
            self.tick(
                "caching_frontend", offset=min(start + batch, len(self.data["labels"]))
            )
        self.state_cache = torch.cat(chunks)
        self.cached_frontend = fingerprint
        return self.state_cache

    @torch.no_grad()
    def evaluate_condition(
        self, seed: int, mode: str, mask: int, model, baseline: dict
    ) -> None:
        root = self.directory(seed, mode, mask)
        root.mkdir(parents=True, exist_ok=True)
        if mask in (0, 31):
            raw = self.sources.endpoint(seed, mode, mask)
        else:
            before = state_hash(model.state_dict())
            raw = infer(
                model,
                self.data,
                self.states(model),
                mask,
                self.config.eval_batch_size,
                self.tick,
            )
            if state_hash(model.state_dict()) != before:
                raise ValueError("Frozen model changed during evaluation")
        verify_concepts(raw, baseline)
        record = {
            **self.expected(seed, mode, mask),
            "n_samples": len(self.data["labels"]),
            "metrics": exact_metrics(raw),
        }
        atomic_checkpoint(root / "joint.pt", raw)
        atomic_json(root / "evaluation.json", record)
        atomic_json(
            root / "evaluation_lock.json",
            {
                **self.expected(seed, mode, mask),
                "artifacts": {
                    p: sha256(root / p) for p in ("joint.pt", "evaluation.json")
                },
            },
        )
        self.new_conditions += 1


def run_experiment(
    shared: Experiment, max_conditions: int | None = None, preflight_only: bool = False
) -> None:
    from .results import summarize  # pylint: disable=import-outside-toplevel

    if preflight_only:
        shared.heartbeat("paused", reason="Preflight complete; resume to evaluate")
        return
    completed = shared.verify_result_lock()
    evaluations = []
    for seed in shared.config.seed_list():
        for mode in MODES:
            baseline = shared.sources.endpoint(seed, mode, 0)
            model = None
            for mask in MASKS:
                shared.cell = f"seed{seed}/{mode}/mask{mask:02d}"
                shared.tick("verifying_condition")
                record = shared.verify_condition(seed, mode, mask, baseline)
                if record is None:
                    if completed is not None:
                        raise ValueError("Completed report is missing a subset")
                    if (
                        max_conditions is not None
                        and shared.new_conditions >= max_conditions
                    ):
                        raise InterruptedError("Configured condition limit reached")
                    if mask not in (0, 31) and model is None:
                        weights = load(shared.sources.checkpoints[seed, mode])["model"]
                        model = shared.sources.models.make_model(weights)
                    shared.evaluate_condition(seed, mode, mask, model, baseline)
                    record = shared.verify_condition(seed, mode, mask, baseline)
                if record is None:
                    raise RuntimeError("Subset was not committed")
                evaluations.append(record)
            report(
                f"[green]seed{seed}/{mode}: all 32 correction subsets complete[/green]"
            )
        shared.state_cache, shared.cached_frontend = None, None
    shared.tick("writing_reports")
    shared.sources.verify_unchanged()
    verify_files(ROOT, shared.manifest["sources"])
    if completed is None:
        summarize(shared, evaluations)
    shared.heartbeat("complete")
