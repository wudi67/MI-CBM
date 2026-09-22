"""Pinned Sequential sources, compatible historical Independent reuse, CUDA data."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch

from experiments.grouped_control_diagnostics.protocol import (
    Config as HistoricalConfig,
)
from experiments.grouped_control_diagnostics.protocol import (
    Experiment as HistoricalExperiment,
)
from experiments.grouped_control_diagnostics.protocol import (
    source_hashes as historical_sources,
)
from experiments.grouped_control_diagnostics.runner import (
    verify_complete as verify_historical,
)
from experiments.grouped_dynamic_vqc.data import stratified_indices
from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    array_hash,
    atomic_json,
    sha256,
    utc_now,
)
from experiments.grouped_feedback_ablation.protocol import (
    cell_name,
    load_checkpoint,
    read_json,
)
from experiments.grouped_sequential_intervention.protocol import (
    DEFAULT_OUTPUT as SEQUENTIAL_OUTPUT,
)
from experiments.grouped_sequential_intervention.protocol import (
    MODES,
    open_reference,
    verify_artifacts,
)
from experiments.grouped_sequential_intervention.protocol import (
    Config as EvaluationConfig,
)
from experiments.grouped_sequential_intervention.protocol import (
    check_output as check_upstream_output,
)
from experiments.grouped_sequential_intervention.protocol import (
    source_hashes as upstream_sources,
)
from experiments.grouped_vqc_training_modes.protocol import state_hash

PACKAGE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "outputs/grouped_independent_intervention/dsprites_l4_five_seeds"
)


@dataclass(frozen=True)
class Config:
    reference: str = EvaluationConfig().reference
    sequential_reference: str = str(SEQUENTIAL_OUTPUT)
    true_reference: str = str(
        ROOT / "outputs/grouped_control_diagnostics/dsprites_l4_seed0"
    )
    seeds: str = "0,1,2,3,4"
    head_epochs: int = 100
    batch_size: int = 1024
    eval_batch_size: int = 2048
    learning_rate: float = 0.01
    grad_clip: float = 5.0
    checkpoint_steps: int = 25
    shots: int = 256
    train_limit: int = 0
    val_limit: int = 0
    development: bool = False
    retrain_reference: bool = False

    def seed_list(self) -> list[int]:
        return [int(item) for item in self.seeds.split(",")]

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        self.evaluation_config().validate()
        if min(self.head_epochs, self.batch_size, self.checkpoint_steps) < 1:
            raise ValueError("Positive epochs, batch and checkpoint interval required")
        if any(
            not math.isfinite(v) or v <= 0 for v in (self.learning_rate, self.grad_clip)
        ):
            raise ValueError("Learning rate and clip must be positive and finite")
        if self.train_limit < 0 or 0 < self.train_limit < 18:
            raise ValueError("train_limit must be zero or at least 18")
        if (self.train_limit or self.val_limit) and not self.development:
            raise ValueError("Subsets require --development")

    def evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            reference=self.reference,
            seeds=self.seeds,
            eval_batch_size=self.eval_batch_size,
            shots=self.shots,
            val_limit=self.val_limit,
        )


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {
        **upstream_sources(),
        **{str(p.relative_to(ROOT)): sha256(p) for p in paths},
    }


def check_output(config: Config, output: Path) -> None:
    check_upstream_output(config.evaluation_config(), output)
    output = output.resolve()
    for name in (config.sequential_reference, config.true_reference):
        if not name:
            continue
        path = Path(name).resolve()
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("Output must be isolated from all source experiments")


def pin_files(root: Path, names) -> dict:
    return {
        str((root / name).resolve()): sha256(root / name) for name in sorted(set(names))
    }


def pin_true_reference(shared: Experiment) -> dict:
    """Do not instantiate the old Experiment: its resume rewrites pairing.json."""
    config = shared.config
    if config.retrain_reference or config.development or not shared.paired_training:
        return {"reuse_seed": None, "artifacts": {}}
    root = Path(config.true_reference).resolve()
    old = HistoricalConfig(**read_json(root / "config.json"))
    if old.seed not in config.seed_list():
        return {"reuse_seed": None, "artifacts": {}}
    manifest = read_json(root / "manifest.json")
    if (
        manifest["config"] != old.to_dict()
        or manifest["sources"] != historical_sources()
    ):
        raise ValueError("Historical Independent config/source mismatch")
    if manifest["runtime"] != shared.runtime:
        raise ValueError("Historical Independent runtime mismatch")
    verify_artifacts(root, manifest["shared_artifacts"])
    reference = read_json(root / "reference_lock.json")
    verify_artifacts(Path("/"), reference["artifacts"])
    if any(
        getattr(old, key) != getattr(config, key)
        for key in (
            "head_epochs",
            "batch_size",
            "learning_rate",
            "grad_clip",
            "train_limit",
            "val_limit",
        )
    ):
        return {"reuse_seed": None, "artifacts": {}}
    for name in ("data.pt", "preprocessing.json"):
        if sha256(root / name) != sha256(shared.source.output / name):
            raise ValueError("Historical Independent data/preprocessing differs")
    initial = load_checkpoint(root / "source_frontend.pt")
    if state_hash(initial["model"]) != state_hash(
        shared.initial_for(old.seed)["model"]
    ):
        raise ValueError("Historical Independent initialization differs")
    proxy = SimpleNamespace(
        output=root,
        config=old,
        start_epoch=shared.source.config.concept_epochs,
        manifest_hash=sha256(root / "manifest.json"),
        data=shared.data,
        initial=load_checkpoint(root / "initialization.pt"),
        source=initial,
    )
    result = verify_historical(cast(HistoricalExperiment, proxy), "head_true")
    if result["control_mode"] != "both" or result["module_steps"]["frontend"] != 0:
        raise ValueError(
            "Historical training must use true controls and frozen frontend"
        )
    expected = read_json(
        shared.source.output
        / cell_name("sequential", old.seed, "feedback")
        / "result.json"
    )
    if result["initial_model_sha256"] != expected["initial_model_sha256"]:
        raise ValueError("Historical Independent and Sequential initializations differ")
    names = [
        "config.json",
        "manifest.json",
        "pairing.json",
        *manifest["shared_artifacts"],
        "head_true/result.json",
        *(f"head_true/{name}" for name in result["artifacts"]),
    ]
    return {
        "reuse_seed": old.seed,
        "endpoint": str(root / "head_true/endpoint.pt"),
        "artifacts": {**pin_files(root, names), **reference["artifacts"]},
    }


def pin_sequential_evaluations(shared: Experiment) -> dict:
    root = Path(shared.config.sequential_reference).resolve()
    if not shared.config.sequential_reference or not (root / "manifest.json").exists():
        return {"reuse": False, "artifacts": {}}
    manifest = read_json(root / "manifest.json")
    old = EvaluationConfig(**manifest["config"])
    if Path(old.reference).resolve() != shared.source.output:
        raise ValueError("Sequential evaluations refer to different training models")
    if (
        manifest["sources"] != upstream_sources()
        or manifest["runtime"] != shared.runtime
    ):
        raise ValueError("Sequential evaluation sources/runtime changed")
    verify_artifacts(root, manifest["artifacts"])
    reference = read_json(root / "reference_lock.json")
    verify_artifacts(Path(reference["root"]), reference["artifacts"])
    compatible = (
        not shared.config.val_limit
        and not old.val_limit
        and shared.config.shots == old.shots
        and shared.config.eval_batch_size == old.eval_batch_size
        and set(shared.config.seed_list()).issubset(old.seed_list())
    )
    if not compatible:
        return {"reuse": False, "artifacts": {}}
    names = ["manifest.json", *manifest["artifacts"]]
    for seed in shared.config.seed_list():
        for mode in MODES:
            directory = root / f"seed{seed}" / mode
            lock = read_json(directory / "evaluation_lock.json")
            expected = {
                "manifest_sha256": sha256(root / "manifest.json"),
                "checkpoint_sha256": sha256(shared.model_path("sequential", seed)),
                "seed": seed,
                "control_mode": mode,
            }
            if any(lock[k] != v for k, v in expected.items()):
                raise ValueError("Sequential condition provenance mismatch")
            verify_artifacts(directory, lock["artifacts"])
            names += [
                f"seed{seed}/{mode}/evaluation_lock.json",
                *(f"seed{seed}/{mode}/{n}" for n in lock["artifacts"]),
            ]
    return {"reuse": True, "root": str(root), "artifacts": pin_files(root, names)}


class Experiment:
    def __init__(self, config: Config, output: Path, resume: bool = False) -> None:
        config.validate()
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        self.stop_requested = False
        self.state_cache: dict[str, torch.Tensor] = {}
        self.cache_frontend_hash = None
        self.source, source_lock = open_reference(config.evaluation_config())
        self.runtime = self.source.runtime
        self.paired_training = not config.train_limit and all(
            getattr(config, k) == getattr(self.source.config, k)
            for k in ("head_epochs", "batch_size", "learning_rate", "grad_clip")
        )
        if not self.paired_training and not config.development:
            raise ValueError("Unpaired training budgets require --development")
        self.data = {}
        for role, limit in (("train", config.train_limit), ("val", config.val_limit)):
            data = self.source.data[role]
            if limit > len(data["labels"]):
                raise ValueError("Subset exceeds source data")
            indices = stratified_indices(data["concepts"].cpu().numpy(), limit, 0)
            self.data[role] = {k: v[indices] for k, v in data.items()}
        self.true_reference = pin_true_reference(self)
        self.sequential_reference = pin_sequential_evaluations(self)
        artifacts = {
            str(Path(source_lock["root"]) / n): h
            for n, h in source_lock["artifacts"].items()
        }
        artifacts.update(self.true_reference["artifacts"])
        artifacts.update(self.sequential_reference["artifacts"])
        reference = {
            "artifacts": artifacts,
            "true_reference": self.true_reference,
            "sequential_reference": self.sequential_reference,
        }
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest: dict
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
            if reference != read_json(self.output / "reference_lock.json"):
                raise ValueError("Pinned reference changed")
            if (self.output / "result_lock.json").exists():
                lock = read_json(self.output / "result_lock.json")
                if lock["manifest_sha256"] != sha256(self.output / "manifest.json"):
                    raise ValueError("Summary provenance mismatch")
                verify_artifacts(self.output, lock["artifacts"])
        else:
            if (self.output / "manifest.json").exists() or (
                self.output / "config.json"
            ).exists():
                raise FileExistsError("Run exists; use --resume or a fresh --out")
            atomic_json(self.output / "config.json", config.to_dict())
            atomic_json(self.output / "reference_lock.json", reference)
            self.manifest = {
                "schema": "grouped_independent_intervention.v1",
                "created_at": utc_now(),
                "config": config.to_dict(),
                "sources": source_hashes(),
                "runtime": self.runtime,
                "artifacts": {
                    n: sha256(self.output / n)
                    for n in ("config.json", "reference_lock.json")
                },
                "paired_training_budget": self.paired_training,
                "engineering_subset": bool(
                    config.development
                    or self.source.config.train_limit
                    or self.source.config.val_limit
                ),
                "reuse_true_seed": self.true_reference["reuse_seed"],
                "reuse_sequential_evaluations": self.sequential_reference["reuse"],
                "data_indices": {
                    role: array_hash(data["source_index"].cpu().numpy())
                    for role, data in self.data.items()
                },
                "architecture": (
                    "existing Fusion L4 frontend (240 frozen), "
                    "retained B5 + readout L1 (24 trained)"
                ),
                "training": (
                    "true Shape/Scale classical X records; retained input-dependent "
                    "B states; fresh Adam; label BCE "
                    "of exact branch probability mixture"
                ),
                "selection": "fixed final epoch; no best-validation selection",
                "evaluation": (
                    "same four classical record interventions; exact and joint (m,y) "
                    "shots; measured is normal unassisted prediction"
                ),
                "finite_shots": (
                    "seed + 937 + 100003 * condition_index; evaluation only"
                ),
                "test_evaluated": False,
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")

    def initial_for(self, seed: int) -> dict:
        return self.source.initial_for(cell_name("sequential", seed, "feedback"))

    def model_path(self, training: str, seed: int) -> Path:
        if training == "sequential":
            return (
                self.source.output
                / cell_name("sequential", seed, "feedback")
                / "endpoint.pt"
            )
        return self.output / f"independent/seed{seed}/endpoint.pt"

    def make_model(self, weights: dict):
        return self.source.make_model(weights)

    def spec(self, seed: int) -> dict:
        return {
            "seed": seed,
            "phase": "label",
            "control_mode": "both",
            "epochs": self.config.head_epochs,
            "offset": self.source.config.concept_epochs,
        }

    def heartbeat(self, status: str, **details: object) -> None:
        atomic_json(
            self.output / "heartbeat.json",
            {
                "status": status,
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "device": self.runtime["device"],
                "completed_cells": sum(
                    (self.output / f"independent/seed{s}/result.json").exists()
                    for s in self.config.seed_list()
                ),
                "total_cells": len(self.config.seed_list()),
                "completed_conditions": len(
                    list(self.output.glob("*/seed*/evaluation/*/evaluation_lock.json"))
                ),
                "total_conditions": 8 * len(self.config.seed_list()),
                "epoch_completed": 0,
                "epochs_total": self.config.head_epochs,
                "global_step": 0,
                "offset": 0,
                **details,
            },
        )

    @torch.no_grad()
    def prepare_states(self, model) -> None:
        digest = state_hash(model.frontend.state_dict())
        if self.cache_frontend_hash == digest:
            return
        self.state_cache.clear()
        self.cache_frontend_hash = None
        for role, data in self.data.items():
            chunks = []
            for start in range(0, len(data["labels"]), self.config.eval_batch_size):
                self.heartbeat("caching_frontend", cell=role, offset=start)
                chunks.append(
                    model.frontend(
                        data["angles"][start : start + self.config.eval_batch_size]
                    )
                )
                if self.stop_requested:
                    raise InterruptedError("Paused while caching frozen states")
            self.state_cache[role] = torch.cat(chunks)
        self.cache_frontend_hash = digest

    def verify_sources(self) -> None:
        verify_artifacts(
            Path("/"), read_json(self.output / "reference_lock.json")["artifacts"]
        )
