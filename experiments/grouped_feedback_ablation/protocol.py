"""Fixed data, independently seeded frontends, and read-only historical reuse."""

from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch

from experiments.grouped_control_diagnostics.protocol import Config as OldConfig
from experiments.grouped_control_diagnostics.protocol import Experiment as OldExperiment
from experiments.grouped_control_diagnostics.protocol import pin_reference
from experiments.grouped_control_diagnostics.protocol import source_hashes as upstream
from experiments.grouped_control_diagnostics.runner import verify_complete as verify_old
from experiments.grouped_dynamic_vqc.data import load_cached_data, stratified_indices
from experiments.grouped_dynamic_vqc.model import GroupedDynamicVQC
from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    array_hash,
    atomic_checkpoint,
    atomic_json,
    cuda_runtime,
    rng_state,
    sha256,
    utc_now,
)
from experiments.grouped_mlp_controls.protocol import read_json
from experiments.grouped_vqc_training_modes.protocol import state_hash

PACKAGE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/grouped_feedback_ablation/dsprites_l4_paired"


@dataclass(frozen=True)
class Config:
    reference: str = str(ROOT / "outputs/grouped_vqc_training_modes/dsprites_l4_seed0")
    zero_reference: str = str(
        ROOT / "outputs/grouped_control_diagnostics/dsprites_l4_seed0"
    )
    seeds: str = "0,1,2,3,4"
    joint_seed: int = 0
    concept_epochs: int = 100
    head_epochs: int = 100
    joint_epochs: int = 200
    batch_size: int = 1024
    eval_batch_size: int = 2048
    learning_rate: float = 0.01
    concept_weight: float = 1.0
    label_weight: float = 1.0
    grad_clip: float = 5.0
    checkpoint_steps: int = 25
    shots: int = 256
    train_limit: int = 0
    val_limit: int = 0
    retrain_reference: bool = False

    def seed_list(self) -> list[int]:
        return [int(value) for value in self.seeds.split(",")]

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        seeds = self.seed_list()
        if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
            raise ValueError("Provide distinct nonnegative Sequential seeds")
        if (
            self.joint_seed < 0
            or min(
                self.concept_epochs,
                self.head_epochs,
                self.joint_epochs,
                self.batch_size,
                self.eval_batch_size,
                self.checkpoint_steps,
                self.shots,
            )
            < 1
        ):
            raise ValueError("Positive budgets and a nonnegative Joint seed required")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (
                self.learning_rate,
                self.concept_weight,
                self.label_weight,
                self.grad_clip,
            )
        ):
            raise ValueError("Learning rate, loss weights and clip must be positive")
        if any(v < 0 or 0 < v < 18 for v in (self.train_limit, self.val_limit)):
            raise ValueError("Subset sizes must be zero or at least 18")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def cell_name(mode: str, seed: int, variant: str) -> str:
    return f"{mode}/seed{seed}/{variant}"


def check_output(config: Config, output: Path) -> None:
    """Reject overlapping paths before even creating the worker lock."""
    output = output.resolve()
    for reference in (config.reference, config.zero_reference):
        path = Path(reference).resolve()
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("Outputs must be isolated from historical references")


def cells(config: Config, stage: str = "all") -> list[str]:
    result = []
    if stage in {"all", "joint"}:
        result.extend(
            cell_name("joint", config.joint_seed, v)
            for v in ("feedback", "no_feedback")
        )
    if stage in {"all", "sequential"}:
        for seed in config.seed_list():
            result.extend(
                cell_name("sequential", seed, v)
                for v in ("concept", "feedback", "no_feedback")
            )
    if not result:
        raise ValueError(f"Unknown/empty stage: {stage}")
    return result


def cell_spec(config: Config, cell: str) -> dict:
    mode, seed_text, variant = cell.split("/")
    if cell not in cells(config):
        raise ValueError(f"Unknown cell: {cell}")
    phase = (
        "joint" if mode == "joint" else ("concept" if variant == "concept" else "label")
    )
    epochs = {
        "joint": config.joint_epochs,
        "concept": config.concept_epochs,
        "label": config.head_epochs,
    }[phase]
    return {
        "mode": mode,
        "seed": int(seed_text.removeprefix("seed")),
        "variant": variant,
        "phase": phase,
        "epochs": epochs,
        "offset": config.concept_epochs if phase == "label" else 0,
        "control_mode": "zero" if variant == "no_feedback" else "measured",
    }


class Experiment:
    """One manifest fixes the seed list; --stage changes execution, not the protocol."""

    def __init__(self, config: Config, output: Path, resume: bool = False) -> None:
        config.validate()
        self.config, self.output = config, output.resolve()
        self.manifest: dict
        self.reference: dict
        check_output(config, self.output)
        self.runtime = cuda_runtime(0)
        self.stop_requested = False
        self.state_cache: dict[str, torch.Tensor] = {}
        self.cache_frontend_hash: str | None = None
        output.mkdir(parents=True, exist_ok=True)
        if resume:
            self.manifest = read_json(output / "manifest.json")
            if self.manifest["config"] != config.to_dict() or (
                self.manifest["sources"] != source_hashes()
            ):
                raise ValueError("Resume requires identical config and source hashes")
            if self.manifest["runtime"] != self.runtime:
                raise ValueError("Resume requires the original CUDA/software runtime")
            for name, digest in self.manifest["artifacts"].items():
                if sha256(output / name) != digest:
                    raise ValueError(f"Shared artifact changed: {name}")
            self.reference = read_json(output / "reference_lock.json")
            for name, digest in self.reference["artifacts"].items():
                if sha256(Path(name)) != digest:
                    raise ValueError(f"Historical artifact changed: {name}")
        else:
            if (output / "manifest.json").exists() or (output / "config.json").exists():
                raise FileExistsError("Run exists; use --resume or a fresh --out")
            self.reference = pin_reference(OldConfig(reference=config.reference))
            for name in ("data.pt", "preprocessing.json"):
                shutil.copyfile(Path(config.reference) / name, output / name)
        data, self.audit = load_cached_data(output)
        if any(
            self.audit[k]
            for k in (
                "test_preprocessed",
                "test_evaluated",
                "cross_split_raster_group_overlap",
            )
        ):
            raise ValueError("Expected a split-safe validation-only cache")
        self.data = {}
        # Subset selection is fixed across seeds; seed only varies model training.
        for role, limit in (("train", config.train_limit), ("val", config.val_limit)):
            if limit > len(data[role]["angles"]):
                raise ValueError("Subset exceeds the reference cache")
            indices = stratified_indices(data[role]["concepts"].numpy(), limit, 0)
            self.data[role] = {k: v[indices].cuda() for k, v in data[role].items()}
        self.old = self.reference["config"]
        if not resume:
            self.pin_zero_reference(data)
            atomic_json(output / "reference_lock.json", self.reference)
            artifact_names = ["data.pt", "preprocessing.json", "reference_lock.json"]
            for seed in sorted(set([config.joint_seed, *config.seed_list()])):
                path = output / "initializations" / f"seed{seed}.pt"
                if seed == self.old["seed"]:
                    initial = load_checkpoint(
                        Path(config.reference) / "initialization.pt"
                    )
                else:
                    cuda_runtime(seed)
                    model = GroupedDynamicVQC(4, 1).cuda()
                    initial = {
                        "model": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        "rng": rng_state(),
                    }
                    del model
                atomic_checkpoint(path, initial)
                artifact_names.append(str(path.relative_to(output)))
            self.manifest = {
                "schema": "grouped_feedback_ablation.v1",
                "created_at": utc_now(),
                "config": config.to_dict(),
                "sources": source_hashes(),
                "runtime": self.runtime,
                "artifacts": {n: sha256(output / n) for n in artifact_names},
                "cells": cells(config),
                "reuse": self.reuse_plan(),
                "data_indices": {
                    role: array_hash(d["source_index"].cpu().numpy())
                    for role, d in self.data.items()
                },
                "architecture": "Fusion L4 (240), retained B5 + readout L1 (24)",
                "selection": "fixed final epochs; all planned seeds reported",
                "evidence_role": "development validation; Joint single-seed pilot",
                "training": "exact branch-averaged probability; analytic BCE/NLL",
                "finite_shots": "sample joint (m,y) in each model's own control mode",
                "ablation": "measurement-conditioned X feedback vs identity feedback",
                "test_evaluated": False,
            }
            atomic_json(output / "config.json", config.to_dict())
            atomic_json(output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(output / "manifest.json")

    def pin_zero_reference(self, data: dict) -> None:
        """Verify the completed, separately trained historical zero-control route."""
        root = Path(self.config.zero_reference)
        old_config = OldConfig(**read_json(root / "config.json"))
        manifest = read_json(root / "manifest.json")
        if manifest["config"] != old_config.to_dict():
            raise ValueError("Historical zero config mismatch")
        for name, digest in manifest["sources"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Historical zero source changed: {name}")
        for name, digest in manifest["shared_artifacts"].items():
            if sha256(root / name) != digest:
                raise ValueError(f"Historical zero shared artifact changed: {name}")
        for name in ("data.pt", "initialization.pt", "preprocessing.json"):
            if sha256(root / name) != sha256(Path(self.config.reference) / name):
                raise ValueError(f"Historical zero reference differs: {name}")
        old_front = (
            Path(self.config.reference)
            / "sequential/endpoints"
            / (f"epoch_{self.old['concept_epochs']:04d}.pt")
        )
        source = load_checkpoint(root / "source_frontend.pt")
        if state_hash(source["model"]) != state_hash(
            load_checkpoint(old_front)["model"]
        ):
            raise ValueError("Historical zero source frontend differs")
        proxy = SimpleNamespace(
            output=root,
            config=old_config,
            start_epoch=self.old["concept_epochs"],
            manifest_hash=sha256(root / "manifest.json"),
            data=data,
            initial=load_checkpoint(root / "initialization.pt"),
            source=source,
        )
        verify_old(cast(OldExperiment, proxy), "head_zero")
        names = [
            "config.json",
            "manifest.json",
            "head_zero/result.json",
            *manifest["shared_artifacts"],
            *(
                f"head_zero/{n}"
                for n in read_json(root / "head_zero/result.json")["artifacts"]
            ),
        ]
        self.reference["artifacts"].update(
            {str(root / name): sha256(root / name) for name in names}
        )
        self.reference["zero_config"] = old_config.to_dict()
        self.reference["zero_runtime"] = manifest["runtime"]

    def reuse_plan(self) -> dict[str, str]:
        c, old = self.config, self.old
        if (
            c.retrain_reference
            or c.train_limit
            or c.val_limit
            or any(
                getattr(c, key) != old[key]
                for key in (
                    "batch_size",
                    "learning_rate",
                    "grad_clip",
                    "concept_weight",
                    "label_weight",
                )
            )
            or self.runtime != read_json(Path(c.reference) / "manifest.json")["runtime"]
        ):
            return {}
        result = {}
        for cell in cells(c):
            spec = cell_spec(c, cell)
            if spec["seed"] != old["seed"]:
                continue
            if (
                spec["phase"] == "joint"
                and spec["variant"] == "feedback"
                and (c.joint_epochs == old["epochs"])
            ):
                result[cell] = str(
                    Path(c.reference)
                    / "joint/endpoints"
                    / f"epoch_{old['epochs']:04d}.pt"
                )
            elif (
                spec["phase"] == "concept" and c.concept_epochs == old["concept_epochs"]
            ):
                result[cell] = str(
                    Path(c.reference)
                    / "sequential/endpoints"
                    / f"epoch_{old['concept_epochs']:04d}.pt"
                )
            elif spec["phase"] == "label" and (
                c.concept_epochs == old["concept_epochs"]
                and c.head_epochs == old["epochs"] - old["concept_epochs"]
            ):
                if spec["variant"] == "feedback":
                    result[cell] = str(
                        Path(c.reference)
                        / "sequential/endpoints"
                        / f"epoch_{old['epochs']:04d}.pt"
                    )
                else:
                    zero = self.reference["zero_config"]
                    matches = all(
                        getattr(c, k) == zero[k]
                        for k in (
                            "head_epochs",
                            "batch_size",
                            "learning_rate",
                            "grad_clip",
                            "train_limit",
                            "val_limit",
                        )
                    )
                    if (
                        matches
                        and zero["seed"] == spec["seed"]
                        and self.reference["zero_runtime"] == self.runtime
                    ):
                        result[cell] = str(
                            Path(c.zero_reference) / "head_zero/endpoint.pt"
                        )
        return result

    def initial_for(self, cell: str) -> dict:
        spec = cell_spec(self.config, cell)
        if spec["phase"] == "label":
            path = (
                self.output
                / cell_name("sequential", spec["seed"], "concept")
                / "endpoint.pt"
            )
        else:
            path = self.output / "initializations" / f"seed{spec['seed']}.pt"
        return load_checkpoint(path)

    def make_model(self, weights: dict) -> GroupedDynamicVQC:
        model = GroupedDynamicVQC(4, 1).cuda()
        model.load_state_dict(weights)
        return model

    @torch.no_grad()
    def prepare_states(self, model: GroupedDynamicVQC) -> None:
        digest = state_hash(model.frontend.state_dict())
        if self.cache_frontend_hash == digest:
            return
        self.state_cache.clear()
        for role, data in self.data.items():
            chunks = []
            for start in range(0, len(data["angles"]), self.config.eval_batch_size):
                self.heartbeat("caching_frontend", cell=f"{role} cache", offset=start)
                chunks.append(
                    model.frontend(
                        data["angles"][start : start + self.config.eval_batch_size]
                    )
                )
                if self.stop_requested:
                    raise InterruptedError("Paused while preparing the frozen frontend")
            self.state_cache[role] = torch.cat(chunks)
        self.cache_frontend_hash = digest

    def heartbeat(self, status: str, **details: object) -> None:
        import os  # pylint: disable=import-outside-toplevel

        atomic_json(
            self.output / "heartbeat.json",
            {
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "status": status,
                "cell": "pending",
                "epoch_completed": 0,
                "epochs_total": 0,
                "global_step": 0,
                "offset": 0,
                "device": self.runtime["device"],
                "completed_cells": sum(
                    (self.output / c / "result.json").exists()
                    for c in cells(self.config)
                ),
                "total_cells": len(cells(self.config)),
                **details,
            },
        )
