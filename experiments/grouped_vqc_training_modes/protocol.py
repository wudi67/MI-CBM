"""Shared initialization, data and provenance for paired training routes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.data import load_cached_data, prepare_data
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
from experiments.grouped_dynamic_vqc.runtime import (
    source_hashes as shared_source_hashes,
)
from experiments.grouped_dynamic_vqc.train import TrainConfig

ROUTES = ("joint", "sequential")
PACKAGE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig(TrainConfig):
    """Both routes have the same total minibatch-update budget."""

    epochs: int = 200
    concept_epochs: int = 100

    def validate(self) -> None:
        super().validate()
        if not 0 < self.concept_epochs < self.epochs:
            raise ValueError("Require 0 < concept_epochs < epochs")


def source_hashes() -> dict[str, str]:
    result = shared_source_hashes()
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    result.update({str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)})
    return result


def state_hash(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor content, independently of torch.save container metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(array_hash(tensor.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def epoch_order(count: int, seed: int, epoch: int) -> torch.Tensor:
    """Epoch-local RNG makes order independent of route, evaluation and resume."""
    generator = torch.Generator().manual_seed((seed + 104729 * epoch + 7813) % 2**63)
    return torch.randperm(count, generator=generator)


class SharedExperiment:
    """One immutable set of inputs and initial weights for both routes."""

    def __init__(self, config: ExperimentConfig, output: Path, resume: bool) -> None:
        config.validate()
        self.config = config
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.runtime = cuda_runtime(config.seed)
        sources = source_hashes()
        if resume:
            manifest = json.loads((output / "manifest.json").read_text())
            if manifest["config"] != asdict(config) or manifest["sources"] != sources:
                raise ValueError("Resume requires identical config and source hashes")
            if manifest["runtime"] != self.runtime:
                raise ValueError("Resume requires the original CUDA/software runtime")
            for name, key in (
                ("preprocessing.json", "preprocessing_sha256"),
                ("initialization.pt", "initialization_sha256"),
            ):
                if sha256(output / name) != manifest[key]:
                    raise ValueError(f"Shared artifact changed: {name}")
            self.data, self.audit = load_cached_data(output)
            self.initial = torch.load(
                output / "initialization.pt", weights_only=False, map_location="cpu"
            )
        else:
            if (output / "config.json").exists() or (output / "manifest.json").exists():
                raise FileExistsError("Experiment exists; use --resume or a new --out")
            model = GroupedDynamicVQC(config.front_layers, config.label_layers).cuda()
            self.initial = {
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "rng": rng_state(),
            }
            del model
            self.data, self.audit = prepare_data(
                Path(config.dataset),
                Path(config.admission),
                output,
                train_limit=config.train_limit,
                val_limit=config.val_limit,
                seed=config.seed,
            )
            atomic_checkpoint(output / "initialization.pt", self.initial)
            manifest = {
                "schema": "grouped_vqc_training_modes.v1",
                "created_at": utc_now(),
                "config": asdict(config),
                "sources": sources,
                "runtime": self.runtime,
                "preprocessing_sha256": sha256(output / "preprocessing.json"),
                "initialization_sha256": sha256(output / "initialization.pt"),
                "initial_model_sha256": state_hash(self.initial["model"]),
                "routes": list(ROUTES),
                "pairing": (
                    "identical initial tensors, cached data "
                    "and epoch-local permutations"
                ),
                "joint": "concept NLL + label BCE for all epochs",
                "sequential": (
                    "concept-only frontend, then frozen full frontend "
                    "and fresh label-only Adam"
                ),
                "label_training_input": (
                    "predicted measurement branches with retained quantum states "
                    "and measured controls"
                ),
                "architecture_boundary": (
                    "input-dependent residual quantum pathway; "
                    "not original Independent CBM"
                ),
                "evidence_role": (
                    "fixed-endpoint development validation; no test evaluation"
                ),
                "budget": (
                    "equal total batch updates; frontend/head updates "
                    "and runtime differ by design"
                ),
            }
            atomic_json(output / "config.json", asdict(config))
            atomic_json(output / "manifest.json", manifest)
        self.manifest = manifest
        self.manifest_hash = sha256(output / "manifest.json")
        self.data = {
            role: {key: value.cuda() for key, value in values.items()}
            for role, values in self.data.items()
        }
