"""Pin an existing VQC run and reuse its inputs without refitting preprocessing."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.data import load_cached_data
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
from experiments.grouped_vqc_training_modes.protocol import (
    epoch_order,
    state_hash,
)
from experiments.grouped_vqc_training_modes.protocol import (
    source_hashes as upstream_source_hashes,
)

from .model import TASKS, TinyMLP

PACKAGE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    reference: str = str(ROOT / "outputs/grouped_vqc_training_modes/dsprites_l4_seed0")
    concept_epochs: int = 100
    label_epochs: int = 200
    batch_size: int = 1024
    eval_batch_size: int = 2048
    learning_rates: tuple[float, ...] = (0.001, 0.003, 0.01)
    seed: int = 0
    grad_clip: float = 5.0
    checkpoint_steps: int = 25
    train_limit: int = 0
    val_limit: int = 0

    def validate(self) -> None:
        if (
            min(
                self.concept_epochs,
                self.label_epochs,
                self.batch_size,
                self.eval_batch_size,
                self.checkpoint_steps,
            )
            < 1
            or self.seed < 0
        ):
            raise ValueError("Positive epoch/batch/checkpoint budgets required")
        if not self.learning_rates or len(set(self.learning_rates)) != len(
            self.learning_rates
        ):
            raise ValueError("Provide distinct positive learning rates")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (*self.learning_rates, self.grad_clip)
        ):
            raise ValueError("Learning rates and grad_clip must be finite and positive")
        if any(
            limit < 0 or 0 < limit < 18 for limit in (self.train_limit, self.val_limit)
        ):
            raise ValueError("Subsets must be zero (all rows) or at least 18")

    def to_dict(self) -> dict:
        return {**asdict(self), "learning_rates": list(self.learning_rates)}


def source_hashes() -> dict[str, str]:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {
        **upstream_source_hashes(),
        **{str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)},
    }


def cell_name(task: str, lr: float) -> str:
    return f"{task}/lr_{str(lr).replace('.', 'p')}"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def lock_reference(config: Config) -> dict:
    """Read only: source launcher changes do not rewrite the historical manifest."""
    root = Path(config.reference).resolve()
    manifest = read_json(root / "manifest.json")
    reference_config = read_json(root / "config.json")
    if (
        manifest["config"] != reference_config
        or config.seed != reference_config["seed"]
    ):
        raise ValueError("Reference config mismatch or seed is not paired")
    if config.batch_size != reference_config["batch_size"]:
        raise ValueError("Batch size must match the reference VQC")
    if max(config.concept_epochs, config.label_epochs) > reference_config["epochs"]:
        raise ValueError("Requested order extends beyond the reference history")
    paths = [
        root / name
        for name in (
            "manifest.json",
            "config.json",
            "preprocessing.json",
            "data.pt",
            "initialization.pt",
        )
    ]
    if manifest["preprocessing_sha256"] != sha256(root / "preprocessing.json"):
        raise ValueError("Reference preprocessing changed")
    if manifest["initialization_sha256"] != sha256(root / "initialization.pt"):
        raise ValueError("Reference initial weights changed")
    endpoints = {}
    orders = {}
    for route in ("joint", "sequential"):
        directory = root / route
        result = read_json(directory / "result.json")
        if (
            result["status"] != "complete"
            or result["epoch"] != reference_config["epochs"]
            or result["manifest_sha256"] != sha256(root / "manifest.json")
        ):
            raise ValueError("Reference route is not a complete fixed-budget run")
        for filename, key in (
            ("resume.pt", "resume_sha256"),
            ("history.json", "history_sha256"),
        ):
            if sha256(directory / filename) != result[key]:
                raise ValueError(f"Reference artifact changed: {route}/{filename}")
        history = read_json(directory / "history.json")
        if [row["epoch"] for row in history] != list(
            range(1, reference_config["epochs"] + 1)
        ):
            raise ValueError("Reference epoch history is incomplete")
        orders[route] = [row["order_sha256"] for row in history]
        paths += [
            directory / name for name in ("result.json", "resume.pt", "history.json")
        ]
        for epoch in (reference_config["concept_epochs"], reference_config["epochs"]):
            stem = directory / "endpoints" / f"epoch_{epoch:04d}"
            record = read_json(stem.with_suffix(".json"))
            if (
                sha256(stem.with_suffix(".json"))
                != result["endpoint_records_sha256"][str(epoch)]
                or sha256(stem.with_suffix(".pt")) != record["checkpoint_sha256"]
                or record["manifest_sha256"] != sha256(root / "manifest.json")
                or record["test_evaluated"]
            ):
                raise ValueError("Reference endpoint provenance mismatch")
            if epoch == reference_config["epochs"] and any(
                result[k] != v for k, v in record.items()
            ):
                raise ValueError("Reference result differs from its fixed endpoint")
            endpoints[f"{route}_{epoch}"] = record
            paths += [stem.with_suffix(".json"), stem.with_suffix(".pt")]
    if orders["joint"] != orders["sequential"]:
        raise ValueError("Reference routes did not use identical sample orders")
    # Check the implementations used directly by the classical comparison.
    for name in (
        "experiments/grouped_dynamic_vqc/data.py",
        "experiments/grouped_dynamic_vqc/evaluation.py",
        "experiments/grouped_vqc_training_modes/protocol.py",
    ):
        if sha256(ROOT / name) != manifest["sources"][name]:
            raise ValueError(f"Shared data/metric/order implementation changed: {name}")
    initial = torch.load(
        root / "initialization.pt", map_location="cpu", weights_only=False
    )
    quantum_parameters = {
        prefix: sum(
            value.numel()
            for key, value in initial["model"].items()
            if key.startswith(prefix + ".")
        )
        for prefix in ("frontend", "label_head")
    }
    return {
        "reference": str(root),
        "config": reference_config,
        "artifacts": {str(path): sha256(path) for path in paths},
        "order_hashes": orders["joint"],
        "endpoints": endpoints,
        "quantum_parameters": quantum_parameters,
    }


class Experiment:
    """One shared input cache and one initialization per task across learning rates."""

    def __init__(self, config: Config, output: Path, resume: bool = False) -> None:
        config.validate()
        self.config, self.output = config, output
        self.manifest: dict
        self.reference: dict
        self.initial: dict
        output.mkdir(parents=True, exist_ok=True)
        self.runtime = cuda_runtime(config.seed)
        sources = source_hashes()
        if resume:
            self.manifest = read_json(output / "manifest.json")
            if (
                self.manifest["config"] != config.to_dict()
                or self.manifest["sources"] != sources
            ):
                raise ValueError("Resume requires identical config and source hashes")
            if self.manifest["runtime"] != self.runtime:
                raise ValueError("Resume requires the original CUDA/software runtime")
            for name, expected in self.manifest["shared_artifacts"].items():
                if sha256(output / name) != expected:
                    raise ValueError(f"Shared artifact changed: {name}")
            self.reference = read_json(output / "reference_lock.json")
            for name, expected in self.reference["artifacts"].items():
                if sha256(Path(name)) != expected:
                    raise ValueError(f"Reference artifact changed: {name}")
            self.initial = torch.load(
                output / "initialization.pt", weights_only=False, map_location="cpu"
            )
        else:
            if (output / "config.json").exists():
                raise FileExistsError("Experiment exists; use --resume or a new --out")
            self.reference = lock_reference(config)
            for name in ("data.pt", "preprocessing.json"):
                shutil.copyfile(Path(config.reference) / name, output / name)
            atomic_json(output / "reference_lock.json", self.reference)
            self.initial = {}
            for task in TASKS:
                torch.manual_seed(config.seed)
                torch.cuda.manual_seed_all(config.seed)
                model = TinyMLP(task).cuda()
                self.initial[task] = {
                    "model": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "rng": rng_state(),
                    "model_sha256": state_hash(model.state_dict()),
                }
            atomic_checkpoint(output / "initialization.pt", self.initial)
            self.manifest = {
                "schema": "grouped_mlp_controls.v1",
                "created_at": utc_now(),
                "config": config.to_dict(),
                "sources": sources,
                "runtime": self.runtime,
                "shared_artifacts": {
                    name: sha256(output / name)
                    for name in (
                        "data.pt",
                        "preprocessing.json",
                        "reference_lock.json",
                        "initialization.pt",
                    )
                },
                "architectures": {"concept": [40, 3, 32], "label": [40, 6, 1]},
                "activation": "tanh",
                "bias": True,
                "selection": (
                    "lowest fixed-final validation NLL/BCE within each task; "
                    "tie lower LR"
                ),
                "evidence_role": (
                    "development validation; classical LR selection; no test evaluation"
                ),
                "supervision": (
                    "concept-only MLP and label-only MLP are separate predictors, "
                    "not a CBM"
                ),
            }
            atomic_json(output / "config.json", config.to_dict())
            atomic_json(output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(output / "manifest.json")
        data, self.audit = load_cached_data(output)
        self.full_reference_inputs = config.train_limit == config.val_limit == 0
        self.data = {}
        for role, limit in (("train", config.train_limit), ("val", config.val_limit)):
            if limit > len(data[role]["angles"]):
                raise ValueError("Subset exceeds cached reference rows")
            self.data[role] = {
                key: value[: limit or None].cuda() for key, value in data[role].items()
            }
        count = len(self.data["train"]["angles"])
        self.order_hashes = [
            array_hash(epoch_order(count, config.seed, epoch).numpy())
            for epoch in range(1, max(config.concept_epochs, config.label_epochs) + 1)
        ]
        if (
            self.full_reference_inputs
            and self.order_hashes
            != self.reference["order_hashes"][: len(self.order_hashes)]
        ):
            raise ValueError("Classical epoch orders do not match the VQC")

    def epochs_for(self, task: str) -> int:
        return (
            self.config.concept_epochs
            if task == "concept"
            else self.config.label_epochs
        )
