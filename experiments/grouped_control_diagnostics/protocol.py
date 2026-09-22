"""Immutable reference artifacts and paired Standard/head training budgets."""

from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.data import load_cached_data, stratified_indices
from experiments.grouped_dynamic_vqc.model import GroupedDynamicVQC
from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    array_hash,
    atomic_json,
    cuda_runtime,
    sha256,
    utc_now,
)
from experiments.grouped_mlp_controls.protocol import Config as ReferenceConfig
from experiments.grouped_mlp_controls.protocol import lock_reference, read_json
from experiments.grouped_mlp_controls.protocol import source_hashes as upstream_hashes
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

PACKAGE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/grouped_control_diagnostics/dsprites_l4_seed0"


@dataclass(frozen=True)
class Config:
    reference: str = str(ROOT / "outputs/grouped_vqc_training_modes/dsprites_l4_seed0")
    epochs: int = 200
    head_epochs: int = 100
    batch_size: int = 1024
    eval_batch_size: int = 2048
    learning_rate: float = 0.01
    grad_clip: float = 5.0
    seed: int = 0
    checkpoint_steps: int = 25
    shots: int = 256
    train_limit: int = 0
    val_limit: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if (
            min(
                self.epochs,
                self.head_epochs,
                self.batch_size,
                self.eval_batch_size,
                self.checkpoint_steps,
                self.shots,
            )
            < 1
            or self.seed < 0
        ):
            raise ValueError(
                "Positive epoch/batch/shot budgets and nonnegative seed required"
            )
        if any(
            not math.isfinite(v) or v <= 0 for v in (self.learning_rate, self.grad_clip)
        ):
            raise ValueError(
                "Learning rate and gradient clip must be finite and positive"
            )
        if any(v < 0 or 0 < v < 18 for v in (self.train_limit, self.val_limit)):
            raise ValueError("Subset limits must be zero or at least 18")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def pin_reference(config: Config) -> dict:
    """Use existing provenance checks; also verify the historical quantum sources."""
    root = Path(config.reference).resolve()
    old = read_json(root / "config.json")
    reference = lock_reference(
        ReferenceConfig(
            reference=str(root),
            seed=config.seed,
            batch_size=old["batch_size"],
            concept_epochs=old["concept_epochs"],
            label_epochs=old["epochs"],
        )
    )
    if (old["front_layers"], old["label_layers"], old["label_weight"]) != (4, 1, 1.0):
        raise ValueError("Expected original L4/head-L1 circuit and unit label loss")
    manifest = read_json(root / "manifest.json")
    checked = {}
    for name, expected in manifest["sources"].items():
        path = ROOT / name
        if sha256(path) != expected:
            # The only accepted historical change is the recorded tmux exit fix.
            change_path = root / "tmux_auto_exit_change.json"
            change = read_json(change_path) if change_path.exists() else {}
            if (
                name != "experiments/grouped_vqc_training_modes/operations.py"
                or change.get("changed_file") != name
                or change.get("original_sha256") != expected
                or change.get("current_sha256") != sha256(path)
                or change.get("training_code_changed") is not False
            ):
                raise ValueError(f"Historical source changed: {name}")
            snapshot = ROOT / change["original_source_snapshot"]
            if sha256(snapshot) != expected:
                raise ValueError("Historical launcher snapshot changed")
            reference["artifacts"].update(
                {str(change_path): sha256(change_path), str(snapshot): sha256(snapshot)}
            )
        checked[name] = expected
    reference["historical_source_checks"] = checked
    return reference


def module_hash(weights: dict, prefix: str) -> str:
    return state_hash(
        {
            k.removeprefix(prefix + "."): v
            for k, v in weights.items()
            if k.startswith(prefix + ".")
        }
    )


class Experiment:
    """Copy reference artifacts without rewriting their originals."""

    def __init__(self, config: Config, output: Path, resume: bool = False) -> None:
        config.validate()
        self.config, self.output = config, output
        self.runtime = cuda_runtime(config.seed)
        output.mkdir(parents=True, exist_ok=True)
        if resume:
            self.manifest = read_json(output / "manifest.json")
            if (
                self.manifest["config"] != config.to_dict()
                or self.manifest["sources"] != source_hashes()
            ):
                raise ValueError("Resume requires identical config and source hashes")
            if self.manifest["runtime"] != self.runtime:
                raise ValueError("Resume requires identical CUDA/software runtime")
            for name, digest in self.manifest["shared_artifacts"].items():
                if sha256(output / name) != digest:
                    raise ValueError(f"Shared artifact changed: {name}")
            self.reference = read_json(output / "reference_lock.json")
            for name, digest in self.reference["artifacts"].items():
                if sha256(Path(name)) != digest:
                    raise ValueError(f"Reference artifact changed: {name}")
        else:
            if (output / "config.json").exists() or (output / "manifest.json").exists():
                raise FileExistsError(
                    "Experiment exists; use --resume or a fresh --out"
                )
            self.reference = pin_reference(config)
            old = self.reference["config"]
            copies = {
                name: name
                for name in ("data.pt", "preprocessing.json", "initialization.pt")
            }
            copies.update(
                {
                    "source_frontend.pt": (
                        f"sequential/endpoints/epoch_{old['concept_epochs']:04d}.pt"
                    ),
                    "joint_final.pt": f"joint/endpoints/epoch_{old['epochs']:04d}.pt",
                    "sequential_final.pt": (
                        f"sequential/endpoints/epoch_{old['epochs']:04d}.pt"
                    ),
                }
            )
            for destination, source in copies.items():
                shutil.copyfile(Path(config.reference) / source, output / destination)
            atomic_json(output / "reference_lock.json", self.reference)
            self.manifest = {
                "schema": "grouped_control_diagnostics.v1",
                "created_at": utc_now(),
                "config": config.to_dict(),
                "sources": source_hashes(),
                "runtime": self.runtime,
                "shared_artifacts": {
                    name: sha256(output / name)
                    for name in (*copies, "reference_lock.json")
                },
                "architecture": (
                    "Fusion L4: 10 uploaded, 5 measured, "
                    "5 retained + readout; 264 parameters"
                ),
                "objective": ("Label BCE only; concept NLL is detached monitoring"),
                "random_control": (
                    "Exact average using OTHER validation images; "
                    "joint 32-code Born distributions"
                ),
                "threshold": 0.5,
                "selection": "fixed final epoch; no early stopping or test selection",
                "interpretation": (
                    "True-control head keeps input-dependent B; "
                    "not original Independent CBM"
                ),
                "test_evaluated": False,
            }
            atomic_json(output / "config.json", config.to_dict())
            atomic_json(output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(output / "manifest.json")
        self.initial = self.load("initialization.pt")
        self.source = self.load("source_frontend.pt")
        self.verify_reference_states()
        old = self.reference["config"]
        self.start_epoch = old["concept_epochs"]
        self.reuse_measured_head = (
            config.train_limit == 0
            and config.head_epochs == old["epochs"] - self.start_epoch
            and config.batch_size == old["batch_size"]
            and config.learning_rate == old["learning_rate"]
            and config.grad_clip == old["grad_clip"]
            and self.runtime
            == read_json(Path(config.reference) / "manifest.json")["runtime"]
        )
        self.standard_paired = (
            config.train_limit == 0
            and config.val_limit == 0
            and config.epochs == old["epochs"]
            and config.batch_size == old["batch_size"]
            and config.learning_rate == old["learning_rate"]
            and config.grad_clip == old["grad_clip"]
            and self.runtime
            == read_json(Path(config.reference) / "manifest.json")["runtime"]
        )
        self.routes = ("standard", "head_true", "head_zero") + (
            () if self.reuse_measured_head else ("head_measured",)
        )
        data, self.audit = load_cached_data(output)
        if (
            self.audit["test_preprocessed"]
            or self.audit["test_evaluated"]
            or self.audit["cross_split_raster_group_overlap"]
        ):
            raise ValueError("Expected preserved validation-only cache")
        self.data = {}
        for role, limit in (("train", config.train_limit), ("val", config.val_limit)):
            if limit > len(data[role]["angles"]):
                raise ValueError("Subset exceeds reference cache")
            indices = stratified_indices(
                data[role]["concepts"].numpy(), limit, config.seed
            )
            self.data[role] = {k: v[indices].cuda() for k, v in data[role].items()}
        self.pairing = {
            "standard_matches_historical_budget": self.standard_paired,
            "measured_head_reused": self.reuse_measured_head,
            "source_frontend_sha256": module_hash(self.source["model"], "frontend"),
            "initial_head_sha256": module_hash(self.initial["model"], "label_head"),
            "train_source_indices_sha256": array_hash(
                self.data["train"]["source_index"].cpu().numpy()
            ),
            "val_source_indices_sha256": array_hash(
                self.data["val"]["source_index"].cpu().numpy()
            ),
            "routes": list(self.routes),
            "test_evaluated": False,
        }
        for epoch in range(
            1, max(config.epochs, self.start_epoch + config.head_epochs) + 1
        ):
            if config.train_limit == 0 and epoch <= len(self.reference["order_hashes"]):
                actual = array_hash(
                    epoch_order(
                        len(self.data["train"]["angles"]), config.seed, epoch
                    ).numpy()
                )
                if actual != self.reference["order_hashes"][epoch - 1]:
                    raise ValueError("Sample order differs from historical reference")
        if resume and read_json(output / "pairing.json") != self.pairing:
            raise ValueError("Pairing metadata changed")
        atomic_json(output / "pairing.json", self.pairing)

    def load(self, filename: str) -> dict:
        return torch.load(
            self.output / filename, weights_only=False, map_location="cpu"
        )

    def verify_reference_states(self) -> None:
        old = self.reference["config"]
        if (
            state_hash(self.initial["model"])
            != read_json(Path(self.config.reference) / "manifest.json")[
                "initial_model_sha256"
            ]
        ):
            raise ValueError("Reference initial tensors changed")
        for filename, key in (
            ("source_frontend.pt", f"sequential_{old['concept_epochs']}"),
            ("joint_final.pt", f"joint_{old['epochs']}"),
            ("sequential_final.pt", f"sequential_{old['epochs']}"),
        ):
            checkpoint = self.load(filename)
            record = self.reference["endpoints"][key]
            if (
                state_hash(checkpoint["model"]) != record["model_sha256"]
                or checkpoint["progress"]["completed_epoch"] != record["epoch"]
                or checkpoint["manifest_sha256"] != record["manifest_sha256"]
            ):
                raise ValueError(f"Reference tensor/progress mismatch: {key}")
        source, final = self.source, self.load("sequential_final.pt")
        if (
            source["phase"] != "concept"
            or source["progress"]["order"] is not None
            or source["progress"]["offset"] != 0
            or source["module_steps"]["label_head"] != 0
            or module_hash(source["model"], "label_head")
            != module_hash(self.initial["model"], "label_head")
            or module_hash(source["model"], "frontend")
            != module_hash(final["model"], "frontend")
        ):
            raise ValueError("Reference concept/head freeze contract failed")

    def make_model(self, weights: dict) -> GroupedDynamicVQC:
        model = GroupedDynamicVQC(4, 1).cuda()
        model.load_state_dict(weights)
        return model
