"""Pinned sources and fixed budgets for the Standard/Joint extension."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.grouped_control_diagnostics.protocol import Config as StandardConfig
from experiments.grouped_control_diagnostics.runner import (
    verify_complete as verify_standard,
)
from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_feedback_ablation.protocol import load_checkpoint, read_json
from experiments.grouped_feedback_ablation.runner import verify_complete as verify_joint
from experiments.grouped_sequential_intervention.protocol import verify_artifacts
from experiments.grouped_shots_final.data import data_hashes
from experiments.grouped_shots_final.protocol import CONDITIONS as OLD_CONDITIONS
from experiments.grouped_shots_final.protocol import Config as EvaluationConfig
from experiments.grouped_shots_final.protocol import Sources as EvaluationSources
from experiments.grouped_shots_final.protocol import check_output as check_parent_output
from experiments.grouped_shots_final.protocol import source_hashes as upstream_hashes
from experiments.grouped_vqc_training_modes.protocol import state_hash

PACKAGE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/grouped_four_modes/dsprites_l4_five_seeds"
TRAININGS = ("standard", "joint", "joint_no_feedback")
CONDITIONS = (
    OLD_CONDITIONS
    + (("standard", "measured"),)
    + tuple(("joint", mode) for mode in ("measured", "shape", "scale", "both"))
    + (("joint_no_feedback", "zero"),)
)


@dataclass(frozen=True)
class Config(EvaluationConfig):
    standard_reference: str = str(
        ROOT / "outputs/grouped_control_diagnostics/dsprites_l4_seed0"
    )
    formal_reference: str = str(
        ROOT / "outputs/grouped_shots_final/dsprites_l4_five_seeds"
    )
    epochs: int = 200
    batch_size: int = 1024
    learning_rate: float = 0.01
    grad_clip: float = 5.0
    checkpoint_steps: int = 25
    train_limit: int = 0

    def validate(self) -> None:
        super().validate()
        StandardConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            grad_clip=self.grad_clip,
            checkpoint_steps=self.checkpoint_steps,
            train_limit=self.train_limit,
        ).validate()
        if not self.development:
            defaults = Config()
            keys = (
                "epochs",
                "batch_size",
                "learning_rate",
                "grad_clip",
                "train_limit",
                "shots",
                "repeats",
                "sampling_seed",
                "eval_batch_size",
            )
            if any(getattr(self, k) != getattr(defaults, k) for k in keys):
                raise ValueError("Formal run requires fixed historical budgets")


def check_output(config: Config, output: Path) -> None:
    check_parent_output(config, output)
    for name in (config.standard_reference, config.formal_reference):
        protected, resolved = Path(name).resolve(), output.resolve()
        if (
            resolved == protected
            or resolved.is_relative_to(protected)
            or protected.is_relative_to(resolved)
        ):
            raise ValueError("New output must be isolated from all historical sources")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def pin_tree(root: Path, lock_path: Path, hashes: dict) -> None:
    """Verify nested result/data/evaluation locks, including their raw observations."""
    if str(lock_path) in hashes:
        return
    lock = read_json(lock_path)
    verify_artifacts(root, lock["artifacts"])
    hashes[str(lock_path)] = sha256(lock_path)
    for name, digest in lock["artifacts"].items():
        path = root / name
        if path.name.endswith("_lock.json") and "artifacts" in read_json(path):
            pin_tree(path.parent, path, hashes)
        hashes[str(path)] = digest


class Sources(EvaluationSources):
    """Historical outputs are only read; training writes through a separate adapter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.output = DEFAULT_OUTPUT
        self.baseline = self.paired.baseline
        standard = Path(config.standard_reference)
        manifest = read_json(standard / "manifest.json")
        verify_artifacts(standard, manifest["shared_artifacts"])
        verify_artifacts(ROOT, manifest["sources"])
        if manifest["runtime"] != self.runtime:
            raise ValueError("Historical Standard runtime differs")
        old_config = StandardConfig(**manifest["config"])
        if (
            old_config.epochs,
            old_config.batch_size,
            old_config.learning_rate,
            old_config.grad_clip,
            old_config.seed,
        ) != (200, 1024, 0.01, 5.0, 0):
            raise ValueError("Unexpected historical Standard budget")
        initial = load_checkpoint(standard / "initialization.pt")
        if state_hash(initial["model"]) != state_hash(self.initial(0)["model"]):
            raise ValueError("Standard and Joint seed-0 initial weights differ")
        old_data = load_checkpoint(standard / "data.pt")
        for role in ("train", "val"):
            if data_hashes(old_data[role]) != data_hashes(self.baseline.data[role]):
                raise ValueError("Standard and Joint preprocessing/splits differ")
        proxy: Any = SimpleNamespace(
            output=standard,
            config=old_config,
            manifest_hash=sha256(standard / "manifest.json"),
            initial=initial,
            data=self.baseline.data,
        )
        result = verify_standard(proxy, "standard")
        self.pin(
            standard, ["manifest.json", "config.json", *manifest["shared_artifacts"]]
        )
        self.pin(standard / "standard", ["result.json", *result["artifacts"]])
        for variant in ("feedback", "no_feedback"):
            cell = f"joint/seed0/{variant}"
            result = verify_joint(self.baseline, cell)
            self.pin(self.baseline.output / cell, ["result.json", *result["artifacts"]])
        formal = Path(config.formal_reference)
        old = read_json(formal / "manifest.json")
        if old["config"] != EvaluationConfig().to_dict() or old["engineering_subset"]:
            raise ValueError("Expected completed full five-seed evaluation protocol")
        if read_json(formal / "result_lock.json")["manifest_sha256"] != sha256(
            formal / "manifest.json"
        ):
            raise ValueError("Historical evaluation manifest mismatch")
        verify_artifacts(formal, old["artifacts"])
        verify_artifacts(ROOT, old["sources"])
        self.pin(formal, ["manifest.json", *old["artifacts"]])
        pin_tree(formal, formal / "result_lock.json", self.hashes)
        for role in ("validation", "test"):
            for seed in config.seed_list():
                for training, mode in OLD_CONDITIONS:
                    lock = read_json(
                        formal
                        / role
                        / f"seed{seed}"
                        / training
                        / mode
                        / "evaluation_lock.json"
                    )
                    if lock["checkpoint_sha256"] != sha256(
                        super().model_path(seed, training)
                    ) or lock["manifest_sha256"] != sha256(formal / "manifest.json"):
                        raise ValueError(
                            "Historical evaluation model/protocol mismatch"
                        )

    def initial(self, seed: int) -> dict:
        return load_checkpoint(self.baseline.output / f"initializations/seed{seed}.pt")

    def reuse_seed(self, seed: int) -> bool:
        return seed == 0 and not self.config.development

    def model_path(self, seed: int, training: str) -> Path:
        if training not in TRAININGS:
            return super().model_path(seed, training)
        if self.reuse_seed(seed):
            if training == "standard":
                return Path(self.config.standard_reference) / "standard/endpoint.pt"
            variant = "feedback" if training == "joint" else "no_feedback"
            return self.baseline.output / f"joint/seed0/{variant}/endpoint.pt"
        if training == "standard":
            return self.output / f"training/seed{seed}/standard/endpoint.pt"
        variant = "feedback" if training == "joint" else "no_feedback"
        return self.output / f"training/joint/seed{seed}/{variant}/endpoint.pt"
