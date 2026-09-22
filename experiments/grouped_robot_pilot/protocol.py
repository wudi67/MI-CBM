"""Fixed seed-zero protocol, isolated output paths and provenance helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_dynamic_vqc.runtime import source_hashes as upstream_hashes
from robot_dataset_schema import ROBOT_CONCEPT_NAMES

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_pilot"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4_seed0"
CONCEPTS = tuple(ROBOT_CONCEPT_NAMES)
CELLS = ("concept", "independent", "no_feedback")
CONDITIONS = (
    (("independent", "measured", 0),)
    + tuple(
        ("independent", f"correct_{name}", 1 << (4 - i))
        for i, name in enumerate(CONCEPTS)
    )
    + (("independent", "correct_all_five", 31), ("no_feedback", "zero", 0))
)


@dataclass(frozen=True)
class Config:
    dataset: str = str(ROOT / "data/robot")
    seed: int = 0
    concept_epochs: int = 100
    head_epochs: int = 100
    batch_size: int = 1024
    eval_batch_size: int = 2048
    learning_rate: float = 0.01
    grad_clip: float = 5.0
    checkpoint_steps: int = 25
    train_limit: int = 0
    val_limit: int = 0
    development: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.seed != 0:
            raise ValueError("This P0 pilot is predeclared for seed 0")
        if (
            min(
                self.concept_epochs,
                self.head_epochs,
                self.batch_size,
                self.eval_batch_size,
                self.checkpoint_steps,
            )
            < 1
        ):
            raise ValueError("Epoch, batch and checkpoint budgets must be positive")
        if any(
            not math.isfinite(v) or v <= 0 for v in (self.learning_rate, self.grad_clip)
        ):
            raise ValueError(
                "Learning rate and gradient clipping must be positive and finite"
            )
        if any(v < 0 or 0 < v < 32 for v in (self.train_limit, self.val_limit)):
            raise ValueError("Development limits must be zero or at least 32")
        if not self.development:
            defaults = Config()
            keys = (
                "concept_epochs",
                "head_epochs",
                "batch_size",
                "eval_batch_size",
                "learning_rate",
                "grad_clip",
                "train_limit",
                "val_limit",
            )
            if any(getattr(self, k) != getattr(defaults, k) for k in keys):
                raise ValueError("Use --development for changes to the fixed P0 budget")


def check_output(config: Config, output: Path) -> None:
    output = output.resolve()
    for protected in (
        ROOT,
        ROOT / "data",
        ROOT / "experiments",
        Path(config.dataset).resolve(),
    ):
        if output == protected or protected.is_relative_to(output):
            raise ValueError("Output must be isolated from data and source roots")
    for protected in (
        ROOT / "data",
        ROOT / "experiments",
        Path(config.dataset).resolve(),
    ):
        if output.is_relative_to(protected):
            raise ValueError("Output must be isolated from data and source roots")
    if output.is_relative_to(ROOT / "outputs") and (
        output == OUTPUT_ROOT or not output.is_relative_to(OUTPUT_ROOT)
    ):
        raise ValueError("Use a new subdirectory under outputs/grouped_robot_pilot")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_files(root: Path, hashes: dict) -> None:
    for name, digest in hashes.items():
        if sha256(root / name) != digest:
            raise ValueError(f"Locked artifact changed: {root / name}")


def source_hashes() -> dict:
    paths = [
        *PACKAGE.rglob("*.py"),
        *(PACKAGE / "scripts").glob("*.sh"),
        ROOT / "robot_dataset_schema.py",
        ROOT / "experiments/grouped_vqc_training_modes/protocol.py",
        *(ROOT / "experiments/dynamic_vqc").rglob("*.py"),
    ]
    return {**upstream_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}
