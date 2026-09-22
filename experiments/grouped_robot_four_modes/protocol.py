"""Complete the four Robot modes with paired fresh, full-circuit trainings."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_sequential.protocol import (
    CONDITIONS as REUSED_CONDITIONS,
)
from experiments.grouped_robot_sequential.protocol import Config as SequentialConfig
from experiments.grouped_robot_sequential.protocol import source_hashes as upstream

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_four_modes"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4_head5_ep600_five_seeds"
CELLS = ("standard", "joint", "joint_no_feedback")
NEW_CONDITIONS = (
    ("standard", "measured", 0),
    ("joint", "measured", 0),
    ("joint", "correct_all_five", 31),
    ("joint_no_feedback", "zero", 0),
)
CONDITIONS = NEW_CONDITIONS + REUSED_CONDITIONS


@dataclass(frozen=True)
class Config:
    reference: str = str(
        ROOT / "outputs/grouped_robot_sequential/robot_v5_l4_head5_five_seeds"
    )
    seeds: str = "0,1,2,3,4"
    epochs: int = 600
    concept_weight: float = 1.0
    label_weight: float = 1.0
    checkpoint_steps: int = 25
    development: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def seed_values(self) -> tuple[int, ...]:
        values = tuple(int(v) for v in self.seeds.split(","))
        if not values or min(values) < 0 or len(set(values)) != len(values):
            raise ValueError("Seeds must be distinct nonnegative integers")
        return values

    def sources(self):
        sequential = SequentialConfig(**read_json(Path(self.reference) / "config.json"))
        independent, pilot = sequential.sources()
        if self.epochs != independent.concept_epochs + independent.head_epochs:
            raise ValueError("Full-circuit epochs must match concept + label budgets")
        if min(self.epochs, self.checkpoint_steps) < 1:
            raise ValueError("Epoch and checkpoint budgets must be positive")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (self.concept_weight, self.label_weight)
        ):
            raise ValueError("Loss weights must be finite and positive")
        if not set(self.seed_values).issubset(sequential.seed_values):
            raise ValueError("Every seed requires matched completed source models")
        if not self.development and (
            sequential.development
            or self.seed_values != (0, 1, 2, 3, 4)
            or self.epochs != 600
            or (self.concept_weight, self.label_weight) != (1.0, 1.0)
        ):
            raise ValueError("Use --development for nonstandard seeds/budgets/weights")
        return sequential, independent, pilot


def check_output(config: Config, output: Path) -> None:
    sequential, independent, pilot = config.sources()
    output = output.resolve()
    for name in (
        config.reference,
        sequential.reference,
        independent.reference,
        independent.concept_reference,
        independent.label_reference,
    ):
        root = Path(name).resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Output overlaps a historical reference")
    for root in (
        ROOT,
        ROOT / "data",
        ROOT / "experiments",
        Path(pilot.dataset).resolve(),
    ):
        if (
            output == root
            or root.is_relative_to(output)
            or (root != ROOT and output.is_relative_to(root))
        ):
            raise ValueError("Output overlaps a protected source/data root")
    if output.is_relative_to(ROOT / "outputs") and (
        output == OUTPUT_ROOT or not output.is_relative_to(OUTPUT_ROOT)
    ):
        raise ValueError("Use a child of outputs/grouped_robot_four_modes")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}
