"""Frozen P0 budget and isolated, hash-bound output protocol."""

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_continuation.protocol import source_hashes as upstream
from experiments.grouped_robot_pilot.protocol import Config as PilotConfig
from experiments.grouped_robot_pilot.protocol import read_json

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_independent"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4_head5_five_seeds"
CELLS = ("concept", "independent", "no_feedback")
CONDITIONS = (
    ("independent", "measured", 0),
    ("independent", "correct_all_five", 31),
    ("no_feedback", "zero", 0),
)
SIGMA = 0.1 / (5 * 6)


@dataclass(frozen=True)
class Config:
    reference: str = str(ROOT / "outputs/grouped_robot_pilot/robot_v5_l4_seed0")
    concept_reference: str = str(
        ROOT / "outputs/grouped_robot_concept_init/robot_v5_l4"
    )
    label_reference: str = str(
        ROOT / "outputs/grouped_robot_label_init/robot_v5_l4_head5"
    )
    seeds: str = "0,1,2,3,4"
    concept_epochs: int = 300
    head_epochs: int = 300
    head_order_offset: int = 100
    checkpoint_steps: int = 25
    fresh_all: bool = False
    train_limit: int = 0
    val_limit: int = 0
    development: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def seed_values(self) -> tuple[int, ...]:
        values = tuple(int(v) for v in self.seeds.split(","))
        if not values or min(values) < 0 or len(set(values)) != len(values):
            raise ValueError("Seeds must be distinct nonnegative integers")
        return values

    @property
    def reuse_seed0(self) -> bool:
        return not self.fresh_all and 0 in self.seed_values

    def source(self) -> PilotConfig:
        pilot = PilotConfig(**read_json(Path(self.reference) / "config.json"))
        pilot.validate()
        if min(self.concept_epochs, self.head_epochs, self.checkpoint_steps) < 1:
            raise ValueError("Epoch and checkpoint budgets must be positive")
        if self.head_order_offset < 0:
            raise ValueError("Sample-order offset must be nonnegative")
        if any(v < 0 or 0 < v < 32 for v in (self.train_limit, self.val_limit)):
            raise ValueError("Development limits must be zero or at least 32")
        if not self.development and (
            pilot.development
            or self.seed_values != (0, 1, 2, 3, 4)
            or (self.concept_epochs, self.head_epochs, self.head_order_offset)
            != (300, 300, 100)
            or self.train_limit
            or self.val_limit
        ):
            raise ValueError("Use --development for nonstandard seeds/data/budgets")
        if self.reuse_seed0 and (
            self.development
            or self.concept_epochs != 300
            or self.head_epochs != 300
            or self.head_order_offset != 100
        ):
            raise ValueError(
                "Historical reuse requires the formal budget; use --fresh-all"
            )
        return pilot


def check_output(config: Config, output: Path) -> None:
    pilot = config.source()
    output = output.resolve()
    for source in (config.reference, config.concept_reference, config.label_reference):
        root = Path(source).resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Output overlaps a historical reference")
    for root in (
        ROOT,
        ROOT / "data",
        ROOT / "experiments",
        Path(pilot.dataset).resolve(),
    ):
        if output == root or root.is_relative_to(output):
            raise ValueError("Output overlaps a protected source/data root")
        if root != ROOT and output.is_relative_to(root):
            raise ValueError("Output overlaps a protected source/data root")
    if output.is_relative_to(ROOT / "outputs") and (
        output == OUTPUT_ROOT or not output.is_relative_to(OUTPUT_ROOT)
    ):
        raise ValueError("Use a child of outputs/grouped_robot_independent")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}
