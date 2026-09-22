"""Fixed paired budgets and separate Sequential output directory."""

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_independent.protocol import Config as IndependentConfig
from experiments.grouped_robot_independent.protocol import source_hashes as upstream
from experiments.grouped_robot_pilot.protocol import read_json

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_sequential"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4_head5_five_seeds"
CONDITIONS = (
    ("sequential", "measured", 0),
    ("sequential", "correct_all_five", 31),
    ("independent", "measured", 0),
    ("independent", "correct_all_five", 31),
    ("no_feedback", "zero", 0),
)


@dataclass(frozen=True)
class Config:
    reference: str = str(
        ROOT / "outputs/grouped_robot_independent/robot_v5_l4_head5_five_seeds"
    )
    seeds: str = "0,1,2,3,4"
    head_epochs: int = 300
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
        source = IndependentConfig(**read_json(Path(self.reference) / "config.json"))
        pilot = source.source()
        if self.head_epochs != source.head_epochs:
            raise ValueError("Sequential and reference label epoch budgets must match")
        if min(self.head_epochs, self.checkpoint_steps) < 1:
            raise ValueError("Epoch and checkpoint budgets must be positive")
        if not set(self.seed_values).issubset(source.seed_values):
            raise ValueError("Every Sequential seed requires a matched reference seed")
        if not self.development and (
            source.development
            or self.seed_values != (0, 1, 2, 3, 4)
            or self.head_epochs != 300
        ):
            raise ValueError(
                "Use --development for nonstandard reference/seeds/budgets"
            )
        return source, pilot


def check_output(config: Config, output: Path) -> None:
    source, pilot = config.sources()
    output = output.resolve()
    for name in (
        config.reference,
        source.reference,
        source.concept_reference,
        source.label_reference,
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
        raise ValueError("Use a child of outputs/grouped_robot_sequential")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}
