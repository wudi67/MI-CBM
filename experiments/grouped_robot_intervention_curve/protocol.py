"""All subsets, fixed endpoints, no selection using test outcomes."""

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_pilot.protocol import CONCEPTS, read_json
from experiments.grouped_robot_shots_final.protocol import Config as ShotsConfig
from experiments.grouped_robot_shots_final.protocol import source_hashes as upstream

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_intervention_curve"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4_head5_five_seeds"
MODES = ("independent", "sequential", "joint")
MASKS = tuple(range(32))


def selected_concepts(mask: int) -> list[str]:
    if mask not in MASKS:
        raise ValueError("Expected a five-bit correction mask")
    return [name for i, name in enumerate(CONCEPTS) if mask & (1 << (4 - i))]


@dataclass(frozen=True)
class Config:
    reference: str = str(
        ROOT / "outputs/grouped_robot_shots_final/robot_v5_l4_head5_five_seeds"
    )
    seeds: str = "0,1,2,3,4"
    eval_batch_size: int = 2048
    limit: int = 0
    development: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def seed_list(self) -> list[int]:
        return [int(v) for v in self.seeds.split(",")]

    def source(self) -> ShotsConfig:
        return ShotsConfig(**read_json(Path(self.reference) / "config.json"))

    @property
    def role(self) -> str:
        return "validation_proxy" if self.development else "test"

    def validate(self) -> None:
        source = self.source()
        source.validate()
        seeds = self.seed_list()
        if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
            raise ValueError("Seeds must be distinct nonnegative integers")
        if not set(seeds).issubset(source.seed_list()):
            raise ValueError("Every seed needs a completed reference model")
        if self.eval_batch_size < 1 or self.limit < 0 or 0 < self.limit < 32:
            raise ValueError("Invalid batch/subset size")
        if not self.development and (
            source.development or seeds != list(range(5)) or self.limit
        ):
            raise ValueError("Formal curves require all five seeds and full test data")


def check_output(config: Config, output: Path) -> None:
    config.validate()
    shots = config.source()
    modes = shots.source()
    sequential, independent, pilot = modes.sources()
    output = output.resolve()
    for name in (
        config.reference,
        shots.reference,
        modes.reference,
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
            raise ValueError("Output overlaps protected source/data roots")
    if output.is_relative_to(ROOT / "outputs") and (
        output == OUTPUT_ROOT or not output.is_relative_to(OUTPUT_ROOT)
    ):
        raise ValueError("Use a child of outputs/grouped_robot_intervention_curve")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}
