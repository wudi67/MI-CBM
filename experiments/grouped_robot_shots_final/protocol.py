"""Fixed shots, immutable model sources and isolated output paths."""

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_four_modes.protocol import CONDITIONS
from experiments.grouped_robot_four_modes.protocol import Config as TrainingConfig
from experiments.grouped_robot_four_modes.protocol import source_hashes as upstream
from experiments.grouped_robot_pilot.protocol import read_json

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_shots_final"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4_head5_five_seeds"
MODES = ("standard", "independent", "sequential", "joint")
SUPERVISED = ("independent", "sequential", "joint")


@dataclass(frozen=True)
class Config:
    reference: str = str(
        ROOT / "outputs/grouped_robot_four_modes/robot_v5_l4_head5_ep600_five_seeds"
    )
    seeds: str = "0,1,2,3,4"
    shots: str = "64,128,256,512,1024"
    repeats: int = 10
    sampling_seed: int = 20270918
    eval_batch_size: int = 2048
    val_limit: int = 0
    development: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def seed_list(self) -> list[int]:
        return [int(x) for x in self.seeds.split(",")]

    def shot_list(self) -> list[int]:
        return [int(x) for x in self.shots.split(",")]

    def source(self) -> TrainingConfig:
        return TrainingConfig(**read_json(Path(self.reference) / "config.json"))

    def validate(self) -> None:
        source = self.source()
        source.sources()
        seeds, shots = self.seed_list(), self.shot_list()
        if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
            raise ValueError("Seeds must be distinct nonnegative integers")
        if not set(seeds).issubset(source.seed_values):
            raise ValueError("Every seed requires an admitted complete source model")
        if not shots or min(shots) < 1 or shots != sorted(set(shots)):
            raise ValueError("Shots must be distinct positive increasing integers")
        if min(self.repeats, self.eval_batch_size) < 1 or self.sampling_seed < 0:
            raise ValueError("Invalid sampling/batch budget")
        if self.val_limit < 0 or 0 < self.val_limit < 32:
            raise ValueError("Development limit must be zero or at least 32")
        if not self.development and (
            source.development
            or seeds != [0, 1, 2, 3, 4]
            or self.val_limit
            or shots != [64, 128, 256, 512, 1024]
            or self.repeats != 10
        ):
            raise ValueError(
                "Use --development for source/seeds/subsets/shot budget changes"
            )


def check_output(config: Config, output: Path) -> None:
    config.validate()
    source = config.source()
    sequential, independent, pilot = source.sources()
    output = output.resolve()
    for name in (
        config.reference,
        source.reference,
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
            raise ValueError("Output overlaps protected data/source roots")
    if output.is_relative_to(ROOT / "outputs") and (
        output == OUTPUT_ROOT or not output.is_relative_to(OUTPUT_ROOT)
    ):
        raise ValueError("Use a child of outputs/grouped_robot_shots_final")


def source_hashes() -> dict:
    paths = [
        *PACKAGE.rglob("*.py"),
        *(PACKAGE / "scripts").glob("*.sh"),
        ROOT / "experiments/grouped_shots_final/evaluation.py",
    ]
    return {**upstream(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}


__all__ = [
    "CONDITIONS",
    "Config",
    "DEFAULT_OUTPUT",
    "MODES",
    "SUPERVISED",
    "check_output",
    "source_hashes",
]
