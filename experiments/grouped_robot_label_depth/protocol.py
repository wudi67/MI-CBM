"""Fixed A/B scope, three label initializations, and protected historical outputs."""

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_label_continuation.protocol import Config as SourceConfig
from experiments.grouped_robot_mlp_diagnostic.protocol import source_hashes as upstream
from experiments.grouped_robot_pilot.protocol import read_json

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_label_depth"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4_seed0"
DEPTHS = (1, 5)
INITIALIZATIONS = (0, 1, 2)
CELLS = tuple((depth, index) for index in INITIALIZATIONS for depth in DEPTHS)
CONDITIONS = (("measured", 0), ("correct_foot_shape", 1), ("correct_all_five", 31))


def cell_name(depth: int, index: int) -> str:
    if (depth, index) not in CELLS:
        raise ValueError(
            "Only A/B depths 1/5 and initialization indices 0/1/2 are supported"
        )
    return f"L{depth}/init_{index}"


@dataclass(frozen=True)
class Config:
    reference: str = str(
        ROOT / "outputs/grouped_robot_label_continuation/robot_v5_l4_seed0"
    )
    head_epochs: int = 300
    diagnostic_every: int = 50
    checkpoint_steps: int = 25
    development: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def sources(self):
        source = SourceConfig(**read_json(Path(self.reference) / "config.json"))
        previous, pilot = source.sources()
        if min(self.head_epochs, self.diagnostic_every, self.checkpoint_steps) < 1:
            raise ValueError("Budgets and intervals must be positive")
        if not self.development and (
            source.development
            or previous.concept_epochs != 300
            or source.head_epochs != 300
            or self.head_epochs != 300
            or self.diagnostic_every != 50
        ):
            raise ValueError(
                "Default protocol requires a concept-300/label-300 source "
                "and 300 label epochs"
            )
        return source, previous, pilot


def check_output(config: Config, output: Path) -> None:
    source, previous, pilot = config.sources()
    output = output.resolve()
    for name in (config.reference, source.reference, previous.reference):
        reference = Path(name).resolve()
        if output.is_relative_to(reference) or reference.is_relative_to(output):
            raise ValueError("Output overlaps a historical experiment")
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
        raise ValueError("Use a child of outputs/grouped_robot_label_depth")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}
