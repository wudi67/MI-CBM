"""Fixed source arm, budgets, paths and provenance for label-only continuation."""

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_continuation.protocol import Config as SourceConfig
from experiments.grouped_robot_continuation.protocol import source_hashes as old_sources
from experiments.grouped_robot_pilot.protocol import Config as PilotConfig
from experiments.grouped_robot_pilot.protocol import read_json

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_label_continuation"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4_seed0"
CELLS = ("independent", "no_feedback")


@dataclass(frozen=True)
class Config:
    reference: str = str(ROOT / "outputs/grouped_robot_continuation/robot_v5_l4_seed0")
    head_epochs: int = 300
    diagnostic_every: int = 50
    development: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def sources(self) -> tuple[SourceConfig, PilotConfig]:
        source = SourceConfig(**read_json(Path(self.reference) / "config.json"))
        pilot = PilotConfig(**read_json(Path(source.reference) / "config.json"))
        source.validate(pilot)
        if self.head_epochs <= pilot.head_epochs or self.diagnostic_every < 1:
            raise ValueError("Target must exceed the source label budget; interval > 0")
        if not self.development and (
            source.development
            or source.concept_epochs != 300
            or pilot.head_epochs != 100
            or self.head_epochs != 300
            or self.diagnostic_every != 50
        ):
            raise ValueError(
                "Formal budget is concept 300, label 100->300, interval 50"
            )
        return source, pilot


def check_output(config: Config, output: Path) -> None:
    source, pilot = config.sources()
    output = output.resolve()
    for reference in (
        Path(config.reference).resolve(),
        Path(source.reference).resolve(),
    ):
        if output.is_relative_to(reference) or reference.is_relative_to(output):
            raise ValueError("Output must be isolated from both reference experiments")
    for protected in (
        ROOT,
        ROOT / "data",
        ROOT / "experiments",
        Path(pilot.dataset).resolve(),
    ):
        if output == protected or protected.is_relative_to(output):
            raise ValueError("Output overlaps a protected source/data root")
        if protected != ROOT and output.is_relative_to(protected):
            raise ValueError("Output overlaps a protected source/data root")
    if output.is_relative_to(ROOT / "outputs") and (
        output == OUTPUT_ROOT or not output.is_relative_to(OUTPUT_ROOT)
    ):
        raise ValueError("Use a child of outputs/grouped_robot_label_continuation")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**old_sources(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}
