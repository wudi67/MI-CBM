"""Fixed diagnostic budget, validation-only scope and isolated output paths."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_label_continuation.protocol import Config as SourceConfig
from experiments.grouped_robot_label_continuation.protocol import (
    source_hashes as upstream,
)

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_mlp_diagnostic"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4_seed0"


@dataclass(frozen=True)
class Config:
    reference: str = str(
        ROOT / "outputs/grouped_robot_label_continuation/robot_v5_l4_seed0"
    )
    epochs: int = 300
    seed: int = 0
    batch_size: int = 1024
    learning_rate: float = 0.01
    grad_clip: float = 5.0
    checkpoint_steps: int = 25
    diagnostic_every: int = 25
    development: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if (
            min(
                self.epochs,
                self.batch_size,
                self.checkpoint_steps,
                self.diagnostic_every,
            )
            < 1
        ):
            raise ValueError("Training budgets must be positive")
        if self.seed < 0 or any(
            not math.isfinite(v) or v <= 0 for v in (self.learning_rate, self.grad_clip)
        ):
            raise ValueError("Invalid seed, learning rate or clipping norm")
        if not self.development:
            expected = Config().to_dict()
            for key, value in self.to_dict().items():
                if (
                    key not in {"reference", "checkpoint_steps"}
                    and value != expected[key]
                ):
                    raise ValueError(
                        "Use --development for a nonstandard diagnostic budget"
                    )

    def source_config(self) -> SourceConfig:
        from experiments.grouped_robot_pilot.protocol import read_json

        source = SourceConfig(**read_json(Path(self.reference) / "config.json"))
        previous, _ = source.sources()
        if not self.development and (
            source.development
            or source.head_epochs != 300
            or previous.concept_epochs != 300
        ):
            raise ValueError(
                "Diagnostic requires the completed concept-300/label-300 source"
            )
        return source


def check_output(config: Config, output: Path) -> None:
    output = output.resolve()
    source = config.source_config()
    previous, pilot = source.sources()
    references = (config.reference, source.reference, previous.reference)
    for name in references:
        root = Path(name).resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Output overlaps a historical experiment")
    for root in (
        ROOT,
        ROOT / "data",
        ROOT / "experiments",
        Path(pilot.dataset).resolve(),
    ):
        if output == root or root.is_relative_to(output):
            raise ValueError("Output overlaps source/data roots")
        if root != ROOT and output.is_relative_to(root):
            raise ValueError("Output overlaps source/data roots")
    if output.is_relative_to(ROOT / "outputs") and (
        output == OUTPUT_ROOT or not output.is_relative_to(OUTPUT_ROOT)
    ):
        raise ValueError("Use a child of outputs/grouped_robot_mlp_diagnostic")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}
