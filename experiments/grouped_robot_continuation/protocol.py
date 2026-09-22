"""Two fixed continuation arms and immutable source/optimizer fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.grouped_dynamic_vqc.runtime import ROOT, array_hash, sha256
from experiments.grouped_robot_pilot.protocol import Config as PilotConfig
from experiments.grouped_robot_pilot.protocol import source_hashes as pilot_sources

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_continuation"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4_seed0"
ARMS = ("long_concept", "long_label")
JOBS = (
    ("long_concept", "concept"),
    ("long_concept", "independent"),
    ("long_concept", "no_feedback"),
    ("long_label", "independent"),
    ("long_label", "no_feedback"),
)


@dataclass(frozen=True)
class Config:
    reference: str = str(ROOT / "outputs/grouped_robot_pilot/robot_v5_l4_seed0")
    concept_epochs: int = 300
    head_epochs: int = 300
    development: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self, source: PilotConfig) -> None:
        source.validate()
        if (
            self.concept_epochs <= source.concept_epochs
            or self.head_epochs <= source.head_epochs
        ):
            raise ValueError("Continuation targets must exceed the source epochs")
        if not self.development and (
            source.development or self.concept_epochs != 300 or self.head_epochs != 300
        ):
            raise ValueError(
                "Use --development for nonstandard source or target budgets"
            )


def check_output(config: Config, output: Path, source: PilotConfig) -> None:
    output = output.resolve()
    reference = Path(config.reference).resolve()
    if output.is_relative_to(reference) or reference.is_relative_to(output):
        raise ValueError("Continuation output must be isolated from its reference")
    for protected in (
        ROOT,
        ROOT / "data",
        ROOT / "experiments",
        Path(source.dataset).resolve(),
    ):
        if output == protected or protected.is_relative_to(output):
            raise ValueError("Output overlaps a protected source/data root")
    for protected in (
        ROOT / "data",
        ROOT / "experiments",
        Path(source.dataset).resolve(),
    ):
        if output.is_relative_to(protected):
            raise ValueError("Output overlaps a protected source/data root")
    if output.is_relative_to(ROOT / "outputs") and (
        output == OUTPUT_ROOT or not output.is_relative_to(OUTPUT_ROOT)
    ):
        raise ValueError("Use a child of outputs/grouped_robot_continuation")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**pilot_sources(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def tree_hash(value) -> str:
    """Content hash for full Adam/RNG/progress trees, independent of tensor device."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor")
            digest.update(array_hash(item.detach().cpu().numpy()).encode())
        elif isinstance(item, np.ndarray):
            digest.update(b"numpy")
            digest.update(array_hash(item).encode())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for element in item:
                visit(element)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
        digest.update(b";")

    visit(value)
    return digest.hexdigest()
