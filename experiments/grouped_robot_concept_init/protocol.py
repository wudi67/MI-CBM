"""Two initializations, identical four-layer Fusion circuit and training budget."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_continuation.protocol import source_hashes as upstream
from experiments.grouped_robot_pilot.protocol import Config as PilotConfig
from experiments.grouped_robot_pilot.protocol import read_json

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_concept_init"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4"
METHODS = ("uniform", "eft_gaussian")
INDICES = (0, 1, 2)
CELLS = tuple((method, index) for index in INDICES for method in METHODS)
FRONT_LAYERS, FRONT_QUBITS = 4, 10
PAPER = "https://arxiv.org/abs/2601.10479v2"
AUTHOR_COMMIT = "893698185ea905ffd3d0c89c0470696dcea89dd5"


def cell_name(method: str, index: int) -> str:
    if (method, index) not in CELLS:
        raise ValueError("Unknown initialization method or index")
    return f"{method}/init_{index}"


@dataclass(frozen=True)
class Config:
    reference: str = str(ROOT / "outputs/grouped_robot_pilot/robot_v5_l4_seed0")
    concept_epochs: int = 300
    init_kappa: float = 0.1
    checkpoint_steps: int = 25
    train_limit: int = 0
    val_limit: int = 0
    development: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sigma(self) -> float:
        return self.init_kappa / (FRONT_LAYERS * FRONT_QUBITS)

    def source(self) -> PilotConfig:
        pilot = PilotConfig(**read_json(Path(self.reference) / "config.json"))
        pilot.validate()
        if min(self.concept_epochs, self.checkpoint_steps) < 1:
            raise ValueError("Epoch and checkpoint budgets must be positive")
        if not math.isfinite(self.init_kappa) or self.init_kappa <= 0:
            raise ValueError("Initialization kappa must be finite and positive")
        if any(v < 0 or 0 < v < 32 for v in (self.train_limit, self.val_limit)):
            raise ValueError("Development limits must be zero or at least 32")
        if not self.development and (
            pilot.development
            or self.concept_epochs != 300
            or self.init_kappa != 0.1
            or self.train_limit
            or self.val_limit
        ):
            raise ValueError("Use --development for nonstandard comparison budgets")
        return pilot


def check_output(config: Config, output: Path) -> None:
    pilot = config.source()
    output = output.resolve()
    reference = Path(config.reference).resolve()
    if output.is_relative_to(reference) or reference.is_relative_to(output):
        raise ValueError("Output overlaps the historical reference")
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
        raise ValueError("Use a child of outputs/grouped_robot_concept_init")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}
