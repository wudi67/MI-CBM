"""Fixed circuit, three initialization methods, and protected historical outputs."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_label_depth.protocol import Config as DepthConfig
from experiments.grouped_robot_label_depth.protocol import source_hashes as upstream
from experiments.grouped_robot_pilot.protocol import read_json

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/grouped_robot_label_init"
DEFAULT_OUTPUT = OUTPUT_ROOT / "robot_v5_l4_head5"
METHODS = ("uniform", "eft_gaussian", "eft_readout")
INDICES = (0, 1, 2)
CELLS = tuple((method, index) for index in INDICES for method in METHODS)
CONDITIONS = (("measured", 0), ("correct_all_five", 31))
DEPTH, LABEL_QUBITS = 5, 6
PAPER = "https://arxiv.org/abs/2601.10479v2"
AUTHOR_COMMIT = "893698185ea905ffd3d0c89c0470696dcea89dd5"


def cell_name(method: str, index: int) -> str:
    if (method, index) not in CELLS:
        raise ValueError("Unknown initialization method or index")
    return f"{method}/init_{index}"


@dataclass(frozen=True)
class Config:
    reference: str = str(ROOT / "outputs/grouped_robot_label_depth/robot_v5_l4_seed0")
    head_epochs: int = 300
    diagnostic_every: int = 50
    checkpoint_steps: int = 25
    init_kappa: float = 0.1
    development: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sigma(self) -> float:
        return self.init_kappa / (DEPTH * LABEL_QUBITS)

    def sources(self):
        depth = DepthConfig(**read_json(Path(self.reference) / "config.json"))
        label, previous, pilot = depth.sources()
        if min(self.head_epochs, self.diagnostic_every, self.checkpoint_steps) < 1:
            raise ValueError("Budgets and intervals must be positive")
        if not math.isfinite(self.init_kappa) or self.init_kappa <= 0:
            raise ValueError("Initialization kappa must be finite and positive")
        if not self.development and (
            depth.development
            or depth.head_epochs != 300
            or self.head_epochs != 300
            or self.diagnostic_every != 50
        ):
            raise ValueError(
                "Default comparison requires completed 300-epoch depth results"
            )
        return depth, label, previous, pilot


def check_output(config: Config, output: Path) -> None:
    depth, label, previous, pilot = config.sources()
    output = output.resolve()
    for name in (
        config.reference,
        depth.reference,
        label.reference,
        previous.reference,
    ):
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
            raise ValueError("Output overlaps a protected source/data root")
        if root != ROOT and output.is_relative_to(root):
            raise ValueError("Output overlaps a protected source/data root")
    if output.is_relative_to(ROOT / "outputs") and (
        output == OUTPUT_ROOT or not output.is_relative_to(OUTPUT_ROOT)
    ):
        raise ValueError("Use a child of outputs/grouped_robot_label_init")


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}
