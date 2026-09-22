"""Predeclared seed-zero oracle and frontend depth protocols."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..runtime import ROOT, RichArgumentParser

PACKAGE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/dynamic_vqc/robot"


@dataclass(frozen=True)
class Config:
    dataset: str = "data/robot"
    seed: int = 0
    device: str = "cuda:0"
    head_depth: int = 1
    head_learning_rates: tuple[float, ...] = (0.01, 0.03)
    head_steps: int = 500
    head_eval_every: int = 10
    oracle_gap: float = 0.01
    depths: tuple[int, ...] = (4, 8, 16)
    epochs: int = 300
    learning_rate: float = 0.01
    batch_size: int = 512
    eval_batch_size: int = 2048
    checkpoint_every: int = 5
    direction: str = "fixed"
    persistent_feedback: bool = False
    smoke: bool = False

    def validate(self) -> None:
        if self.seed != 0 or not self.device.startswith("cuda"):
            raise ValueError("This pilot requires seed0 and CUDA")
        if self.head_depth not in (1, 2):
            raise ValueError("Use one head layer; two layers are an optional follow-up")
        if not self.depths or len(set(self.depths)) != len(self.depths):
            raise ValueError("Depths must be nonempty and unique")
        if (
            min(
                *self.depths,
                self.head_steps,
                self.head_eval_every,
                self.epochs,
                self.batch_size,
                self.eval_batch_size,
                self.checkpoint_every,
            )
            < 1
        ):
            raise ValueError("Depths, budgets and intervals must be positive")
        rates = (*self.head_learning_rates, self.learning_rate)
        if (
            not self.head_learning_rates
            or len(set(self.head_learning_rates)) != len(self.head_learning_rates)
            or any(not math.isfinite(rate) or rate <= 0 for rate in rates)
        ):
            raise ValueError("Learning rates must be finite, positive and unique")
        if not 0 <= self.oracle_gap < 0.1:
            raise ValueError("Invalid E0 development accuracy gap")
        if self.direction not in ("fixed", "alternating") or self.persistent_feedback:
            raise ValueError("Unknown direction or discontinued persistent feedback")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


def parser() -> RichArgumentParser:
    result = RichArgumentParser(description="Robot E0 oracle + E1 depth pilot (CUDA)")
    result.add_argument("--config", type=Path, default=PACKAGE / "configs/pilot.json")
    result.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "pilot_seed0")
    result.add_argument("--stage", choices=("all", "e0", "e1"), default="all")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--plan-only", action="store_true")
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--smoke", action="store_true", default=None)
    result.add_argument("--dataset")
    result.add_argument("--device")
    result.add_argument("--head-depth", type=int)
    result.add_argument("--head-steps", type=int)
    result.add_argument("--head-learning-rates", type=float, nargs="+")
    result.add_argument("--depths", type=int, nargs="+")
    result.add_argument("--epochs", type=int)
    result.add_argument("--batch-size", type=int)
    result.add_argument("--learning-rate", type=float)
    return result


def configure(args) -> Config:
    values = json.loads(args.config.read_text(encoding="utf-8"))
    known = {field.name for field in fields(Config)}
    if set(values) - known:
        raise ValueError(f"Unknown configuration keys: {set(values) - known}")
    for key in known:
        if getattr(args, key, None) is not None:
            values[key] = getattr(args, key)
    for key in ("depths", "head_learning_rates"):
        if key in values:
            values[key] = tuple(values[key])
    if values.get("smoke"):
        values.update(head_steps=3, head_eval_every=1, epochs=2, checkpoint_every=1)
    values["dataset"] = str((ROOT / values.get("dataset", Config.dataset)).resolve())
    config = Config(**values)
    config.validate()
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or output == OUTPUT_ROOT:
        raise ValueError(f"Use an isolated output subdirectory below {OUTPUT_ROOT}")
    if config.smoke and output == OUTPUT_ROOT / "pilot_seed0":
        raise ValueError("--smoke requires a separate --output-dir")
    return config
