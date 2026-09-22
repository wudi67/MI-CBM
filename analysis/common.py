"""Shared paths and helpers for the paper-analysis scripts."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

# Result directories written by the experiment pipelines (see README.md).
DSPRITES_TEST = Path("grouped_four_modes/dsprites_l4_five_seeds/test")
ROBOT_TEST = Path("grouped_robot_shots_final/robot_v5_l4_head5_five_seeds/test")
ROBOT_CURVE = Path("grouped_robot_intervention_curve/robot_v5_l4_head5_five_seeds")
SEEDS = range(5)

# Corrected-record condition names used by the two pipelines.
CORRECTED = {"dSprites": "both", "Robot": "correct_all_five"}
TEST_DIR = {"dSprites": DSPRITES_TEST, "Robot": ROBOT_TEST}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: list[float]) -> str:
    return f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}"


def record_bits(n_bits: int = 5):
    """Bit table of all 2^n records, most significant bit first."""
    import torch

    codes = torch.arange(2**n_bits)[:, None]
    return (codes >> torch.arange(n_bits - 1, -1, -1)) & 1


def dsprites_rule(records):
    """dSprites label rule on records: (shape == heart) XOR (scale > 2)."""
    shape, scale = records // 8, records % 8
    return ((shape == 2) ^ (scale > 2)).long()


def robot_rule(records):
    """Robot label rule on records (bit order: head, body, antennae, ears, foot)."""
    head, body, antennae, ears, foot = [(records >> (4 - i)) & 1 for i in range(5)]
    score = 2 * head + 5 * body + 3 * antennae + ears + 4 * foot - 7.5
    return (score > 0).long()


def predicted_concepts(name: str, probabilities):
    """Most probable value of each concept, marginalising the record distribution."""
    import torch

    if name == "dSprites":
        groups = probabilities.reshape(-1, 4, 8)
        return torch.stack([groups.sum(2).argmax(1), groups.sum(1).argmax(1)], 1)
    return ((probabilities @ record_bits().to(probabilities.dtype)) >= 0.5).long()


def load_joint(root: Path, name: str, seed: int, condition: str):
    import torch

    path = root / TEST_DIR[name] / f"seed{seed}" / "independent" / condition / "joint.pt"
    return torch.load(path, map_location="cpu", weights_only=False)
