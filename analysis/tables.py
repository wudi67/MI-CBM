"""Print the numbers of Tables 1 and 3 from the experiment outputs.

Usage: python -m analysis.tables [--outputs outputs]
Table 2 is written by the label-depth experiment itself:
outputs/grouped_robot_label_depth/robot_v5_l4_seed0/summary.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from analysis.common import (
    CORRECTED,
    SEEDS,
    TEST_DIR,
    dsprites_rule,
    load_joint,
    mean_std,
    read_rows,
    robot_rule,
)

REGIMES = ("standard", "joint", "sequential", "independent")


def collect(root: Path, name: str) -> dict:
    values: dict = defaultdict(list)
    for row in read_rows(root / TEST_DIR[name] / "condition_results.csv"):
        if row["shots"] != "0":
            continue
        key = (row["training"], row["control_mode"])
        values[key + ("label",)].append(100 * float(row["label_accuracy"]))
        if row["training"] == "standard":
            continue  # no concept supervision
        if name == "dSprites":
            shape = row["concept_shape_marginal_argmax_accuracy"]
            scale = row["concept_scale_marginal_argmax_accuracy"]
            concept = 50 * (float(shape) + float(scale))
        else:
            concept = 100 * float(row["concept_mean_bit_accuracy"])
        values[key + ("concept",)].append(concept)
    return values


def rule_on_record(root: Path, name: str) -> list[float]:
    """Accuracy of the true rule applied to the Born distribution of the records."""
    import torch

    rule = dsprites_rule if name == "dSprites" else robot_rule
    label_of_record = rule(torch.arange(32)).double()
    accuracies = []
    for seed in SEEDS:
        joint = load_joint(root, name, seed, "measured")
        probabilities = joint["concept_probabilities"].double()
        labels = joint["labels"].long()
        mass_on_one = (probabilities * label_of_record).sum(1)
        accuracies.append(100 * float(((mass_on_one >= 0.5).long() == labels).double().mean()))
    return accuracies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    root = parser.parse_args().outputs

    print("Table 1: test accuracy (%), mean ± std over five seeds")
    print(f"{'Dataset':<9}{'Regime':<13}{'Label':>13}{'Concept':>13}{'Corrected':>13}")
    for name in ("dSprites", "Robot"):
        values = collect(root, name)
        for regime in REGIMES:
            label = mean_std(values[(regime, "measured", "label")])
            concept = corrected = "-"
            if regime != "standard":
                concept = mean_std(values[(regime, "measured", "concept")])
                corrected = mean_std(values[(regime, CORRECTED[name], "label")])
            print(f"{name:<9}{regime:<13}{label:>13}{concept:>13}{corrected:>13}")

    print("\nTable 3: test label accuracy (%) without intervention")
    print(f"{'Dataset':<9}{'Rule on record':>16}{'No feedforward':>16}{'With feedforward':>18}")
    for name in ("dSprites", "Robot"):
        values = collect(root, name)
        rule = mean_std(rule_on_record(root, name))
        no_ff = mean_std(values[("no_feedback", "zero", "label")])
        with_ff = mean_std(values[("independent", "measured", "label")])
        print(f"{name:<9}{rule:>16}{no_ff:>16}{with_ff:>18}")


if __name__ == "__main__":
    main()
