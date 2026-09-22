"""Five binary concepts and exact retained-state prediction on validation only."""

from __future__ import annotations

import torch

from experiments.grouped_dynamic_vqc.evaluation import label_metrics

from .model import RobotVQC, bits, codes, concept_loss, forward_control
from .protocol import CONCEPTS


def concept_metrics(probabilities: torch.Tensor, concepts: torch.Tensor) -> dict:
    if (
        not torch.isfinite(probabilities).all()
        or probabilities.min() < 0
        or (probabilities.sum(1) - 1).abs().max() > 2e-5
    ):
        raise ValueError("Invalid Born probability distribution")
    marginals = probabilities @ bits(probabilities.device).float()
    predicted = marginals >= 0.5
    truth = concepts.bool()
    matches = predicted == truth
    return {
        "joint_nll": float(concept_loss(probabilities, concepts)),
        "joint_single_shot_probability": float(
            probabilities.gather(1, codes(concepts)[:, None]).mean()
        ),
        "joint_map_accuracy": float(
            (probabilities.argmax(1) == codes(concepts)).float().mean()
        ),
        "all_concepts_accuracy": float(matches.all(1).float().mean()),
        "mean_bit_accuracy": float(matches.float().mean()),
        "per_concept": {
            name: label_metrics(marginals[:, i].clamp(0, 1), concepts[:, i])
            for i, name in enumerate(CONCEPTS)
        },
        "max_normalization_error": float((probabilities.sum(1) - 1).abs().max()),
    }


def metrics(raw: dict) -> dict:
    result = {"concept": concept_metrics(raw["concept_probabilities"], raw["concepts"])}
    if "branch_label_mass" in raw:
        probability = raw["branch_label_mass"].sum(1)
        if (
            not torch.isfinite(probability).all()
            or probability.min() < -1e-6
            or probability.max() > 1 + 2e-5
        ):
            raise ValueError("Invalid label probability")
        result["label"] = label_metrics(probability, raw["labels"])
    return result


@torch.no_grad()
def evaluate(
    model: RobotVQC,
    data: dict,
    batch: int,
    *,
    zero: bool = False,
    mask: int = 0,
    states: torch.Tensor | None = None,
    concept_only: bool = False,
    tick=None,
) -> tuple[dict, dict]:
    model.eval()
    probabilities, masses = [], []
    for start in range(0, len(data["angles"]), batch):
        state = (
            model.frontend(data["angles"][start : start + batch])
            if states is None
            else states[start : start + batch]
        )
        p = state.reshape(-1, 32, 32).abs().square().sum(-1)
        probabilities.append(p.cpu())
        if not concept_only:
            output = forward_control(
                model,
                state,
                data["concepts"][start : start + batch],
                zero=zero,
                mask=mask,
            )
            torch.testing.assert_close(p, output["concept_probs"], atol=2e-6, rtol=2e-6)
            masses.append(output["branch_label_mass"].cpu())
        if tick is not None:
            tick("evaluating", offset=min(start + batch, len(data["angles"])))
    raw = {
        k: data[k].cpu() for k in ("concepts", "labels", "source_index", "robot_ids")
    }
    raw["concept_probabilities"] = torch.cat(probabilities)
    if masses:
        raw["branch_label_mass"] = torch.cat(masses)
    return metrics(raw), raw
