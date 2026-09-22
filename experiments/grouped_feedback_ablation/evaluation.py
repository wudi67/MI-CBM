"""Evaluate exact and finite-shot predictions with the same explicit controls."""

from __future__ import annotations

import torch

from experiments.grouped_dynamic_vqc.evaluation import concept_metrics, label_metrics
from experiments.grouped_dynamic_vqc.model import (
    GroupedDynamicVQC,
    controls_for,
    sample_shots,
)


def forward_control(model: GroupedDynamicVQC, states: torch.Tensor, mode: str) -> dict:
    if mode not in {"measured", "zero"}:
        raise ValueError("Only measured and zero feedback belong to this ablation")
    return model.from_state(states, controls_for(len(states), states.device, mode))


@torch.no_grad()
def evaluate(
    model: GroupedDynamicVQC,
    data: dict,
    batch_size: int,
    mode: str,
    *,
    shots: int = 0,
    seed: int = 0,
    cached_states: torch.Tensor | None = None,
) -> tuple[dict, dict]:
    model.eval()
    probabilities, labels, shot_labels, shot_correct = [], [], [], []
    generator = torch.Generator(device=data["angles"].device).manual_seed(seed + 937)
    for start in range(0, len(data["angles"]), batch_size):
        end = start + batch_size
        states = (
            model.frontend(data["angles"][start:end])
            if cached_states is None
            else cached_states[start:end]
        )
        out = forward_control(model, states, mode)
        p = states.reshape(-1, 32, 32).abs().square().sum(-1)
        if not torch.allclose(p, out["concept_probs"], atol=2e-6, rtol=2e-6):
            raise RuntimeError(
                "Feedback changed the pre-feedback concept probabilities"
            )
        probabilities.append(p.cpu())
        labels.append(out["label_prob"].cpu())
        if shots:
            measured, sampled_labels = sample_shots(out, shots, generator)
            truth = data["concepts"][start:end]
            codes = truth[:, 0] * 8 + truth[:, 1]
            shot_labels.append(sampled_labels.float().mean(1).cpu())
            shot_correct.append((measured == codes[:, None]).float().mean(1).cpu())
    p, y = torch.cat(probabilities), torch.cat(labels)
    if not torch.isfinite(y).all() or y.min() < -1e-5 or y.max() > 1 + 1e-5:
        raise RuntimeError("Invalid label probability")
    targets = data["labels"].cpu()
    raw = {
        "concept_probabilities": p,
        "label_probabilities": y,
        "source_index": data["source_index"].cpu(),
        "labels": targets,
        "concepts": data["concepts"].cpu(),
    }
    result = {
        "n_samples": len(y),
        "control_mode": mode,
        "test_evaluated": False,
        "concept": concept_metrics(p, raw["concepts"]),
        "label": label_metrics(y, targets),
    }
    if shots:
        raw["shot_label_probabilities"] = torch.cat(shot_labels)
        result["finite_shots"] = {
            "control_mode": mode,
            "shots_per_image": shots,
            "seed": seed + 937,
            "sampling": "joint (m,y); same control as exact evaluation",
            "label": label_metrics(raw["shot_label_probabilities"], targets),
            "joint_correct_concept_frequency": float(torch.cat(shot_correct).mean()),
        }
    return result, raw
