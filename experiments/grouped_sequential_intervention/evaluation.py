"""Correct classical controls while retaining all original measurement branches."""

from __future__ import annotations

from collections.abc import Callable

import torch

from experiments.grouped_dynamic_vqc.evaluation import concept_metrics, label_metrics
from experiments.grouped_dynamic_vqc.model import (
    GroupedDynamicVQC,
    controls_for,
    sample_shots,
)

from .protocol import MODES


def forward_control(
    model: GroupedDynamicVQC, states: torch.Tensor, mode: str, concepts: torch.Tensor
) -> dict:
    if mode not in MODES:
        raise ValueError(f"Unknown intervention: {mode}")
    if concepts.shape != (len(states), 2) or concepts.dtype != torch.long:
        raise ValueError("Expected integer Shape/Scale targets [batch, 2]")
    if (
        (concepts < 0).any()
        or (concepts[:, 0] >= 3).any()
        or (concepts[:, 1] >= 6).any()
    ):
        raise ValueError("Targets must be valid Shape (0..2), Scale (0..5) values")
    controls = controls_for(len(states), states.device, mode, concepts)
    return model.from_state(states, controls)


def sampling_seed(seed: int, mode: str) -> int:
    # The measured baseline reproduces the previous evaluator at the same batch size.
    return seed + 937 + 100003 * MODES.index(mode)


@torch.no_grad()
def evaluate(
    model: GroupedDynamicVQC,
    data: dict,
    states: torch.Tensor,
    mode: str,
    *,
    batch_size: int,
    shots: int,
    seed: int,
    progress: Callable[[int], None] | None = None,
) -> tuple[dict, dict]:
    model.eval()
    if len(states) != len(data["labels"]) or min(batch_size, shots) < 1:
        raise ValueError("State/data length mismatch or invalid evaluation budget")
    generator = torch.Generator(device=states.device).manual_seed(
        sampling_seed(seed, mode)
    )
    concepts, exact, finite = [], [], []
    max_branch_error = 0.0
    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        batch_states = states[start:stop]
        out = forward_control(model, batch_states, mode, data["concepts"][start:stop])
        # These are ORIGINAL physical branch probabilities, before record correction.
        p = batch_states.reshape(-1, 32, 32).abs().square().sum(-1)
        error = float((p - out["concept_probs"]).abs().max())
        max_branch_error = max(max_branch_error, error)
        if error > 2e-6 or not torch.allclose(
            p.sum(1), torch.ones(len(p), device=p.device), atol=1e-5, rtol=0
        ):
            raise RuntimeError(
                "Intervention changed measurement weights or state normalization"
            )
        mass = out["branch_label_mass"]
        if (
            not torch.isfinite(mass).all()
            or mass.min() < -1e-6
            or (mass - p).max() > 2e-6
        ):
            raise RuntimeError("Invalid joint (m,y) probability")
        _, sampled_y = sample_shots(out, shots, generator)
        concepts.append(p.cpu())
        exact.append(out["label_prob"].cpu())
        finite.append(sampled_y.float().mean(1).cpu())
        if progress is not None:
            progress(stop)
    raw = {key: data[key].cpu() for key in ("source_index", "concepts", "labels")}
    raw.update(
        concept_probabilities=torch.cat(concepts),
        label_probabilities=torch.cat(exact),
        shot_label_probabilities=torch.cat(finite),
    )
    metrics = {
        "n_samples": len(states),
        "control_mode": mode,
        "test_evaluated": False,
        "concept_metric_scope": (
            "original pre-feedback measurement distribution; "
            "targets are not credited as predictions"
        ),
        "concept": concept_metrics(raw["concept_probabilities"], raw["concepts"]),
        "label": label_metrics(raw["label_probabilities"], raw["labels"]),
        "max_branch_weight_difference": max_branch_error,
        "finite_shots": {
            "control_mode": mode,
            "shots_per_image": shots,
            "seed": sampling_seed(seed, mode),
            "sampling": (
                "joint (original measured m, label y) "
                "under this condition's corrected controls"
            ),
            "label": label_metrics(raw["shot_label_probabilities"], raw["labels"]),
        },
    }
    return metrics, raw


def paired_changes(
    original: torch.Tensor, corrected: torch.Tensor, labels: torch.Tensor
) -> dict:
    before = (original >= 0.5) == labels.bool()
    after = (corrected >= 0.5) == labels.bool()
    gains = int((~before & after).sum())
    losses = int((before & ~after).sum())
    return {
        "wrong_to_right_count": gains,
        "right_to_wrong_count": losses,
        "unchanged_correct_count": int((before & after).sum()),
        "unchanged_wrong_count": int((~before & ~after).sum()),
        "delta_accuracy_pp": 100 * (gains - losses) / len(labels),
    }
