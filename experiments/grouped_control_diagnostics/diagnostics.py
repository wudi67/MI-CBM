"""Exact control interventions, paired transitions and finite-shot readout."""

from __future__ import annotations

from collections.abc import Callable

import torch

from experiments.grouped_dynamic_vqc.evaluation import concept_metrics, label_metrics
from experiments.grouped_dynamic_vqc.model import (
    GroupedDynamicVQC,
    controlled_branches,
    controls_for,
    sample_shots,
)

MODES = ("measured", "shape", "scale", "both", "zero", "random")


def fixed_control_probabilities(
    states: torch.Tensor, amplitudes: torch.Tensor
) -> torch.Tensor:
    """Label probabilities for each constant 5-bit X record, summing physical m.

    This contracts density matrices only inside the retained register. Actual
    measured branches are mixed incoherently; amplitudes across m are never added.
    """
    branches = states.reshape(-1, 32, 32)
    density = branches.transpose(1, 2) @ branches.conj()
    kernel = amplitudes @ amplitudes.conj().T
    codes = torch.arange(32, device=states.device)
    indices = codes[:, None] ^ codes[None, :]
    kernels = kernel[indices[:, :, None], indices[:, None, :]]
    return (density.flatten(1) @ kernels.flatten(1).T).real


def control_output(
    model: GroupedDynamicVQC, states: torch.Tensor, concepts: torch.Tensor, mode: str
) -> dict:
    """Reuse the original differentiable measurement and conditional-X circuit."""
    if mode not in ("measured", "both", "zero"):
        raise ValueError(f"Unknown training control: {mode}")
    controls = controls_for(len(states), states.device, mode, concepts)
    return model.from_state(states, controls)


@torch.no_grad()
def evaluate_control(
    model: GroupedDynamicVQC, data: dict, batch_size: int, mode: str = "measured"
) -> dict:
    model.eval()
    concept, label = [], []
    for start in range(0, len(data["angles"]), batch_size):
        end = start + batch_size
        states = model.frontend(data["angles"][start:end])
        out = control_output(model, states, data["concepts"][start:end], mode)
        concept.append(out["concept_probs"].cpu())
        label.append(out["label_prob"].cpu())
    return {
        "concept": concept_metrics(torch.cat(concept), data["concepts"].cpu()),
        "label": label_metrics(torch.cat(label), data["labels"].cpu()),
        "control_mode": mode,
    }


def paired_metrics(
    probabilities: torch.Tensor, baseline: torch.Tensor, targets: torch.Tensor
) -> dict:
    if len(targets) == 0:
        return {"n_samples": 0}
    predicted, original, truth = probabilities >= 0.5, baseline >= 0.5, targets.bool()
    before, after = original == truth, predicted == truth
    corrected = int((~before & after).sum())
    harmed = int((before & ~after).sum())
    signed_change = (probabilities - baseline) * (targets * 2 - 1)
    return {
        "n_samples": len(targets),
        **label_metrics(probabilities, targets),
        "delta_accuracy": (corrected - harmed) / len(targets),
        "wrong_to_right_count": corrected,
        "right_to_wrong_count": harmed,
        "wrong_to_right_fraction": corrected / len(targets),
        "right_to_wrong_fraction": harmed / len(targets),
        "prediction_flip_fraction": float((predicted != original).float().mean()),
        "mean_abs_probability_change": float((probabilities - baseline).abs().mean()),
        "mean_true_label_probability_change": float(signed_change.mean()),
    }


def leave_one_out_controls(concept_probabilities: torch.Tensor) -> torch.Tensor:
    """Uniform donor image other than self, then donor's full joint Born record."""
    if len(concept_probabilities) < 2:
        raise ValueError("Random control requires at least two images")
    return (concept_probabilities.sum(0, keepdim=True) - concept_probabilities) / (
        len(concept_probabilities) - 1
    )


@torch.no_grad()
def diagnose(
    model: GroupedDynamicVQC,
    data: dict,
    batch_size: int,
    shots: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict, dict]:
    """All conditions share original states and branch weights, with exact averaging."""
    model.eval()
    if not next(model.parameters()).is_cuda or not data["angles"].is_cuda:
        raise RuntimeError("Diagnostics require the actual CUDA model and input")
    kernel = model.label_head.one_amplitudes()
    values: dict[str, list[torch.Tensor]] = {mode: [] for mode in MODES[:-1]}
    concept_outputs, constant_outputs, shot_labels, shot_concepts = [], [], [], []
    generator = torch.Generator(device=data["angles"].device).manual_seed(seed + 937)
    total = len(data["angles"])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        concepts = data["concepts"][start:end]
        states = model.frontend(data["angles"][start:end])
        probabilities = states.reshape(-1, 32, 32).abs().square().sum(-1)
        concept_outputs.append(probabilities.cpu())
        fixed = fixed_control_probabilities(states, kernel)
        constant_outputs.append(fixed.cpu())
        codes = concepts[:, 0] * 8 + concepts[:, 1]
        values["both"].append(fixed.gather(1, codes[:, None]).squeeze(1).cpu())
        values["zero"].append(fixed[:, 0].cpu())
        for mode in ("measured", "shape", "scale"):
            controls = controls_for(len(states), states.device, mode, concepts)
            mass = (
                (controlled_branches(states, controls) @ kernel).abs().square().sum(-1)
            )
            values[mode].append(mass.sum(-1).cpu())
            if mode == "measured":
                out = {"concept_probs": probabilities, "branch_label_mass": mass}
                measured, labels = sample_shots(out, shots, generator)
                shot_labels.append(labels.float().mean(1).cpu())
                shot_concepts.append((measured == codes[:, None]).float().mean(1).cpu())
        if progress:
            progress(end, total)
    concept_probabilities = torch.cat(concept_outputs)
    constants = torch.cat(constant_outputs)
    donor = leave_one_out_controls(concept_probabilities)
    outputs = {mode: torch.cat(chunks) for mode, chunks in values.items()}
    outputs["random"] = (constants * donor).sum(-1)
    if any(
        not torch.isfinite(p).all() or p.min() < -1e-5 or p.max() > 1 + 1e-5
        for p in outputs.values()
    ):
        raise RuntimeError("Invalid intervention probabilities")
    targets, concepts = data["labels"].cpu(), data["concepts"].cpu()
    baseline = outputs["measured"]
    masks = {
        f"shape_{shape}_scale_{scale}": (concepts[:, 0] == shape)
        & (concepts[:, 1] == scale)
        for shape in range(3)
        for scale in range(6)
    }
    margins = (baseline - 0.5).abs()
    for lo, hi in ((0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.50001)):
        masks[f"original_margin_{lo}_{hi}"] = (margins >= lo) & (margins < hi)
    for label in (0, 1):
        masks[f"label_{label}"] = targets == label
    result = {
        "n_samples": total,
        "threshold": 0.5,
        "test_evaluated": False,
        "concept": concept_metrics(concept_probabilities, concepts),
        "controls": {
            mode: paired_metrics(p, baseline, targets) for mode, p in outputs.items()
        },
        "groups": {
            name: {
                mode: paired_metrics(p[mask], baseline[mask], targets[mask])
                for mode, p in outputs.items()
            }
            for name, mask in masks.items()
        },
        "finite_shots": {
            "shots_per_image": shots,
            "seed": seed + 937,
            "label": label_metrics(torch.cat(shot_labels), targets),
            "joint_correct_concept_frequency": float(torch.cat(shot_concepts).mean()),
        },
        "random_control": {
            "method": (
                "exact leave-one-image-out donor joint Born distribution; "
                "all 32 codes retained"
            ),
            "monte_carlo_error": False,
            "mean_code_frequency_error": float(
                (donor.mean(0) - concept_probabilities.mean(0)).abs().max()
            ),
        },
        "semantics": [
            "Replace classical X records; preserve original branches and states.",
            "Random control also breaks record/state correspondence.",
            "Standard true-code replacement is a structural control only.",
            "MAP image correctness does not imply a correct actual measurement record.",
            "Pre-intervention margin bins can contain different images across models.",
        ],
    }
    raw = {
        "source_index": data["source_index"].cpu(),
        "concepts": concepts,
        "labels": targets,
        "concept_probabilities": concept_probabilities,
        "label_probabilities": outputs,
        "constant_control_label_probabilities": constants,
        "random_control_distributions": donor,
        "shot_label_probabilities": torch.cat(shot_labels),
    }
    return result, raw
