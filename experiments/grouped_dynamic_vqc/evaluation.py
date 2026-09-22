"""Validation-only concept, label, finite-shot and control-record diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import GroupedDynamicVQC, controls_for, sample_shots


def label_metrics(probabilities: torch.Tensor, targets: torch.Tensor) -> dict:
    predictions = probabilities >= 0.5
    truth = targets.bool()
    recalls = [
        float((predictions[truth == value] == value).float().mean())
        for value in (False, True)
        if (truth == value).any()
    ]
    return {
        "accuracy": float((predictions == truth).float().mean()),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "bce": float(
            F.binary_cross_entropy(probabilities.clamp(1e-7, 1 - 1e-7), targets.float())
        ),
    }


def concept_metrics(probabilities: torch.Tensor, concepts: torch.Tensor) -> dict:
    correct_codes = concepts[:, 0] * 8 + concepts[:, 1]
    correct = probabilities.gather(1, correct_codes[:, None]).squeeze(1)
    groups = probabilities.reshape(-1, 4, 8)
    shape_prediction = groups.sum(2).argmax(1)
    scale_prediction = groups.sum(1).argmax(1)
    codes = torch.arange(32, device=probabilities.device)
    invalid = (codes // 8 == 3) | (codes % 8 >= 6)
    return {
        "joint_nll": float(-correct.clamp_min(1e-7).log().mean()),
        "joint_single_shot_probability": float(correct.mean()),
        "joint_map_accuracy": float(
            (probabilities.argmax(1) == correct_codes).float().mean()
        ),
        "shape_marginal_argmax_accuracy": float(
            (shape_prediction == concepts[:, 0]).float().mean()
        ),
        "scale_marginal_argmax_accuracy": float(
            (scale_prediction == concepts[:, 1]).float().mean()
        ),
        "group_argmax_exact_accuracy": float(
            (
                (shape_prediction == concepts[:, 0])
                & (scale_prediction == concepts[:, 1])
            )
            .float()
            .mean()
        ),
        "invalid_code_probability": float(probabilities[:, invalid].sum(1).mean()),
        "invalid_joint_map_fraction": float(
            invalid[probabilities.argmax(1)].float().mean()
        ),
        "max_normalization_error": float((probabilities.sum(1) - 1).abs().max()),
    }


@torch.no_grad()
def evaluate(
    model: GroupedDynamicVQC,
    data: dict[str, torch.Tensor],
    batch_size: int,
    *,
    diagnostics: bool = False,
    shots: int = 256,
    seed: int = 0,
) -> dict:
    model.eval()
    concept_outputs = []
    label_outputs = []
    corrected: dict[str, list[torch.Tensor]] = {
        mode: [] for mode in ("shape", "scale", "both", "zero")
    }
    shot_probabilities = []
    shot_concepts = []
    generator = torch.Generator(device=data["angles"].device).manual_seed(seed + 937)
    for start in range(0, len(data["angles"]), batch_size):
        angles = data["angles"][start : start + batch_size]
        concepts = data["concepts"][start : start + batch_size]
        states = model.frontend(angles)
        output = model.from_state(states)
        concept_outputs.append(output["concept_probs"].cpu())
        label_outputs.append(output["label_prob"].cpu())
        if diagnostics:
            for mode in ("shape", "scale", "both", "zero"):
                controls = controls_for(len(angles), angles.device, mode, concepts)
                corrected[mode].append(
                    model.from_state(states, controls)["label_prob"].cpu()
                )
            measured, labels = sample_shots(output, shots, generator)
            target_codes = concepts[:, 0] * 8 + concepts[:, 1]
            shot_probabilities.append(labels.float().mean(1).cpu())
            shot_concepts.append(
                (measured == target_codes[:, None]).float().mean(1).cpu()
            )
    probabilities = torch.cat(concept_outputs)
    label_probabilities = torch.cat(label_outputs)
    targets = data["labels"].cpu()
    result = {
        "n_samples": len(targets),
        "concept": concept_metrics(probabilities, data["concepts"].cpu()),
        "label": label_metrics(label_probabilities, targets),
    }
    if diagnostics:
        result["control_record_interventions"] = {
            mode: {
                **label_metrics(torch.cat(values), targets),
                "mean_abs_probability_change": float(
                    (torch.cat(values) - label_probabilities).abs().mean()
                ),
            }
            for mode, values in corrected.items()
        }
        result["finite_shots"] = {
            "shots_per_image": shots,
            "seed": seed + 937,
            "label": label_metrics(torch.cat(shot_probabilities), targets),
            "joint_correct_concept_frequency": float(torch.cat(shot_concepts).mean()),
        }
        result["intervention_semantics"] = (
            "Replace only classical X controls; keep actual measured branch and "
            "its input-dependent conditional quantum state. "
            "This is not state correction."
        )
    return result
