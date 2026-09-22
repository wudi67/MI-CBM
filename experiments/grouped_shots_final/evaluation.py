"""Frozen CUDA inference and repeated joint (concept record, label) sampling."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import cast

import torch

from experiments.grouped_dynamic_vqc.evaluation import concept_metrics, label_metrics
from experiments.grouped_dynamic_vqc.model import (
    ControlMode,
    controls_for,
    sample_shots,
)

from .protocol import Config


def validate_joint(raw: dict) -> None:
    p, one = raw["concept_probabilities"], raw["branch_label_mass"]
    if p.shape != (len(raw["labels"]), 32) or one.shape != p.shape:
        raise ValueError("Expected all 32 measured branches per image")
    if (
        not torch.isfinite(p).all()
        or not torch.isfinite(one).all()
        or p.min() < -1e-6
        or one.min() < -1e-6
        or (one - p).max() > 1e-6
        or (p.sum(1) - 1).abs().max() > 1e-5
    ):
        raise ValueError("Invalid joint measurement distribution")


@torch.no_grad()
def infer(
    model, data: dict, states: torch.Tensor, mode: str, batch: int, tick: Callable
) -> dict:
    concepts, masses = [], []
    for start in range(0, len(states), batch):
        stop = min(start + batch, len(states))
        original = states[start:stop]
        controls = controls_for(
            len(original),
            original.device,
            cast(ControlMode, mode),
            data["concepts"][start:stop].cuda(),
        )
        out = model.from_state(original, controls)
        p = original.reshape(-1, 32, 32).abs().square().sum(-1)
        if (p - out["concept_probs"]).abs().max() > 2e-6:
            raise ValueError("Feedback changed the original measurement probabilities")
        concepts.append(p.cpu())
        masses.append(out["branch_label_mass"].cpu())
        tick("inference", offset=stop)
    raw = {k: data[k].cpu() for k in ("source_index", "labels", "concepts")}
    raw.update(
        concept_probabilities=torch.cat(concepts), branch_label_mass=torch.cat(masses)
    )
    validate_joint(raw)
    return raw


def exact_metrics(raw: dict) -> dict:
    validate_joint(raw)
    return {
        "concept": concept_metrics(raw["concept_probabilities"], raw["concepts"]),
        "label": label_metrics(raw["branch_label_mass"].sum(1), raw["labels"]),
    }


def draw_seed(
    base: int, role: str, seed: int, training: str, mode: str, repeat: int
) -> int:
    text = f"{base}/{role}/{seed}/{training}/{mode}/{repeat}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (
        2**63 - 1
    )


SAMPLE_FIELDS = (
    "label_ones",
    "shape_prediction",
    "scale_prediction",
    "joint_prediction",
    "true_record_count",
    "invalid_record_count",
)


@torch.no_grad()
def sample_repeats(
    raw: dict, config: Config, identity: tuple[str, int, str, str], tick: Callable
) -> dict:
    """Use nested shot prefixes and independent RNGs across repetitions."""
    validate_joint(raw)
    n = len(raw["labels"])
    budgets = config.shot_list()
    samples = {
        key: torch.empty((config.repeats, len(budgets), n), dtype=torch.int32)
        for key in SAMPLE_FIELDS
    }
    targets = (raw["concepts"][:, 0] * 8 + raw["concepts"][:, 1]).cuda()
    codes = torch.arange(32, device="cuda")
    invalid = (codes // 8 == 3) | (codes % 8 >= 6)
    seeds = []
    p = raw["concept_probabilities"].cuda()
    mass = raw["branch_label_mass"].cuda()
    for repetition in range(config.repeats):
        seed = draw_seed(config.sampling_seed, *identity, repetition)
        seeds.append(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        for start in range(0, n, config.eval_batch_size):
            stop = min(start + config.eval_batch_size, n)
            measured, labels = sample_shots(
                {"concept_probs": p[start:stop], "branch_label_mass": mass[start:stop]},
                budgets[-1],
                generator,
            )
            for j, shots in enumerate(budgets):
                records = measured[:, :shots]
                counts = torch.zeros(
                    (stop - start, 32), device="cuda", dtype=torch.int64
                )
                counts.scatter_add_(1, records, torch.ones_like(records))
                groups = counts.reshape(-1, 4, 8)
                values = {
                    "label_ones": labels[:, :shots].sum(1),
                    "shape_prediction": groups.sum(2).argmax(1),
                    "scale_prediction": groups.sum(1).argmax(1),
                    "joint_prediction": counts.argmax(1),
                    "true_record_count": counts.gather(
                        1, targets[start:stop, None]
                    ).squeeze(1),
                    "invalid_record_count": counts[:, invalid].sum(1),
                }
                for key, value in values.items():
                    samples[key][repetition, j, start:stop] = value.cpu().to(
                        torch.int32
                    )
            tick("sampling", repetition=repetition + 1, offset=stop)
    return {"budgets": budgets, "seeds": seeds, "values": samples}


def sampled_metrics(raw: dict, samples: dict) -> list[dict]:
    """Recompute finite-shot metrics from saved per-image observations."""
    truth = raw["concepts"]
    codes = truth[:, 0] * 8 + truth[:, 1]
    rows = []
    for repetition, seed in enumerate(samples["seeds"]):
        for j, shots in enumerate(samples["budgets"]):
            v = {key: value[repetition, j] for key, value in samples["values"].items()}
            if any(x.shape != codes.shape for x in v.values()):
                raise ValueError("Sample rows do not match evaluation images")
            for key in ("label_ones", "true_record_count", "invalid_record_count"):
                if v[key].min() < 0 or v[key].max() > shots:
                    raise ValueError("Sample count outside measurement budget")
            shape = v["shape_prediction"] == truth[:, 0]
            scale = v["scale_prediction"] == truth[:, 1]
            predicted = v["joint_prediction"]
            concept = {
                "shape_marginal_argmax_accuracy": float(shape.float().mean()),
                "scale_marginal_argmax_accuracy": float(scale.float().mean()),
                "group_argmax_exact_accuracy": float((shape & scale).float().mean()),
                "joint_map_accuracy": float((predicted == codes).float().mean()),
                "joint_single_shot_probability": float(
                    v["true_record_count"].float().mean() / shots
                ),
                "invalid_code_probability": float(
                    v["invalid_record_count"].float().mean() / shots
                ),
                "invalid_joint_map_fraction": float(
                    ((predicted // 8 == 3) | (predicted % 8 >= 6)).float().mean()
                ),
                "joint_nll": float(
                    -(v["true_record_count"].float() / shots)
                    .clamp_min(1e-7)
                    .log()
                    .mean()
                ),
            }
            rows.append(
                {
                    "shots": shots,
                    "repeat": repetition,
                    "sampling_seed": seed,
                    "label": label_metrics(
                        v["label_ones"].float() / shots, raw["labels"]
                    ),
                    "concept": concept,
                }
            )
    return rows
