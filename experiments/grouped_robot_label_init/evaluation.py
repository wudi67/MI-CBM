"""Evaluate normal and corrected classical controls on the same retained states."""

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    atomic_checkpoint,
    atomic_json,
    report,
    sha256,
)
from experiments.grouped_robot_pilot.evaluation import evaluate, metrics
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .protocol import CONDITIONS


def expected(job, role: str, name: str, mask: int) -> dict:
    return {
        **job.identity(),
        "role": role,
        "condition": name,
        "correction_mask": mask,
        "checkpoint_sha256": sha256(job.checkpoint_path),
        "zero_feedback": False,
        "test_evaluated": False,
    }


def verify_condition(job, role: str, name: str, mask: int) -> dict | None:
    directory = job.output / role / name
    if not (directory / "evaluation_lock.json").exists():
        return None
    lock = read_json(directory / "evaluation_lock.json")
    verify_files(directory, lock["artifacts"])
    value = read_json(directory / "evaluation.json")
    identity = expected(job, role, name, mask)
    if any(value.get(k) != v or lock.get(k) != v for k, v in identity.items()):
        raise ValueError("Evaluation identity changed")
    raw = load(directory / "predictions.pt")
    for key in ("source_index", "robot_ids", "concepts", "labels"):
        if not torch.equal(raw[key], job.data[role][key].cpu()):
            raise ValueError("Evaluation samples changed")
    if value["metrics"] != metrics(raw):
        raise ValueError("Raw predictions do not reproduce reported metrics")
    original = job.parent.reference.reference.quantum_raw[role]["measured"][
        "concept_probabilities"
    ]
    torch.testing.assert_close(
        raw["concept_probabilities"], original, atol=2e-6, rtol=2e-6
    )
    if job.reused:
        gold = job.parent.reference.uniform_raw(job.index, role, name)
        torch.testing.assert_close(
            raw["branch_label_mass"], gold["branch_label_mass"], atol=2e-6, rtol=2e-6
        )
    return value


@torch.no_grad()
def evaluate_job(job, existing: bool) -> list:
    values, model = [], None
    for role in ("train", "validation"):
        for name, mask in CONDITIONS:
            job.parent.tick("verifying_evaluation", role=role, condition=name)
            value = verify_condition(job, role, name, mask)
            if value is None:
                if existing:
                    raise ValueError("Locked result is missing an evaluation")
                if model is None:
                    model = job.make_model(load(job.checkpoint_path)["model"])
                    model.eval().requires_grad_(False)
                before = state_hash(model.state_dict())
                metric, raw = evaluate(
                    model,
                    job.data[role],
                    job.config.eval_batch_size,
                    mask=mask,
                    states=job.state_cache[role],
                    tick=job.parent.tick,
                )
                if before != state_hash(model.state_dict()):
                    raise ValueError("Evaluation changed circuit parameters")
                directory = job.output / role / name
                identity = expected(job, role, name, mask)
                atomic_checkpoint(directory / "predictions.pt", raw)
                atomic_json(
                    directory / "evaluation.json",
                    {**identity, "n_samples": len(raw["labels"]), "metrics": metric},
                )
                atomic_json(
                    directory / "evaluation_lock.json",
                    {
                        **identity,
                        "artifacts": {
                            n: sha256(directory / n)
                            for n in ("predictions.pt", "evaluation.json")
                        },
                    },
                )
                value = verify_condition(job, role, name, mask)
                report(
                    f"{job.name}/{role}/{name}: label={metric['label']['accuracy']:.2%}"
                )
            assert value is not None
            values.append(value)
    # Every freshly trained cell must report the same final true-input metrics in
    # training history and in the standalone final evaluations.
    if not job.reused:
        history = read_json(job.output / "training/independent/history.json")
        for role, key in (("train", "train_true"), ("validation", "validation_true")):
            metric = next(
                v["metrics"]
                for v in values
                if v["role"] == role and v["condition"] == "correct_all_five"
            )
            if history[-1][key] != metric:
                raise ValueError(
                    "Final true-control training diagnostic differs from evaluation"
                )
    return values
