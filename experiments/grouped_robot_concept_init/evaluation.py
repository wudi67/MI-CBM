"""Fixed-endpoint concept scores on unchanged train/validation records."""

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    atomic_checkpoint,
    atomic_json,
    sha256,
)
from experiments.grouped_robot_pilot.evaluation import evaluate, metrics
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash


def evaluate_job(job, existing: bool = False) -> list:
    values, model = [], None
    for role in ("train", "validation"):
        root = job.output / role
        identity = {
            "manifest_sha256": job.manifest_hash,
            "data_lock_sha256": job.data_hash,
            "cell_name": job.name,
            "role": role,
            "checkpoint_sha256": sha256(job.checkpoint_path),
            "epochs": job.config.concept_epochs,
            "test_evaluated": False,
        }
        lock_path = root / "evaluation_lock.json"
        if not lock_path.exists():
            if existing:
                raise ValueError("Locked result is missing concept evaluation")
            if model is None:
                model = job.make_model(load(job.checkpoint_path)["model"])
                model.eval().requires_grad_(False)
            before = state_hash(model.state_dict())
            metric, raw = evaluate(
                model,
                job.data[role],
                job.config.eval_batch_size,
                concept_only=True,
                tick=job.parent.tick,
            )
            if before != state_hash(model.state_dict()):
                raise ValueError("Evaluation changed model parameters")
            atomic_checkpoint(root / "predictions.pt", raw)
            atomic_json(
                root / "evaluation.json",
                {
                    **identity,
                    "n_samples": len(raw["labels"]),
                    "metrics": metric,
                },
            )
            atomic_json(
                lock_path,
                {
                    **identity,
                    "artifacts": {
                        n: sha256(root / n)
                        for n in ("evaluation.json", "predictions.pt")
                    },
                },
            )
        lock, value = read_json(lock_path), read_json(root / "evaluation.json")
        if any(lock.get(k) != v or value.get(k) != v for k, v in identity.items()):
            raise ValueError("Concept evaluation identity changed")
        verify_files(root, lock["artifacts"])
        raw = load(root / "predictions.pt")
        if "branch_label_mass" in raw:
            raise ValueError("Concept-only comparison must not evaluate label circuit")
        for key in ("concepts", "labels", "source_index", "robot_ids"):
            if not torch.equal(raw[key], job.data[role][key].cpu()):
                raise ValueError("Concept evaluation samples changed")
        if value["n_samples"] != len(raw["labels"]) or value["metrics"] != metrics(raw):
            raise ValueError("Predictions do not reproduce concept metrics")
        if role == "validation":
            last = load(job.checkpoint_path)["progress"]["history"][-1]
            if value["metrics"] != last["validation"]:
                raise ValueError("Final training/evaluation concept scores differ")
        values.append(value)
    return values
