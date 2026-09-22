"""Four new conditions plus read-only Independent/Sequential predictions."""

from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    atomic_checkpoint,
    atomic_json,
    sha256,
)
from experiments.grouped_robot_independent.model import make_model
from experiments.grouped_robot_pilot.evaluation import evaluate, metrics
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load

from .protocol import CELLS, CONDITIONS


def evaluate_seed(shared, seed: int, locked: bool) -> list[dict]:
    rows = []
    for role, data in shared.data.items():
        joint_probabilities = None
        for cell, condition, mask in CONDITIONS:
            shared.cell = f"seed_{seed}/{role}/{cell}/{condition}"
            shared.tick("evaluating", offset=0)
            directory = shared.output / shared.cell
            if cell not in CELLS:
                row = shared.reference.evaluations[seed, role, cell, condition]
                marker = {
                    "manifest_sha256": shared.manifest_hash,
                    "evaluation_path": row["evaluation_path"],
                    "predictions_path": row["predictions_path"],
                    "evaluation_sha256": sha256(Path(row["evaluation_path"])),
                    "predictions_sha256": sha256(Path(row["predictions_path"])),
                    "reused": True,
                }
                name = str(
                    (directory / "evaluation_reference.json").relative_to(shared.output)
                )
                if locked and not (shared.output / name).exists():
                    raise ValueError("Locked reference evaluation is missing")
                shared.save_once(name, marker)
                rows.append({**row, "reused": True})
                continue
            checkpoint = shared.output / f"seed_{seed}/training/{cell}/endpoint.pt"
            expected = {
                "manifest_sha256": shared.manifest_hash,
                "data_lock_sha256": shared.data_hash,
                "checkpoint_sha256": sha256(checkpoint),
                "seed": seed,
                "role": role,
                "cell": cell,
                "condition": condition,
                "mask": mask,
                "zero_control": cell == "joint_no_feedback",
                "concept_supervised": cell != "standard",
                "test_evaluated": False,
            }
            if (directory / "evaluation_lock.json").exists():
                lock = read_json(directory / "evaluation_lock.json")
                if lock["identity"] != expected:
                    raise ValueError("Full-circuit evaluation identity changed")
                verify_files(directory, lock["artifacts"])
                row, raw = (
                    read_json(directory / "evaluation.json"),
                    load(directory / "predictions.pt"),
                )
                if any(row.get(k) != v for k, v in expected.items()) or row[
                    "metrics"
                ] != metrics(raw):
                    raise ValueError(
                        "Full-circuit metrics differ from raw probabilities"
                    )
            else:
                if locked:
                    raise ValueError("Locked full-circuit evaluation is missing")
                model = make_model(load(checkpoint)["model"])
                metric, raw = evaluate(
                    model,
                    data,
                    shared.pilot_config.eval_batch_size,
                    mask=mask,
                    zero=expected["zero_control"],
                    tick=shared.tick,
                )
                row = {**expected, "metrics": metric, "n_samples": len(data["labels"])}
                atomic_checkpoint(directory / "predictions.pt", raw)
                atomic_json(directory / "evaluation.json", row)
                atomic_json(
                    directory / "evaluation_lock.json",
                    {
                        "identity": expected,
                        "artifacts": {
                            name: sha256(directory / name)
                            for name in ("predictions.pt", "evaluation.json")
                        },
                    },
                )
                del model
            for key in ("labels", "concepts", "source_index", "robot_ids"):
                if not torch.equal(raw[key], data[key].cpu()):
                    raise ValueError("Evaluation sample identities changed")
            if cell == "joint":
                if joint_probabilities is None:
                    joint_probabilities = raw["concept_probabilities"]
                else:
                    torch.testing.assert_close(
                        raw["concept_probabilities"],
                        joint_probabilities,
                        atol=2e-6,
                        rtol=2e-6,
                    )
            rows.append(
                {
                    **row,
                    "reused": False,
                    "evaluation_path": str(directory / "evaluation.json"),
                    "predictions_path": str(directory / "predictions.pt"),
                }
            )
    return rows
