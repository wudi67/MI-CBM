"""Sequential normal/corrected predictions and unchanged historical controls."""

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

from .protocol import CONDITIONS


def evaluate_seed(shared, seed: int, locked: bool) -> list[dict]:
    rows = []
    for role, data in shared.data.items():
        original_path = (
            shared.reference.output
            / f"seed_{seed}/{role}/independent/measured/predictions.pt"
        )
        original = load(original_path)
        for cell, condition, mask in CONDITIONS:
            shared.cell = f"seed_{seed}/{role}/{cell}/{condition}"
            shared.tick("evaluating", offset=0)
            directory = shared.output / shared.cell
            if cell != "sequential":
                source_directory = shared.reference.output / shared.cell
                marker = {
                    "manifest_sha256": shared.manifest_hash,
                    "evaluation_path": str(source_directory / "evaluation.json"),
                    "predictions_path": str(source_directory / "predictions.pt"),
                    "evaluation_sha256": sha256(source_directory / "evaluation.json"),
                    "predictions_sha256": sha256(source_directory / "predictions.pt"),
                    "reused": True,
                }
                marker_name = str(
                    (directory / "evaluation_reference.json").relative_to(shared.output)
                )
                if locked and not (shared.output / marker_name).exists():
                    raise ValueError("Locked reference evaluation is missing")
                shared.save_once(marker_name, marker)
                row = shared.reference.evaluations[seed, role, cell, condition]
                rows.append(
                    {
                        **row,
                        "reused": True,
                        "evaluation_path": marker["evaluation_path"],
                        "predictions_path": marker["predictions_path"],
                    }
                )
                continue
            checkpoint = shared.output / f"seed_{seed}/training/sequential/endpoint.pt"
            expected = {
                "manifest_sha256": shared.manifest_hash,
                "data_lock_sha256": shared.data_hash,
                "checkpoint_sha256": sha256(checkpoint),
                "seed": seed,
                "role": role,
                "cell": cell,
                "condition": condition,
                "mask": mask,
                "zero_control": False,
                "test_evaluated": False,
            }
            if (directory / "evaluation_lock.json").exists():
                lock = read_json(directory / "evaluation_lock.json")
                if lock["identity"] != expected:
                    raise ValueError("Sequential evaluation identity changed")
                verify_files(directory, lock["artifacts"])
                row = read_json(directory / "evaluation.json")
                raw = load(directory / "predictions.pt")
                if any(row.get(k) != v for k, v in expected.items()) or row[
                    "metrics"
                ] != metrics(raw):
                    raise ValueError("Sequential metrics differ from raw probabilities")
            else:
                if locked:
                    raise ValueError("Locked Sequential evaluation is missing")
                model = make_model(load(checkpoint)["model"])
                shared.prepare_states(model)
                metric, raw = evaluate(
                    model,
                    data,
                    shared.reference.reference.config.eval_batch_size,
                    mask=mask,
                    states=shared.state_cache[role],
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
                if not torch.equal(raw[key], data[key].cpu()) or not torch.equal(
                    raw[key], original[key]
                ):
                    raise ValueError("Sequential evaluation sample identities changed")
            torch.testing.assert_close(
                raw["concept_probabilities"],
                original["concept_probabilities"],
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
