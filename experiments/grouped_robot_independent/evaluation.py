"""Three evaluation conditions on identical samples and retained Born branches."""

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    atomic_checkpoint,
    atomic_json,
    sha256,
)
from experiments.grouped_robot_pilot.evaluation import evaluate, metrics
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load

from .model import make_model
from .protocol import CONDITIONS


def evaluate_seed(shared, seed: int, locked: bool = False) -> list[dict]:
    rows = []
    for role, data in shared.data.items():
        reference = None
        for cell, condition, mask in CONDITIONS:
            shared.cell = f"seed_{seed}/{role}/{cell}/{condition}"
            shared.tick("evaluating", offset=0)
            directory = shared.output / shared.cell
            path = shared.checkpoint_path(seed, cell)
            expected = {
                "manifest_sha256": shared.manifest_hash,
                "data_lock_sha256": shared.data_hash,
                "checkpoint_sha256": sha256(path),
                "seed": seed,
                "role": role,
                "cell": cell,
                "condition": condition,
                "mask": mask,
                "zero_control": cell == "no_feedback",
                "test_evaluated": False,
            }
            if (directory / "evaluation_lock.json").exists():
                lock = read_json(directory / "evaluation_lock.json")
                if lock["identity"] != expected:
                    raise ValueError("Evaluation identity changed")
                verify_files(directory, lock["artifacts"])
                row = read_json(directory / "evaluation.json")
                raw = load(directory / "predictions.pt")
                if any(row.get(k) != v for k, v in expected.items()) or row[
                    "metrics"
                ] != metrics(raw):
                    raise ValueError("Saved evaluation does not match predictions")
            else:
                if locked:
                    raise ValueError("Locked evaluation is missing")
                model = make_model(load(path)["model"])
                shared.prepare_states(model)
                metric, raw = evaluate(
                    model,
                    data,
                    shared.reference.config.eval_batch_size,
                    zero=cell == "no_feedback",
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
            for key in ("concepts", "labels", "source_index", "robot_ids"):
                if not torch.equal(raw[key], data[key].cpu()):
                    raise ValueError("Evaluation sample identities changed")
            if reference is None:
                reference = raw["concept_probabilities"]
            else:
                torch.testing.assert_close(
                    raw["concept_probabilities"], reference, atol=2e-6, rtol=2e-6
                )
            rows.append(row)
    return rows
