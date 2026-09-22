"""Read-only audit of the actual 300+300 model, data, and saved quantum predictions."""

from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_continuation.runner import TRAIN_CONDITIONS, verify_pair
from experiments.grouped_robot_continuation.training import Job, verify_job
from experiments.grouped_robot_label_continuation.protocol import Config as SourceConfig
from experiments.grouped_robot_label_continuation.protocol import (
    source_hashes as upstream,
)
from experiments.grouped_robot_label_continuation.reference import (
    Reference as EarlierReference,
)
from experiments.grouped_robot_label_continuation.runner import (
    Experiment as SourceExperiment,
)
from experiments.grouped_robot_pilot.protocol import CONDITIONS, read_json, verify_files
from experiments.grouped_robot_pilot.runner import Experiment as PilotExperiment
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .model import CONDITIONS as MLP_CONDITIONS


class Reference(SourceExperiment):
    """Invoke only upstream verification methods; never its writing constructor."""

    def __init__(self, output: Path, runtime: dict, tick) -> None:  # pylint: disable=super-init-not-called
        self.output = output.resolve()
        self.config = SourceConfig(**read_json(self.output / "config.json"))
        self.config.sources()
        self.runtime = runtime
        self.manifest = read_json(self.output / "manifest.json")
        self.manifest_hash = sha256(self.output / "manifest.json")
        root = Path(self.config.reference)
        expected = {
            "config": self.config.to_dict(),
            "runtime": runtime,
            "sources": upstream(),
            "reference_manifest_sha256": sha256(root / "manifest.json"),
            "reference_result_lock_sha256": sha256(root / "result_lock.json"),
        }
        if any(self.manifest.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "Source configuration, CUDA runtime or source pins changed"
            )
        verify_files(self.output, self.manifest["artifacts"])
        locked = self.verify_result_lock()
        if locked is None:
            raise ValueError("The source label continuation must be fully completed")
        self.reference = EarlierReference(root, runtime, tick)
        if (
            read_json(self.output / "reference_lock.json")["artifacts"]
            != self.reference.pins
        ):
            raise ValueError("Source reference lock changed")
        self.data = self.reference.data
        self.training = {}
        for cell in ("independent", "no_feedback"):
            tick("verifying_source_training", source_cell=cell)
            self.training[f"long_label/{cell}"] = verify_job(
                Job(self, "long_label", cell)
            )
        verify_pair(self, "long_label", self.training)
        for arm in ("baseline", "long_label"):
            for role, conditions in (
                ("validation", CONDITIONS),
                ("train", TRAIN_CONDITIONS),
            ):
                for cell, name, mask in conditions:
                    tick(
                        "verifying_source_evaluation",
                        source_cell=f"{arm}/{role}/{name}",
                    )
                    if self.verify_condition(arm, role, cell, name, mask) is None:
                        raise ValueError("Source evaluation is incomplete")
        self.pins = {
            **self.reference.pins,
            **{
                str(self.output / name): sha256(self.output / name)
                for name in (*locked["artifacts"], "result_lock.json", "manifest.json")
            },
        }
        self.checkpoint = self.checkpoint_path("long_label", "independent")
        self.frontend_hash = self.training["long_label/independent"]["frontend_sha256"]
        self.quantum_raw = {}
        for role in ("train", "validation"):
            self.quantum_raw[role] = {}
            for name, _ in MLP_CONDITIONS:
                if role == "train" and name == "correct_foot_shape":
                    continue  # This historical training diagnostic was not run.
                path = (
                    self.output
                    / "long_label"
                    / role
                    / "independent"
                    / name
                    / "predictions.pt"
                )
                self.quantum_raw[role][name] = load(path)

    @torch.no_grad()
    def export(self, tick) -> tuple[dict, dict]:
        """Recompute Born probabilities on CUDA and reuse the exact locked originals."""
        model = PilotExperiment.make_model(load(self.checkpoint)["model"])
        model.eval().requires_grad_(False)
        before = state_hash(model.state_dict())
        if state_hash(model.frontend.state_dict()) != self.frontend_hash:
            raise ValueError("Wrong frozen frontend checkpoint")
        packed, audit = {}, {}
        batch = self.reference.config.eval_batch_size
        for role in ("train", "validation"):
            data = self.data[role]
            saved = self.quantum_raw[role]["measured"]
            maximum = 0.0
            for start in range(0, len(data["labels"]), batch):
                state = model.frontend(data["angles"][start : start + batch])
                fresh = state.reshape(-1, 32, 32).abs().square().sum(-1).cpu()
                original = saved["concept_probabilities"][start : start + batch]
                torch.testing.assert_close(fresh, original, atol=2e-6, rtol=2e-6)
                maximum = max(maximum, float((fresh - original).abs().max()))
                tick(
                    "verifying_frozen_frontend_cuda",
                    role=role,
                    offset=start + len(fresh),
                )
            packed[role] = {
                k: saved[k].clone()
                for k in (
                    "concepts",
                    "labels",
                    "source_index",
                    "robot_ids",
                    "concept_probabilities",
                )
            }
            packed[role]["quantum_label_probabilities"] = {
                name: raw["branch_label_mass"].sum(1)
                for name, raw in self.quantum_raw[role].items()
            }
            audit[role] = {
                "n_samples": len(data["labels"]),
                "max_born_difference": maximum,
            }
        if before != state_hash(model.state_dict()):
            raise ValueError("Inference changed the frozen quantum model")
        audit.update(
            frontend_sha256=self.frontend_hash,
            checkpoint_sha256=sha256(self.checkpoint),
            device=str(next(model.parameters()).device),
            test_read=False,
        )
        return packed, audit

    def verify_unchanged(self) -> None:
        verify_files(Path("/"), self.pins)
        verify_files(ROOT, self.manifest["sources"])
        self.reference.verify_unchanged()
