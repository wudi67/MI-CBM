"""Read-only verification of the completed A/B experiment and its frozen states."""

from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_label_depth.evaluation import verify_condition
from experiments.grouped_robot_label_depth.model import verify_initializations
from experiments.grouped_robot_label_depth.protocol import CELLS, CONDITIONS, Config
from experiments.grouped_robot_label_depth.protocol import source_hashes as old_sources
from experiments.grouped_robot_label_depth.runner import Experiment as DepthExperiment
from experiments.grouped_robot_label_depth.training import Job, verify_job
from experiments.grouped_robot_mlp_diagnostic.reference import (
    Reference as EarlierReference,
)
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash


class Reference(DepthExperiment):
    """Call only read-only helpers of the previous experiment."""

    def __init__(self, output: Path, runtime: dict, tick) -> None:  # pylint: disable=super-init-not-called
        self.output = output.resolve()
        self.config = Config(**read_json(self.output / "config.json"))
        source, _, _ = self.config.sources()
        self.reuse_a0 = self.config.head_epochs == source.head_epochs
        self.runtime = runtime
        self.manifest = read_json(self.output / "manifest.json")
        self.manifest_hash = sha256(self.output / "manifest.json")
        root = Path(self.config.reference)
        identity = {
            "config": self.config.to_dict(),
            "runtime": runtime,
            "sources": old_sources(),
            "reference_manifest_sha256": sha256(root / "manifest.json"),
            "reference_result_lock_sha256": sha256(root / "result_lock.json"),
        }
        if any(self.manifest.get(k) != v for k, v in identity.items()):
            raise ValueError(
                "Depth source configuration, runtime or source pins changed"
            )
        verify_files(self.output, self.manifest["artifacts"])
        locked = self.verify_result_lock()
        if locked is None:
            raise ValueError(
                "Depth source must have completed all cells and evaluations"
            )
        self.reference = EarlierReference(root, runtime, tick)
        if (
            read_json(self.output / "reference_lock.json")["artifacts"]
            != self.reference.pins
        ):
            raise ValueError("Depth source reference lock changed")
        self.data = self.reference.data
        self.frontend_hash = self.reference.frontend_hash
        self.pilot = self.reference.reference.config
        for name in ("initialization_lock.json", "state_cache_lock.json"):
            lock = read_json(self.output / name)
            if lock["manifest_sha256"] != self.manifest_hash:
                raise ValueError("Depth source cache/initialization manifest changed")
            verify_files(self.output, lock["artifacts"])
        self.initial = load(self.output / "initialization.pt")
        verify_initializations(self.initial, self.frontend_hash)
        initial_lock = read_json(self.output / "initialization_lock.json")
        if initial_lock["model_hashes"] != {
            k: state_hash(v["model"]) for k, v in self.initial.items()
        }:
            raise ValueError("Depth source initial model hashes changed")
        self.data_hash = sha256(self.output / "state_cache_lock.json")
        states = load(self.output / "frontend_states.pt")
        if set(states) != {"train", "validation"}:
            raise ValueError("Only train/validation states may be loaded")
        self.state_cache = {role: value.cuda() for role, value in states.items()}
        for role, value in self.state_cache.items():
            if value.shape != (len(self.data[role]["labels"]), 1024):
                raise ValueError("Cached frontend state dimensions changed")
            born = value.reshape(-1, 32, 32).abs().square().sum(-1).cpu()
            original = self.reference.quantum_raw[role]["measured"][
                "concept_probabilities"
            ]
            torch.testing.assert_close(born, original, atol=2e-6, rtol=2e-6)
        self.training = {}
        for depth, index in CELLS:
            job = Job(self, depth, index)
            tick("verifying_depth_training", source_cell=job.name)
            self.training[job.name] = verify_job(job)
            for role in ("train", "validation"):
                for name, mask in CONDITIONS:
                    tick("verifying_depth_evaluation", source_cell=job.name)
                    if verify_condition(job, role, name, mask) is None:
                        raise ValueError("Depth source evaluation is incomplete")
        self.pins = {
            **self.reference.pins,
            **{
                str(self.output / name): sha256(self.output / name)
                for name in (*locked["artifacts"], "result_lock.json", "manifest.json")
            },
        }

    def checkpoint(self, index: int) -> Path:
        return self.output / f"L5/init_{index}/training/independent/endpoint.pt"

    def uniform_raw(self, index: int, role: str, condition: str) -> dict:
        return load(
            self.output / f"L5/init_{index}" / role / condition / "predictions.pt"
        )

    def verify_unchanged(self) -> None:
        verify_files(Path("/"), self.pins)
        verify_files(ROOT, self.manifest["sources"])
        self.reference.verify_unchanged()
