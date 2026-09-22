"""Audit all four completed Robot modes without invoking writing constructors."""

from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, cuda_runtime, sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_four_modes.evaluation import evaluate_seed
from experiments.grouped_robot_four_modes.protocol import CELLS, source_hashes
from experiments.grouped_robot_four_modes.reference import (
    Reference as SequentialReference,
)
from experiments.grouped_robot_four_modes.training import Job, verify_job, verify_pair
from experiments.grouped_robot_independent.model import make_model
from experiments.grouped_robot_independent.runner import (
    Experiment as IndependentExperiment,
)
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load

from .protocol import Config


class Sources:
    verify_result_lock = IndependentExperiment.verify_result_lock

    def __init__(self, config: Config, tick=lambda *_a, **_k: None):
        config.validate()
        self.output, self.config = Path(config.reference).resolve(), config.source()
        _, _, self.pilot_config = self.config.sources()
        self.runtime, self._tick = cuda_runtime(0), tick
        self.manifest = read_json(self.output / "manifest.json")
        self.manifest_hash = sha256(self.output / "manifest.json")
        root = Path(self.config.reference)
        identity = {
            "config": self.config.to_dict(),
            "runtime": self.runtime,
            "sources": source_hashes(),
            "reference_manifest_sha256": sha256(root / "manifest.json"),
            "reference_result_lock_sha256": sha256(root / "result_lock.json"),
        }
        if any(self.manifest.get(k) != v for k, v in identity.items()):
            raise ValueError("Four-mode source config/runtime/code/reference changed")
        verify_files(self.output, self.manifest["artifacts"])
        locked = self.verify_result_lock()
        if locked is None:
            raise ValueError("Four-mode source must be complete and locked")
        self.reference = SequentialReference(root, self.runtime, tick)
        self.data = self.reference.data
        self.save_once(
            "reference_lock.json",
            {
                "manifest_sha256": self.manifest_hash,
                "artifacts": self.reference.pins,
            },
        )
        self.save_once(
            "data_reference.json",
            {
                "manifest_sha256": self.manifest_hash,
                "reference_data_lock_sha256": self.reference.data_hash,
                "data_hashes": self.reference.data_hashes,
                "test_read": False,
            },
        )
        self.data_hash = sha256(self.output / "data_reference.json")
        self.initial = load(self.output / "initialization.pt")
        expected = {
            str(s): self.reference.reference.initial[str(s)]
            for s in self.config.seed_values
        }
        if tree_hash(self.initial) != tree_hash(expected):
            raise ValueError("Four-mode initialization/RNG changed")
        self.summary = read_json(self.output / "summary.json")
        if (
            self.summary["status"] != "complete"
            or self.summary["test_read"]
            or self.summary["test_evaluated"]
            or self.summary["seeds"] != list(self.config.seed_values)
        ):
            raise ValueError("Four-mode reference incomplete or wrong protocol")
        self.training, self.evaluations = {}, {}
        for seed in self.config.seed_values:
            paired = []
            for cell in CELLS:
                tick("verifying_reference_training", source_cell=f"seed_{seed}/{cell}")
                record = verify_job(Job(self, seed, cell))
                if record not in self.summary["training"]:
                    raise ValueError("Source checkpoint differs from summary")
                self.training[seed, cell] = record
                paired.append(record)
            for record in paired[1:]:
                verify_pair(paired[0], record)
            for row in evaluate_seed(self, seed, locked=True):
                self.evaluations[seed, row["role"], row["cell"], row["condition"]] = row
        self.hashes = {
            **self.reference.pins,
            **{
                str(self.output / name): sha256(self.output / name)
                for name in {*locked["artifacts"], "manifest.json", "result_lock.json"}
            },
        }
        pilot = self.reference.reference.reference
        self.audit = pilot.audit
        self.preprocessing_path = pilot.output / "preprocessing.json"
        self.dataset = Path(self.pilot_config.dataset)

    def save_once(self, name: str, value: dict) -> None:
        if read_json(self.output / name) != value:
            raise ValueError(f"Read-only four-mode record changed: {name}")

    def tick(self, status: str, **details) -> None:
        self._tick(status, **details)

    def model_path(self, seed: int, training: str) -> Path:
        if training in CELLS:
            return self.output / f"seed_{seed}/training/{training}/endpoint.pt"
        return self.reference.checkpoint_path(seed, training)

    def reference_predictions(self, seed: int, training: str, mode: str) -> Path:
        return Path(
            self.evaluations[seed, "validation", training, mode]["predictions_path"]
        )

    @staticmethod
    def make_model(weights: dict):
        return make_model(weights).eval().requires_grad_(False)

    def verify_unchanged(self) -> None:
        verify_files(Path("/"), self.hashes)
        verify_files(ROOT, self.manifest["sources"])
        self.reference.verify_unchanged()
