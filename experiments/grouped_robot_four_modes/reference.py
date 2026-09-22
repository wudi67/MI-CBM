"""Read-only admission of completed Sequential and Independent evidence."""

from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_independent.runner import (
    Experiment as IndependentExperiment,
)
from experiments.grouped_robot_independent.training import Job as IndependentJob
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_robot_sequential.evaluation import evaluate_seed
from experiments.grouped_robot_sequential.protocol import Config, source_hashes
from experiments.grouped_robot_sequential.reference import (
    Reference as IndependentReference,
)
from experiments.grouped_robot_sequential.training import Job, verify_job


class Reference:
    """Never invoke a historical experiment's writing constructor."""

    verify_result_lock = IndependentExperiment.verify_result_lock

    def __init__(self, root: Path, runtime: dict, tick):
        self.output, self.runtime, self._tick = root.resolve(), runtime, tick
        self.config = Config(**read_json(self.output / "config.json"))
        self.config.sources()
        self.manifest = read_json(self.output / "manifest.json")
        self.manifest_hash = sha256(self.output / "manifest.json")
        independent_root = Path(self.config.reference)
        identity = {
            "config": self.config.to_dict(),
            "runtime": runtime,
            "sources": source_hashes(),
            "reference_manifest_sha256": sha256(independent_root / "manifest.json"),
            "reference_result_lock_sha256": sha256(
                independent_root / "result_lock.json"
            ),
        }
        if any(self.manifest.get(k) != v for k, v in identity.items()):
            raise ValueError("Sequential source config/runtime/code/reference changed")
        verify_files(self.output, self.manifest["artifacts"])
        locked = self.verify_result_lock()
        if locked is None:
            raise ValueError("Sequential reference must be complete and locked")
        self.reference = IndependentReference(independent_root, runtime, tick)
        self.data, self.data_hashes = self.reference.data, self.reference.data_hashes
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
                "data_hashes": self.data_hashes,
                "test_read": False,
            },
        )
        self.data_hash = sha256(self.output / "data_reference.json")
        self.initial = load(self.output / "initialization.pt")
        expected = {
            str(seed): IndependentJob(self.reference, seed, "independent").initial_for(
                "independent"
            )
            for seed in self.config.seed_values
        }
        if tree_hash(self.initial) != tree_hash(expected):
            raise ValueError("Sequential initial trained frontend/head/RNG changed")
        self.summary = read_json(self.output / "summary.json")
        if (
            self.summary["status"] != "complete"
            or self.summary["test_read"]
            or self.summary["test_evaluated"]
            or self.summary["seeds"] != list(self.config.seed_values)
        ):
            raise ValueError(
                "Sequential reference uses an incomplete/different protocol"
            )
        self.training = dict(self.reference.training)
        self.evaluations = {}
        for seed in self.config.seed_values:
            tick("verifying_sequential_reference", source_cell=f"seed_{seed}")
            record = verify_job(Job(self, seed))
            if record not in self.summary["training"]:
                raise ValueError("Sequential summary differs from actual checkpoint")
            self.training[seed, "sequential"] = record
            for row in evaluate_seed(self, seed, locked=True):
                self.evaluations[seed, row["role"], row["cell"], row["condition"]] = row
        self.pins = {
            **self.reference.pins,
            **{
                str(self.output / name): sha256(self.output / name)
                for name in {*locked["artifacts"], "manifest.json", "result_lock.json"}
            },
        }

    def save_once(self, name: str, value: dict) -> None:
        # Historical verify/evaluate helpers call save_once: only compare here.
        if read_json(self.output / name) != value:
            raise ValueError(f"Read-only Sequential record changed: {name}")

    def tick(self, status: str, **details) -> None:
        self._tick(status, **details)

    def checkpoint_path(self, seed: int, cell: str) -> Path:
        if cell == "sequential":
            return self.output / f"seed_{seed}/training/sequential/endpoint.pt"
        return self.reference.checkpoint_path(seed, cell)

    def verify_unchanged(self) -> None:
        verify_files(Path("/"), self.pins)
        verify_files(ROOT, self.manifest["sources"])
        self.reference.verify_unchanged()
