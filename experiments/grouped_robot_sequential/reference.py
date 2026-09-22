"""Load completed Independent results through read-only validation paths."""

from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, array_hash, sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_continuation.reference import Reference as PilotReference
from experiments.grouped_robot_independent.evaluation import evaluate_seed
from experiments.grouped_robot_independent.model import initializations, module_state
from experiments.grouped_robot_independent.protocol import CELLS, Config, source_hashes
from experiments.grouped_robot_independent.reference import reuse_sources
from experiments.grouped_robot_independent.runner import (
    Experiment as IndependentExperiment,
)
from experiments.grouped_robot_independent.training import Job, verify_pair
from experiments.grouped_robot_pilot.data import selected_indices
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load, verify_complete
from experiments.grouped_vqc_training_modes.protocol import state_hash


class Reference(IndependentExperiment):
    """The parent constructor writes artifacts and is deliberately never called."""

    def __init__(self, root: Path, runtime: dict, tick):  # pylint: disable=super-init-not-called
        self.output, self.runtime, self._tick = root.resolve(), runtime, tick
        self.config = Config(**read_json(self.output / "config.json"))
        self.config.source()
        self.manifest = read_json(self.output / "manifest.json")
        self.manifest_hash = sha256(self.output / "manifest.json")
        pilot_root = Path(self.config.reference)
        identity = {
            "config": self.config.to_dict(),
            "runtime": runtime,
            "sources": source_hashes(),
            "reference_manifest_sha256": sha256(pilot_root / "manifest.json"),
            "reference_result_lock_sha256": sha256(pilot_root / "result_lock.json"),
        }
        if any(self.manifest.get(k) != v for k, v in identity.items()):
            raise ValueError("Independent source config/runtime/code/reference changed")
        verify_files(self.output, self.manifest["artifacts"])
        locked = self.verify_result_lock()
        if locked is None:
            raise ValueError("Independent reference must be complete and locked")
        self.reference = PilotReference(pilot_root, runtime, tick)
        self.data = {}
        for role, limit in (
            ("train", self.config.train_limit),
            ("validation", self.config.val_limit),
        ):
            source = self.reference.data[role]
            index = selected_indices(source["concepts"].cpu(), limit, 0).cuda()
            self.data[role] = {k: v[index] for k, v in source.items()}
        self.data_hashes = {
            r: {k: array_hash(v.cpu().numpy()) for k, v in d.items()}
            for r, d in self.data.items()
        }
        expected_data = {
            "manifest_sha256": self.manifest_hash,
            "reference_data_lock_sha256": self.reference.data_hash,
            "data_hashes": self.data_hashes,
            "test_read": False,
        }
        if read_json(self.output / "data_reference.json") != expected_data:
            raise ValueError("Independent cached inputs/identities changed")
        self.data_hash = sha256(self.output / "data_reference.json")
        self.initial = load(self.output / "initialization.pt")
        if tree_hash(self.initial) != tree_hash(
            initializations(self.config.seed_values, self.reference.initial)
        ):
            raise ValueError("Independent initial parameters/RNG changed")
        self.reused, reuse_pins = reuse_sources(self)
        prior_pins = {**self.reference.pins, **reuse_pins}
        if read_json(self.output / "reference_lock.json") != {
            "manifest_sha256": self.manifest_hash,
            "artifacts": prior_pins,
            "reused": self.reused,
        }:
            raise ValueError("Independent historical reuse references changed")
        self.summary = read_json(self.output / "summary.json")
        if (
            self.summary["status"] != "complete"
            or self.summary["test_read"]
            or self.summary["test_evaluated"]
            or self.summary["seeds"] != list(self.config.seed_values)
        ):
            raise ValueError(
                "Independent summary is incomplete or uses another protocol"
            )
        self.training, self.evaluations = {}, {}
        for seed in self.config.seed_values:
            for cell in CELLS:
                tick("verifying_reference_training", source_cell=f"seed_{seed}/{cell}")
                self.training[seed, cell] = self.verify_training(seed, cell)
            verify_pair(
                self.training[seed, "independent"], self.training[seed, "no_feedback"]
            )
            # locked=True forbids missing evaluations from being written.
            for row in evaluate_seed(self, seed, locked=True):
                self.evaluations[seed, row["role"], row["cell"], row["condition"]] = row
        self.pins = {
            **prior_pins,
            **{
                str(self.output / name): sha256(self.output / name)
                for name in {*locked["artifacts"], "manifest.json", "result_lock.json"}
            },
        }

    def tick(self, status: str, **details) -> None:
        self._tick(status, **details)

    def verify_training(self, seed: int, cell: str) -> dict:
        job = Job(self, seed, cell)
        record = read_json(job.output / f"training/{cell}/training_reference.json")
        result = (
            read_json(job.checkpoint_path.parent / "result.json")
            if job.reused
            else verify_complete(job, cell)
        )
        verify_files(job.checkpoint_path.parent, result["artifacts"])
        value = load(job.checkpoint_path)
        initial = job.initial_for(cell)
        scope = (
            module_state(initial["model"], "frontend")
            if cell == "concept"
            else initial["model"]
        )
        expected = {
            "seed": seed,
            "cell": cell,
            "reused": job.reused,
            "reused_module": "frontend_only"
            if job.reused and cell == "concept"
            else "full_model",
            "checkpoint_path": str(job.checkpoint_path),
            "checkpoint_sha256": sha256(job.checkpoint_path),
            "model_sha256": state_hash(value["model"]),
            "frontend_sha256": state_hash(module_state(value["model"], "frontend")),
            "initial_scope_sha256": state_hash(scope),
            "epochs": value["epochs"],
            "epoch_offset": value["epoch_offset"],
            "global_step": value["progress"]["global_step"],
            "training_control": value["training_control"],
            "history_order_sha256": [
                r["order_sha256"] for r in value["progress"]["history"]
            ],
        }
        if record != expected or record not in self.summary["training"]:
            raise ValueError("Independent training record differs from checkpoint")
        if {int(v["step"]) for v in value["optimizer"]["state"].values()} != {
            record["global_step"]
        }:
            raise ValueError("Independent Adam budget changed")
        if (
            cell != "concept"
            and record["frontend_sha256"]
            != self.training[seed, "concept"]["frontend_sha256"]
        ):
            raise ValueError("Independent label routes changed the shared frontend")
        return record

    def verify_unchanged(self) -> None:
        verify_files(Path("/"), self.pins)
        verify_files(ROOT, self.manifest["sources"])
        self.reference.verify_unchanged()
