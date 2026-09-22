"""Fresh Standard/Joint routes, read-only comparisons and final validation lock."""

import os
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    atomic_checkpoint,
    atomic_json,
    cuda_runtime,
    report,
    sha256,
    utc_now,
)
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_independent.runner import (
    Experiment as IndependentExperiment,
)
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .evaluation import evaluate_seed
from .protocol import CELLS, CONDITIONS, Config, check_output, source_hashes
from .reference import Reference
from .training import Job, Route, verify_job, verify_pair


class Experiment:
    save_once = IndependentExperiment.save_once
    verify_result_lock = IndependentExperiment.verify_result_lock
    tick = IndependentExperiment.tick

    def __init__(
        self, config: Config, output: Path, resume: bool = False, control=None
    ):
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        _, _, self.pilot_config = config.sources()
        self.control = {"stop": False} if control is None else control
        self.runtime = cuda_runtime(0)
        self.stage, self.cell, self.details = "preparation", "reference", {}
        self.output.mkdir(parents=True, exist_ok=True)
        root = Path(config.reference)
        identity = {
            "config": config.to_dict(),
            "runtime": self.runtime,
            "sources": source_hashes(),
            "reference_manifest_sha256": sha256(root / "manifest.json"),
            "reference_result_lock_sha256": sha256(root / "result_lock.json"),
        }
        self.manifest: dict
        if resume:
            self.manifest = read_json(self.output / "manifest.json")
            if any(self.manifest.get(k) != v for k, v in identity.items()):
                raise ValueError(
                    "Resume requires unchanged config, source, reference and runtime"
                )
            verify_files(self.output, self.manifest["artifacts"])
        else:
            if any(
                (self.output / n).exists() for n in ("manifest.json", "config.json")
            ):
                raise FileExistsError("Output exists; use --resume or a fresh --out")
            atomic_json(self.output / "config.json", config.to_dict())
            source_manifest = read_json(root / "manifest.json")
            self.manifest = {
                "schema": "grouped_robot_four_modes.v1",
                "created_at": utc_now(),
                **identity,
                "artifacts": {"config.json": sha256(self.output / "config.json")},
                "architecture": source_manifest["architecture"],
                "training": {
                    "standard": (
                        "label BCE only; both modules train; measured X controls"
                    ),
                    "joint": (
                        "concept joint-record NLL + label BCE; both modules train; "
                        "measured X controls"
                    ),
                    "joint_no_feedback": (
                        "same Joint losses; both modules train; zero X controls; "
                        "measurement retained"
                    ),
                },
                "pairing": (
                    "Same original uniform/Gaussian parameters, RNG, data, minibatch "
                    "order, Adam and update count within each seed"
                ),
                "budget": (
                    "Full-circuit epochs match concept + label total optimizer "
                    "updates; not equal per-module updates or runtime"
                ),
                "normal_inference": (
                    "Exact Born-weighted 32 branches, no MAP substitution or "
                    "postselection"
                ),
                "intervention": source_manifest["intervention"],
                "ablation_scope": source_manifest["ablation_scope"],
                "standard_semantics": (
                    "Unsupervised intermediate bits; concept metrics diagnostic "
                    "only; no concept correction"
                ),
                "selection": (
                    "All declared seeds at the fixed final epoch; "
                    "no best-seed or best-epoch selection"
                ),
                "test_read": False,
                "test_evaluated": False,
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        self.tick("verifying_reference")
        self.reference = Reference(root, self.runtime, self.tick)
        self.save_once(
            "reference_lock.json",
            {
                "manifest_sha256": self.manifest_hash,
                "artifacts": self.reference.pins,
            },
        )
        self.data = self.reference.data
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
        # Independent.initial stores ORIGINAL frontends; Sequential.initial stores
        # trained frontends and must never be used to warm-start these new modes.
        expected = {
            str(s): self.reference.reference.initial[str(s)] for s in config.seed_values
        }
        path = self.output / "initialization.pt"
        if not path.exists():
            atomic_checkpoint(path, expected)
        self.initial = load(path)
        if tree_hash(self.initial) != tree_hash(expected):
            raise ValueError("Full-circuit original initialization/RNG changed")
        self.save_once(
            "initialization_lock.json",
            {
                "manifest_sha256": self.manifest_hash,
                "artifacts": {"initialization.pt": sha256(path)},
                "model_hashes": {
                    k: state_hash(v["model"]) for k, v in self.initial.items()
                },
            },
        )
        self.verify_result_lock()
        report(
            f"[green]CUDA ready:[/green] {self.runtime['device']}; "
            f"{len(config.seed_values) * len(CELLS)} new full-circuit trainings; "
            "4+5 layers / 352 active parameters"
        )

    def heartbeat(self, status: str, **details) -> None:
        self.details.update(details)
        atomic_json(
            self.output / "heartbeat.json",
            {
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "status": status,
                "stage": self.stage,
                "cell": self.cell,
                "device": self.runtime["device"],
                "completed_cells": len(
                    list(self.output.glob("seed_*/training/*/training_reference.json"))
                ),
                "total_cells": len(self.config.seed_values) * len(CELLS),
                "completed_conditions": sum(
                    len(list(self.output.glob(f"seed_*/*/*/*/{name}")))
                    for name in ("evaluation_lock.json", "evaluation_reference.json")
                ),
                "total_conditions": len(self.config.seed_values) * len(CONDITIONS) * 2,
                "test_evaluated": False,
                **self.details,
            },
        )

    def verify_unchanged(self) -> None:
        self.reference.verify_unchanged()
        verify_files(ROOT, self.manifest["sources"])


def run_experiment(
    shared: Experiment, max_steps: int | None = None, preflight_only: bool = False
) -> None:
    if preflight_only:
        shared.heartbeat("paused", reason="Preflight complete; resume to train")
        return
    existing = shared.verify_result_lock() is not None
    training, evaluations, remaining = [], [], max_steps
    for seed in shared.config.seed_values:
        paired = []
        for cell in CELLS:
            job = Job(shared, seed, cell)
            shared.cell, shared.stage, shared.details = job.name, "training", {}
            shared.tick("preparing_cell")
            if not (job.checkpoint_path.parent / "result.json").exists():
                if existing:
                    raise ValueError("Locked full-circuit training is missing")
                if remaining is not None and remaining <= 0:
                    raise InterruptedError("Requested new-update budget reached")
                route = Route(job)
                before = route.progress["global_step"]
                route.run(None if remaining is None else before + remaining)
                if remaining is not None:
                    remaining -= route.progress["global_step"] - before
                del route
            record = verify_job(job)
            paired.append(record)
            training.append(record)
        for record in paired[1:]:
            verify_pair(paired[0], record)
        shared.stage, shared.details = "evaluation", {}
        evaluations.extend(evaluate_seed(shared, seed, existing))
    shared.stage, shared.cell, shared.details = "summary", "all", {}
    shared.tick("verifying_sources")
    shared.verify_unchanged()
    if not existing:
        from .results import write_results  # pylint: disable=import-outside-toplevel

        write_results(shared, evaluations, training)
    shared.verify_result_lock()
    shared.stage = "complete"
    shared.heartbeat("complete")
    report(
        "[green]Robot four-mode comparison complete:[/green] "
        f"{shared.output / 'summary.md'}"
    )
