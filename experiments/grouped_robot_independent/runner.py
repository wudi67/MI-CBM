"""Five complete training seeds, paired label routes, and immutable provenance."""

import os
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    array_hash,
    atomic_checkpoint,
    atomic_json,
    cuda_runtime,
    report,
    sha256,
    utc_now,
)
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_continuation.reference import Reference
from experiments.grouped_robot_pilot.data import selected_indices
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .evaluation import evaluate_seed
from .model import initializations
from .protocol import CELLS, SIGMA, Config, check_output, source_hashes
from .reference import reuse_sources
from .training import Job, Route, verify_job, verify_pair


class Experiment:
    def __init__(
        self, config: Config, output: Path, resume: bool = False, control=None
    ):
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        self.control = {"stop": False} if control is None else control
        self.runtime = cuda_runtime(0)
        self.stage, self.cell, self.details = "preparation", "reference", {}
        self.reused: dict = {}
        self.state_cache: dict = {}
        self.cached_frontend = None
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
            self.manifest = {
                "schema": "grouped_robot_independent.v1",
                "created_at": utc_now(),
                **identity,
                "artifacts": {"config.json": sha256(self.output / "config.json")},
                "architecture": {
                    "frontend_layers": 4,
                    "frontend_parameters": 240,
                    "head_layers": 5,
                    "head_parameters": 112,
                    "total_parameters": 352,
                    "uploaded_qubits": 10,
                    "concept_qubits": 5,
                    "physical_qubits_with_readout": 11,
                    "frontend_initialization": "uniform [0, pi]",
                    "head_sigma": SIGMA,
                    "readout_ry_initial_shift": float(torch.pi / 2),
                    "data_reuploading_every_front_layer": True,
                },
                "training": (
                    "Concept NLL then frozen frontend; Independent uses true X "
                    "controls; paired baseline uses zero X controls"
                ),
                "normal_inference": (
                    "Exact Born-weighted 32-branch measurement-feedback inference; "
                    "no MAP substitution or postselection"
                ),
                "intervention": (
                    "Replace all five classical X controls; preserve Born weights "
                    "and post-measurement states"
                ),
                "ablation_scope": (
                    "Measurement-record feedback; both routes retain measurement. "
                    "Not a measurement-versus-no-measurement claim."
                ),
                "selection": (
                    "Fixed final epochs, all predefined seeds; "
                    "no best seed/epoch selection"
                ),
                "head_order_offset_note": (
                    "100 selects legacy shuffle streams 101..400; "
                    "actual concept training is 300 epochs"
                ),
                "test_read": False,
                "test_evaluated": False,
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        self.tick("verifying_reference")
        self.reference = Reference(root, self.runtime, self.tick)
        self.data = {}
        for role, limit in (
            ("train", config.train_limit),
            ("validation", config.val_limit),
        ):
            source = self.reference.data[role]
            index = selected_indices(source["concepts"].cpu(), limit, 0).cuda()
            self.data[role] = {k: v[index] for k, v in source.items()}
        self.data_hashes = {
            r: {k: array_hash(v.cpu().numpy()) for k, v in d.items()}
            for r, d in self.data.items()
        }
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
        path = self.output / "initialization.pt"
        expected_initial = initializations(config.seed_values, self.reference.initial)
        if not path.exists():
            atomic_checkpoint(path, expected_initial)
        self.initial = load(path)
        if tree_hash(self.initial) != tree_hash(expected_initial):
            raise ValueError("Initial model/RNG changed")
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
        self.reused, reuse_pins = reuse_sources(self)
        self.pins = {**self.reference.pins, **reuse_pins}
        self.save_once(
            "reference_lock.json",
            {
                "manifest_sha256": self.manifest_hash,
                "artifacts": self.pins,
                "reused": self.reused,
            },
        )
        self.verify_result_lock()
        report(
            f"[green]CUDA ready:[/green] {self.runtime['device']}; "
            f"{len(config.seed_values)} seeds, 4+5 layers / 352 parameters; "
            f"{len(self.reused)} historical endpoints admitted"
        )

    def save_once(self, name: str, value: dict) -> None:
        path = self.output / name
        if path.exists():
            if read_json(path) != value:
                raise ValueError(f"Existing artifact changed: {name}")
        else:
            atomic_json(path, value)

    def checkpoint_path(self, seed: int, cell: str) -> Path:
        return Path(
            self.reused.get(
                f"{seed}/{cell}",
                self.output / f"seed_{seed}/training/{cell}/endpoint.pt",
            )
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
                "total_cells": len(self.config.seed_values) * 3,
                "completed_conditions": len(
                    list(self.output.glob("seed_*/*/*/*/evaluation_lock.json"))
                ),
                "total_conditions": len(self.config.seed_values) * 6,
                "test_evaluated": False,
                **self.details,
            },
        )

    def tick(self, status: str, **details) -> None:
        self.heartbeat(status, **details)
        if self.control["stop"]:
            raise InterruptedError("Pause requested")

    @torch.no_grad()
    def prepare_states(self, model) -> None:
        fingerprint = state_hash(model.frontend.state_dict())
        if self.cached_frontend == fingerprint:
            return
        self.state_cache.clear()
        batch = self.reference.config.eval_batch_size
        model.eval()
        for role, data in self.data.items():
            chunks = []
            for start in range(0, len(data["angles"]), batch):
                chunks.append(
                    model.frontend(data["angles"][start : start + batch]).detach()
                )
                self.tick(
                    "caching_frontend",
                    role=role,
                    offset=min(start + batch, len(data["angles"])),
                )
            self.state_cache[role] = torch.cat(chunks)
        self.cached_frontend = fingerprint

    def verify_result_lock(self) -> dict | None:
        path = self.output / "result_lock.json"
        if not path.exists():
            return None
        value = read_json(path)
        if value["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Result lock manifest changed")
        verify_files(self.output, value["artifacts"])
        return value

    def verify_unchanged(self) -> None:
        self.reference.verify_unchanged()
        verify_files(Path("/"), self.pins)
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
        seed_training = {}
        for cell in CELLS:
            job = Job(shared, seed, cell)
            shared.cell, shared.stage, shared.details = job.name, "training", {}
            shared.tick("preparing_cell")
            if (
                not job.reused
                and not (job.output / f"training/{cell}/result.json").exists()
            ):
                if existing:
                    raise ValueError("Locked training endpoint is missing")
                if remaining is not None and remaining <= 0:
                    raise InterruptedError("Requested new-update budget reached")
                route = Route(job)
                before = route.progress["global_step"]
                route.run(None if remaining is None else before + remaining)
                if remaining is not None:
                    remaining -= route.progress["global_step"] - before
                del route
            record = verify_job(job)
            training.append(record)
            seed_training[cell] = record
        verify_pair(seed_training["independent"], seed_training["no_feedback"])
        shared.stage, shared.details = "evaluation", {}
        evaluations.extend(evaluate_seed(shared, seed, existing))
        shared.state_cache.clear()
        shared.cached_frontend = None
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
        "[green]Robot Independent experiment complete:[/green] "
        f"{shared.output / 'summary.md'}"
    )
