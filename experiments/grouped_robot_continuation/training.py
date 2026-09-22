"""Reuse pilot optimization with exact Adam imports and unchanged label orders."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    atomic_checkpoint,
    atomic_json,
    restore_rng,
    sha256,
)
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.runner import Experiment as PilotExperiment
from experiments.grouped_robot_pilot.training import Route as PilotRoute
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_robot_pilot.training import verify_complete as verify_pilot
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .protocol import tree_hash


class Job:
    """Adapt one phase to the unmodified pilot's training interface."""

    def __init__(self, parent: Any, arm: str, cell: str) -> None:
        self.parent, self.arm, self.cell = parent, arm, cell
        source = parent.reference
        self.output = parent.output / arm
        # Crucial: label shuffles remain source epochs 101..200 (then 201..400),
        # even when the new concept model has been trained for 300 epochs.
        self.config = replace(
            source.config,
            concept_epochs=parent.config.concept_epochs
            if cell == "concept"
            else source.config.concept_epochs,
            head_epochs=source.config.head_epochs
            if arm == "long_concept"
            else parent.config.head_epochs,
        )
        self.manifest_hash, self.data_hash = parent.manifest_hash, source.data_hash
        self.data = source.data
        self.continued = cell == "concept" or arm == "long_label"
        self.origin = (
            source.output / "training" / cell / "endpoint.pt"
            if self.continued
            else parent.output / "long_concept/training/concept/endpoint.pt"
        )

    @property
    def stop_requested(self) -> bool:
        return self.parent.stop_requested

    @property
    def state_cache(self) -> dict:
        return self.parent.state_cache

    def initial_for(self, cell: str) -> dict:
        source = self.parent.reference
        if self.continued:
            return source.initial_for(cell)
        return {"model": load(self.origin)["model"], "rng": source.initial["rng"]}

    make_model = staticmethod(PilotExperiment.make_model)

    def prepare_states(self, model) -> None:
        self.parent.prepare_states(model)

    def heartbeat(self, status: str, **details) -> None:
        self.parent.heartbeat(status, **details)


class Route(PilotRoute):
    """Only initialization/provenance differ; forward/backward/loop are inherited."""

    # Replace only initialization; the parent's constructor cannot import Adam.
    def __init__(self, job: Job) -> None:  # pylint: disable=super-init-not-called
        self.shared, self.config, self.cell = job, job.config, job.cell
        self.output = job.output / "training" / job.cell
        self.output.mkdir(parents=True, exist_ok=True)
        self.epochs = (
            self.config.concept_epochs
            if self.cell == "concept"
            else self.config.head_epochs
        )
        self.epoch_offset = 0 if self.cell == "concept" else self.config.concept_epochs
        initial = job.initial_for(self.cell)
        self.initial_hash = state_hash(initial["model"])
        self.origin_hash = sha256(job.origin)
        source = load(job.origin) if job.continued else None
        self.start_epoch = source["progress"]["completed_epoch"] if source else 0
        self.start_step = source["progress"]["global_step"] if source else 0
        self.source_seconds = source["training_seconds"] if source else 0.0
        self.model = job.make_model(initial["model"])
        self.model.frontend.requires_grad_(self.cell == "concept")
        self.model.label_head.requires_grad_(self.cell != "concept")
        self.frozen_module = "label_head" if self.cell == "concept" else "frontend"
        self.frozen_hash = state_hash(
            getattr(self.model, self.frozen_module).state_dict()
        )
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.config.learning_rate)
        self.progress = {
            "completed_epoch": 0,
            "global_step": 0,
            "order": None,
            "offset": 0,
            "loss_sums": [0.0],
            "history": [],
        }
        self.gradient_checks = {}
        self.training_seconds = 0.0
        restore_rng(initial["rng"])
        if not (self.output / "import_lock.json").exists():
            if (self.output / "resume.pt").exists():
                raise ValueError("Resume exists without an import lock")
            if source is not None:
                self.model.load_state_dict(source["model"])
                self.optimizer.load_state_dict(source["optimizer"])
                self.progress = deepcopy(source["progress"])
                self.training_seconds = source["training_seconds"]
                restore_rng(source["rng"])
            imported = self.checkpoint()
            atomic_checkpoint(self.output / "imported.pt", imported)
            atomic_json(
                self.output / "initialization.json",
                {
                    **self.identity(),
                    "fresh_adam": not job.continued,
                    "source_checkpoint": str(job.origin),
                    "starting_model_sha256": state_hash(imported["model"]),
                    "starting_optimizer_sha256": tree_hash(imported["optimizer"]),
                    "starting_rng_sha256": tree_hash(imported["rng"]),
                    "starting_progress_sha256": tree_hash(imported["progress"]),
                    "frontend_sha256": state_hash(self.model.frontend.state_dict()),
                    "head_sha256": state_hash(self.model.label_head.state_dict()),
                },
            )
            atomic_json(
                self.output / "import_lock.json",
                {
                    **self.identity(),
                    "artifacts": {
                        name: sha256(self.output / name)
                        for name in ("initialization.json", "imported.pt")
                    },
                },
            )
        lock = read_json(self.output / "import_lock.json")
        verify_files(self.output, lock["artifacts"])
        if any(lock.get(k) != v for k, v in self.identity().items()):
            raise ValueError("Imported checkpoint identity changed")
        imported = load(self.output / "imported.pt")
        if source is not None:
            for key in ("model", "optimizer", "progress", "rng", "training_seconds"):
                if tree_hash(imported[key]) != tree_hash(source[key]):
                    raise ValueError(f"Continuation did not preserve original {key}")
        elif (
            state_hash(imported["model"]) != self.initial_hash
            or imported["optimizer"]["state"]
            or imported["progress"]["global_step"] != 0
            or tree_hash(imported["rng"]) != tree_hash(initial["rng"])
        ):
            raise ValueError(
                "Fresh label phase must use original initialization/RNG and new Adam"
            )
        path = self.output / "resume.pt"
        checkpoint = load(path) if path.exists() else imported
        if any(checkpoint.get(k) != v for k, v in self.identity().items()):
            raise ValueError("Continuation resume identity changed")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.progress = checkpoint["progress"]
        self.gradient_checks = checkpoint["gradient_checks"]
        self.training_seconds = checkpoint["training_seconds"]
        restore_rng(checkpoint["rng"])
        validate_progress(self, imported)
        self.verify_frozen()
        if not path.exists():
            self.save()
        if self.cell != "concept":
            job.prepare_states(self.model)
        torch.cuda.reset_peak_memory_stats()

    def identity(self) -> dict:
        return {
            **super().identity(),
            "arm": self.shared.arm,
            "source_checkpoint_sha256": self.origin_hash,
            "start_epoch": self.start_epoch,
            "start_step": self.start_step,
            "fresh_adam": not self.shared.continued,
        }

    def run(self, max_steps: int | None = None) -> dict:
        result = super().run(max_steps)
        result.update(
            new_updates=self.progress["global_step"] - self.start_step,
            additional_training_seconds=self.training_seconds - self.source_seconds,
        )
        atomic_json(self.output / "result.json", result)
        verify_job(self.shared, require_lock=False)
        names = ("result.json", "import_lock.json", "imported.pt", *result["artifacts"])
        atomic_json(
            self.output / "completion_lock.json",
            {
                **self.identity(),
                "artifacts": {name: sha256(self.output / name) for name in names},
            },
        )
        return result


def validate_progress(route: Route, imported: dict) -> None:
    p = route.progress
    n = len(route.shared.data["train"]["labels"])
    batch = route.config.batch_size
    per_epoch = (n + batch - 1) // batch
    offset = p["offset"]
    if (
        not route.start_epoch <= p["completed_epoch"] <= route.epochs
        or len(p["history"]) != p["completed_epoch"]
        or not 0 <= offset <= n
        or (offset != n and offset % batch)
        or p["global_step"]
        != per_epoch * p["completed_epoch"] + (offset + batch - 1) // batch
        or (p["order"] is None and offset != 0)
        or tree_hash(p["history"][: route.start_epoch])
        != tree_hash(imported["progress"]["history"])
    ):
        raise ValueError("Invalid continuation epoch/minibatch/history state")
    steps = {int(v["step"]) for v in route.optimizer.state.values()}
    if p["order"] is not None and not torch.equal(
        p["order"],
        epoch_order(
            n, route.config.seed, p["completed_epoch"] + 1 + route.epoch_offset
        ),
    ):
        raise ValueError("Saved minibatch order differs from the fixed phase order")
    if p["completed_epoch"] == route.epochs and (offset or p["order"] is not None):
        raise ValueError("Completed phase is not at an epoch boundary")
    if steps != ({p["global_step"]} if p["global_step"] else set()):
        raise ValueError("Adam step counters differ from the continuation budget")
    if any(g["lr"] != route.config.learning_rate for g in route.optimizer.param_groups):
        raise ValueError("Learning rate changed during a duration-only experiment")


def verify_job(job: Job, *, require_lock: bool = True) -> dict:
    directory = job.output / "training" / job.cell
    if require_lock:
        lock = read_json(directory / "completion_lock.json")
        verify_files(directory, lock["artifacts"])
    result = verify_pilot(job, job.cell)
    imported = load(directory / "imported.pt")
    origin = load(job.origin)
    init = read_json(directory / "initialization.json")
    verify_files(directory, read_json(directory / "import_lock.json")["artifacts"])
    start = origin["progress"]["global_step"] if job.continued else 0
    if (
        result["arm"] != job.arm
        or result["source_checkpoint_sha256"] != sha256(job.origin)
        or result["fresh_adam"] == job.continued
        or result["start_step"] != start
        or result["new_updates"] != result["global_step"] - start
        or init["starting_optimizer_sha256"] != tree_hash(imported["optimizer"])
    ):
        raise ValueError("Completed continuation lineage changed")
    if job.continued:
        for key in ("model", "optimizer", "progress", "rng", "training_seconds"):
            if tree_hash(imported[key]) != tree_hash(origin[key]):
                raise ValueError(f"Imported original {key} differs")
        history = read_json(directory / "history.json")
        if tree_hash(history[: origin["progress"]["completed_epoch"]]) != tree_hash(
            origin["progress"]["history"]
        ):
            raise ValueError("Continuation rewrote earlier history")
    elif imported["optimizer"]["state"] or state_hash(imported["model"]) != state_hash(
        job.initial_for(job.cell)["model"]
    ):
        raise ValueError("Fresh label phase initialization differs")
    endpoint = load(directory / "endpoint.pt")
    if {int(v["step"]) for v in endpoint["optimizer"]["state"].values()} != {
        result["global_step"]
    }:
        raise ValueError("Completed Adam counters differ")
    return result
