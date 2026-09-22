"""Adapters around the original TorchQuantum NLL/BCE training and Adam recovery."""

import math
from dataclasses import replace

import torch

from experiments.grouped_dynamic_vqc.runtime import array_hash, atomic_json, sha256
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import Route as PilotRoute
from experiments.grouped_robot_pilot.training import load, verify_complete
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .model import label_initial, make_model, module_state


class Job:
    def __init__(self, parent, seed: int, cell: str):
        self.parent, self.seed, self.cell = parent, seed, cell
        self.name = f"seed_{seed}/{cell}"
        self.output = parent.output / f"seed_{seed}"
        self.config = replace(
            parent.reference.config,
            seed=seed,
            # The inherited loop uses this field as the label shuffle offset.
            # Actual concept training always uses the declared 300-epoch budget.
            concept_epochs=(
                parent.config.concept_epochs
                if cell == "concept"
                else parent.config.head_order_offset
            ),
            head_epochs=parent.config.head_epochs,
            checkpoint_steps=parent.config.checkpoint_steps,
        )
        self.data, self.data_hash = parent.data, parent.data_hash
        self.manifest_hash = parent.manifest_hash

    @property
    def reused(self) -> bool:
        return f"{self.seed}/{self.cell}" in self.parent.reused

    @property
    def checkpoint_path(self):
        return self.parent.checkpoint_path(self.seed, self.cell)

    @property
    def stop_requested(self) -> bool:
        return self.parent.control["stop"]

    def initial_for(self, cell: str) -> dict:
        if cell != self.cell:
            raise ValueError("Job route mismatch")
        initial = self.parent.initial[str(self.seed)]
        if cell == "concept":
            return initial
        frontend = module_state(
            load(self.parent.checkpoint_path(self.seed, "concept"))["model"], "frontend"
        )
        return label_initial(initial, frontend)

    make_model = staticmethod(make_model)

    def prepare_states(self, model) -> None:
        self.parent.prepare_states(model)

    @property
    def state_cache(self) -> dict:
        return self.parent.state_cache

    def heartbeat(self, status: str, **details) -> None:
        self.parent.heartbeat(status, **details)


class Route(PilotRoute):
    def __init__(self, shared: Job):
        if shared.reused:
            raise ValueError("Reused checkpoints are read-only")
        super().__init__(shared, shared.cell)
        self.verify_progress()

    def identity(self) -> dict:
        return {**super().identity(), "seed": self.shared.seed, "head_layers": 5}

    def verify_progress(self) -> None:
        count, p = len(self.shared.data["train"]["angles"]), self.progress
        completed, offset = p["completed_epoch"], p["offset"]
        if not 0 <= completed <= self.epochs or len(p["history"]) != completed:
            raise ValueError("Resume epoch/history changed")
        if not 0 <= offset <= count or (
            offset != count and offset % self.config.batch_size
        ):
            raise ValueError("Resume minibatch position changed")
        expected = completed * math.ceil(count / self.config.batch_size) + math.ceil(
            offset / self.config.batch_size
        )
        if p["global_step"] != expected:
            raise ValueError("Resume optimizer step/position changed")
        if p["order"] is None:
            if offset or p["loss_sums"] != [0.0]:
                raise ValueError("Resume epoch boundary changed")
        elif completed >= self.epochs or not torch.equal(
            p["order"],
            epoch_order(count, self.config.seed, completed + 1 + self.epoch_offset),
        ):
            raise ValueError("Resume sample order changed")
        for epoch, row in enumerate(p["history"], 1):
            if row["order_epoch"] != epoch + self.epoch_offset or row[
                "order_sha256"
            ] != array_hash(
                epoch_order(count, self.config.seed, epoch + self.epoch_offset).numpy()
            ):
                raise ValueError("Resume historical sample order changed")
        steps = {int(v["step"]) for v in self.optimizer.state_dict()["state"].values()}
        if steps != ({expected} if expected else set()):
            raise ValueError("Resume Adam step changed")

    def check_gradients(self) -> None:
        super().check_gradients()
        active = "frontend" if self.cell == "concept" else "label_head"
        self.gradient_checks["active_parameters"] = sum(
            p.numel() for p in self.parameters
        )
        self.gradient_checks["layer_gradient_l2"] = {
            name: [float(layer.norm()) for layer in tensor.grad]
            for name, tensor in getattr(self.model, active).named_parameters()
            if tensor.grad is not None and tensor.ndim > 1
        }
        atomic_json(self.output / "gradient_checks.json", self.gradient_checks)


def verify_job(job: Job) -> dict:
    if not job.reused:
        result = verify_complete(job, job.cell)
        checkpoint = load(job.checkpoint_path)
        if checkpoint["seed"] != job.seed or checkpoint["head_layers"] != 5:
            raise ValueError("Completed checkpoint seed/architecture changed")
        if {int(v["step"]) for v in checkpoint["optimizer"]["state"].values()} != {
            result["global_step"]
        }:
            raise ValueError("Completed Adam steps differ from training budget")
        if checkpoint["gradient_checks"]["active_parameters"] != (
            240 if job.cell == "concept" else 112
        ):
            raise ValueError("Unexpected active parameter count")
    else:
        result = read_json(job.checkpoint_path.parent / "result.json")
        verify_files(job.checkpoint_path.parent, result["artifacts"])
    checkpoint = load(job.checkpoint_path)
    initial = job.initial_for(job.cell)
    # Concept reuse intentionally imports only the trained frontend of the old
    # 4+1 circuit. Its frozen one-layer head is never imported into the 4+5 model.
    initial_scope = (
        module_state(initial["model"], "frontend")
        if job.cell == "concept"
        else initial["model"]
    )
    record = {
        "seed": job.seed,
        "cell": job.cell,
        "reused": job.reused,
        "reused_module": "frontend_only"
        if job.reused and job.cell == "concept"
        else "full_model",
        "checkpoint_path": str(job.checkpoint_path),
        "checkpoint_sha256": sha256(job.checkpoint_path),
        "model_sha256": result["model_sha256"],
        "frontend_sha256": result["frontend_sha256"],
        "initial_scope_sha256": state_hash(initial_scope),
        "epochs": result["epochs"],
        "epoch_offset": result["epoch_offset"],
        "global_step": result["global_step"],
        "training_control": result["training_control"],
        "history_order_sha256": [
            r["order_sha256"] for r in checkpoint["progress"]["history"]
        ],
    }
    relative = f"seed_{job.seed}/training/{job.cell}/training_reference.json"
    job.parent.save_once(relative, record)
    return record


def verify_pair(left: dict, right: dict) -> None:
    for key in (
        "seed",
        "frontend_sha256",
        "initial_scope_sha256",
        "global_step",
        "epochs",
        "epoch_offset",
        "history_order_sha256",
    ):
        if left[key] != right[key]:
            raise ValueError(f"Label routes are not paired: {key}")
