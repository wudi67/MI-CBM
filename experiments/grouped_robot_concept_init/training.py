"""Use the original concept NLL/Adam loop, with isolated cells and recovery checks."""

import math
from dataclasses import replace

import torch

from experiments.grouped_dynamic_vqc.runtime import array_hash, atomic_json
from experiments.grouped_robot_pilot.training import Route as PilotRoute
from experiments.grouped_robot_pilot.training import load, verify_complete
from experiments.grouped_vqc_training_modes.protocol import epoch_order

from .model import make_model
from .protocol import cell_name


class Job:
    def __init__(self, parent, method: str, index: int):
        self.parent = parent
        self.name = cell_name(method, index)
        self.output = parent.output / self.name
        self.config = replace(
            parent.reference.config,
            concept_epochs=parent.config.concept_epochs,
            checkpoint_steps=parent.config.checkpoint_steps,
        )
        self.data, self.data_hash = parent.data, parent.data_hash
        self.manifest_hash = parent.manifest_hash

    @property
    def stop_requested(self) -> bool:
        return self.parent.control["stop"]

    def initial_for(self, cell: str) -> dict:
        if cell != "concept":
            raise ValueError("This comparison trains only the concept circuit")
        return self.parent.initial[self.name]

    make_model = staticmethod(make_model)

    def heartbeat(self, status: str, **details) -> None:
        self.parent.heartbeat(status, **details)

    @property
    def checkpoint_path(self):
        return self.output / "training/concept/endpoint.pt"


class Route(PilotRoute):
    def __init__(self, shared: Job):
        super().__init__(shared, "concept")
        self.verify_progress()

    def verify_progress(self) -> None:
        count = len(self.shared.data["train"]["angles"])
        p = self.progress
        completed, offset = p["completed_epoch"], p["offset"]
        if not 0 <= completed <= self.epochs or len(p["history"]) != completed:
            raise ValueError("Resume epoch/history changed")
        if not 0 <= offset <= count or (
            offset != count and offset % self.config.batch_size
        ):
            raise ValueError("Resume minibatch position changed")
        expected_steps = completed * math.ceil(
            count / self.config.batch_size
        ) + math.ceil(offset / self.config.batch_size)
        if p["global_step"] != expected_steps:
            raise ValueError("Resume optimizer step/position changed")
        if p["order"] is None:
            if offset != 0 or p["loss_sums"] != [0.0]:
                raise ValueError("Resume epoch boundary changed")
        elif completed >= self.epochs or not torch.equal(
            p["order"], epoch_order(count, self.config.seed, completed + 1)
        ):
            raise ValueError("Resume sample order changed")
        for epoch, row in enumerate(p["history"], 1):
            if row["order_epoch"] != epoch or row["order_sha256"] != array_hash(
                epoch_order(count, self.config.seed, epoch).numpy()
            ):
                raise ValueError("Resume historical sample order changed")
        steps = {int(v["step"]) for v in self.optimizer.state_dict()["state"].values()}
        if steps != ({expected_steps} if expected_steps else set()):
            raise ValueError("Resume Adam step changed")

    def check_gradients(self) -> None:
        super().check_gradients()
        self.gradient_checks["front_layer_gradient_l2"] = {
            name: [float(layer.norm()) for layer in tensor.grad]
            for name, tensor in self.model.frontend.named_parameters()
            if tensor.grad is not None
        }
        self.gradient_checks["frontend_parameters"] = sum(
            p.numel() for p in self.parameters
        )
        atomic_json(self.output / "gradient_checks.json", self.gradient_checks)


def verify_job(job: Job) -> dict:
    result = verify_complete(job, "concept")
    checkpoint = load(job.checkpoint_path)
    if {int(v["step"]) for v in checkpoint["optimizer"]["state"].values()} != {
        result["global_step"]
    }:
        raise ValueError("Completed Adam step differs from training budget")
    gradients = checkpoint["gradient_checks"]
    if (
        gradients["frontend_parameters"] != 240
        or gradients["gradient_l2"]["label_head"] is not None
    ):
        raise ValueError("Unexpected active parameters or label-circuit gradients")
    return result
