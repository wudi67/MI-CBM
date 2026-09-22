"""Adapt the original Independent training loop; preserve Adam and sample order."""

import math
from dataclasses import replace

import torch

from experiments.grouped_dynamic_vqc.runtime import atomic_json, sha256
from experiments.grouped_robot_pilot.evaluation import evaluate
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import Route as PilotRoute
from experiments.grouped_robot_pilot.training import load, verify_complete
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .model import make_model
from .protocol import cell_name


class Job:
    def __init__(self, parent, depth: int, index: int) -> None:
        self.parent, self.depth, self.index = parent, depth, index
        self.name = cell_name(depth, index)
        self.output = parent.output / self.name
        # Shuffle seed and offset are historical frontend seed 0 / epoch 100.
        # Initialization index never changes the sample permutation.
        self.config = replace(
            parent.reference.reference.config,
            head_epochs=parent.config.head_epochs,
            checkpoint_steps=parent.config.checkpoint_steps,
        )
        self.data, self.state_cache = parent.data, parent.state_cache
        self.data_hash, self.manifest_hash = parent.data_hash, parent.manifest_hash
        self.reused = depth == 1 and index == 0 and parent.reuse_a0

    @property
    def stop_requested(self) -> bool:
        return self.parent.control["stop"]

    def initial_for(self, cell: str) -> dict:
        if cell != "independent":
            raise ValueError("Only Independent training is supported")
        return self.parent.initial[self.name]

    make_model = staticmethod(make_model)

    def prepare_states(self, model) -> None:
        if (
            state_hash(model.frontend.state_dict())
            != self.parent.reference.frontend_hash
        ):
            raise ValueError("Job is using a different frontend")

    def heartbeat(self, status: str, **details) -> None:
        self.parent.heartbeat(status, **details)

    @property
    def checkpoint_path(self):
        if self.reused:
            return self.parent.reference.checkpoint
        return self.output / "training/independent/endpoint.pt"

    def identity(self) -> dict:
        return {
            "manifest_sha256": self.manifest_hash,
            "cell_name": self.name,
            "head_layers": self.depth,
            "initialization_index": self.index,
            "data_lock_sha256": self.data_hash,
            "reused_historical_training": self.reused,
            "epochs": self.config.head_epochs,
            "frontend_sha256": self.parent.reference.frontend_hash,
        }


class Route(PilotRoute):
    """The inherited step/run methods perform the unchanged measurement/X/BCE loop."""

    def __init__(self, job: Job) -> None:
        if job.reused:
            raise ValueError("Historical A0 must be referenced, never retrained here")
        super().__init__(job, "independent")
        count = len(job.data["train"]["angles"])
        progress = self.progress
        expected = progress["completed_epoch"] * math.ceil(
            count / job.config.batch_size
        )
        expected += math.ceil(progress["offset"] / job.config.batch_size)
        if progress["global_step"] != expected or not 0 <= progress["offset"] <= count:
            raise ValueError("Resumed optimizer/sample position is inconsistent")
        if progress["order"] is not None:
            order = epoch_order(
                count,
                job.config.seed,
                progress["completed_epoch"] + 1 + self.epoch_offset,
            )
            if not torch.equal(progress["order"], order):
                raise ValueError("Resumed sample order changed")
        elif progress["offset"] != 0:
            raise ValueError("Resumed sample offset has no order")
        if len(progress["history"]) != progress["completed_epoch"]:
            raise ValueError("Resumed history does not match completed epochs")
        if progress["global_step"] and {
            int(v["step"]) for v in self.optimizer.state.values()
        } != {expected}:
            raise ValueError("Adam step count does not match resumed progress")

    def identity(self) -> dict:
        return {**super().identity(), **self.shared.identity()}

    def check_gradients(self) -> None:
        super().check_gradients()
        norms = []
        for layer in range(self.shared.depth):
            squared = sum(
                float(getattr(self.model.label_head, name).grad[layer].square().sum())
                for name in ("a", "b", "gamma", "kappa")
            )
            norms.append(math.sqrt(squared))
        if any(value <= 0 for value in norms):
            raise ValueError("An active label layer has no gradient")
        self.gradient_checks["label_layer_gradient_l2"] = norms
        self.gradient_checks["label_parameters"] = sum(
            p.numel() for p in self.parameters
        )
        atomic_json(self.output / "gradient_checks.json", self.gradient_checks)

    def save(self) -> None:
        epoch = self.progress["completed_epoch"]
        if (
            epoch > 0
            and self.progress["order"] is None
            and (
                epoch % self.shared.parent.config.diagnostic_every == 0
                or epoch == self.epochs
            )
            and "train_true" not in self.progress["history"][-1]
        ):
            for role, key in (
                ("train", "train_true"),
                ("validation", "validation_true"),
            ):
                self.heartbeat("diagnostic_true_controls", role=role)
                metric, _ = evaluate(
                    self.model,
                    self.shared.data[role],
                    self.config.eval_batch_size,
                    mask=31,
                    states=self.shared.state_cache[role],
                )
                self.progress["history"][-1][key] = metric
                self.heartbeat("diagnostic_true_controls", **{key: metric})
        super().save()


def verify_job(job: Job) -> dict:
    lock = read_json(job.output / "training_lock.json")
    if any(lock.get(k) != v for k, v in job.identity().items()):
        raise ValueError("Completed job identity changed")
    verify_files(job.output, lock["artifacts"])
    if job.reused:
        record = read_json(job.output / "training_reference.json")
        if record["source_checkpoint"] != str(job.checkpoint_path) or record[
            "checkpoint_sha256"
        ] != sha256(job.checkpoint_path):
            raise ValueError("Historical checkpoint reference changed")
    else:
        verify_complete(job, "independent")
    checkpoint = load(job.checkpoint_path)
    weights = checkpoint["model"]
    steps = job.config.head_epochs * math.ceil(
        len(job.data["train"]["angles"]) / job.config.batch_size
    )
    count = sum(v.numel() for k, v in weights.items() if k.startswith("label_head."))
    if (
        checkpoint["progress"]["global_step"] != steps
        or checkpoint["progress"]["completed_epoch"] != job.config.head_epochs
        or count != 22 * job.depth + 2
        or {int(v["step"]) for v in checkpoint["optimizer"]["state"].values()}
        != {steps}
    ):
        raise ValueError("Endpoint depth, epoch budget or Adam count changed")
    front = {
        k.removeprefix("frontend."): v
        for k, v in weights.items()
        if k.startswith("frontend.")
    }
    if state_hash(front) != job.parent.reference.frontend_hash:
        raise ValueError("Endpoint changed the frozen frontend")
    return {
        **job.identity(),
        "label_parameters": count,
        "global_step": steps,
        "checkpoint_sha256": sha256(job.checkpoint_path),
        "model_sha256": state_hash(weights),
        "training_seconds": checkpoint["training_seconds"],
        "test_evaluated": False,
    }


def lock_training(job: Job) -> None:
    if job.reused:
        checkpoint = load(job.checkpoint_path)
        if (
            state_hash(job.initial_for("independent")["model"])
            != checkpoint["initial_model_sha256"]
        ):
            raise ValueError("A0 reference has a different initial model")
        atomic_json(
            job.output / "training_reference.json",
            {
                **job.identity(),
                "source_checkpoint": str(job.checkpoint_path),
                "checkpoint_sha256": sha256(job.checkpoint_path),
                "new_optimizer_updates": 0,
            },
        )
        names = ["training_reference.json"]
    else:
        verify_complete(job, "independent")
        names = [
            str(p.relative_to(job.output))
            for p in (job.output / "training/independent").iterdir()
            if p.is_file()
        ]
    atomic_json(
        job.output / "training_lock.json",
        {
            **job.identity(),
            "artifacts": {n: sha256(job.output / n) for n in names},
        },
    )
