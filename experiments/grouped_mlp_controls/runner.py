"""CUDA training cells with exact minibatch recovery and fixed-final results."""

from __future__ import annotations

import math
import os
import time
from typing import TypedDict

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    array_hash,
    atomic_checkpoint,
    atomic_json,
    report,
    restore_rng,
    rng_state,
    sha256,
    utc_now,
)
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .model import PARAMETERS, TinyMLP, evaluate, objective
from .protocol import Experiment, cell_name, read_json


class Progress(TypedDict):
    epoch: int
    steps: int
    offset: int
    order: torch.Tensor | None
    loss_sum: float
    history: list[dict]


class CellRun:
    """All learning rates for a task start from the same actual tensors."""

    def __init__(
        self, experiment: Experiment, task: str, lr: float, resume: bool = False
    ) -> None:
        self.experiment, self.task, self.lr = experiment, task, lr
        self.config = experiment.config
        self.name = cell_name(task, lr)
        self.output = experiment.output / self.name
        self.output.mkdir(parents=True, exist_ok=True)
        self.model = TinyMLP(task).cuda()
        self.model.load_state_dict(experiment.initial[task]["model"])
        restore_rng(experiment.initial[task]["rng"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.progress: Progress = {
            "epoch": 0,
            "steps": 0,
            "offset": 0,
            "order": None,
            "loss_sum": 0.0,
            "history": [],
        }
        self.stop_requested = False
        self.training_seconds = 0.0
        self.gradient_check: dict = {}
        if resume:
            checkpoint = torch.load(
                self.output / "resume.pt", map_location="cpu", weights_only=False
            )
            if (
                checkpoint["manifest_sha256"] != experiment.manifest_hash
                or checkpoint["cell"] != self.name
            ):
                raise ValueError("Checkpoint does not belong to this experiment/cell")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.progress = checkpoint["progress"]
            self.training_seconds = checkpoint["training_seconds"]
            self.gradient_check = checkpoint["gradient_check"]
            restore_rng(checkpoint["rng"])
        else:
            if (self.output / "resume.pt").exists():
                raise FileExistsError("Cell exists; use resume")
            atomic_json(
                self.output / "initialization.json",
                {
                    "model_sha256": state_hash(self.model.state_dict()),
                    "manifest_sha256": experiment.manifest_hash,
                },
            )
            self.save()
        torch.cuda.reset_peak_memory_stats()

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True

    def heartbeat(self, status: str, **details: object) -> None:
        total = 2 * len(self.config.learning_rates)
        completed = sum(
            (self.experiment.output / cell_name(task, lr) / "result.json").exists()
            for task in ("concept", "label")
            for lr in self.config.learning_rates
        )
        content = {
            "updated_at": utc_now(),
            "pid": os.getpid(),
            "status": status,
            "cell": self.name,
            "task": self.task,
            "learning_rate": self.lr,
            "completed_cells": completed,
            "total_cells": total,
            "epoch_completed": self.progress["epoch"],
            "epochs_total": self.experiment.epochs_for(self.task),
            "global_step": self.progress["steps"],
            "offset": self.progress["offset"],
            "device": self.experiment.runtime["device"],
            **details,
        }
        atomic_json(self.output / "heartbeat.json", content)
        atomic_json(
            self.experiment.output / "heartbeat.json",
            {**content, "status": "between_cells" if status == "complete" else status},
        )

    def save(self) -> None:
        atomic_checkpoint(
            self.output / "resume.pt",
            {
                "cell": self.name,
                "manifest_sha256": self.experiment.manifest_hash,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "progress": self.progress,
                "rng": rng_state(),
                "training_seconds": self.training_seconds,
                "gradient_check": self.gradient_check,
            },
        )

    def train_step(self, indices: torch.Tensor) -> float:
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data = self.experiment.data["train"]
        loss = objective(self.task, self.model(data["angles"][indices]), data, indices)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True
        )
        if not self.gradient_check:
            if any(p.grad is None for p in self.model.parameters()) or float(norm) <= 0:
                raise RuntimeError("Missing or zero CUDA gradient")
            self.gradient_check = {
                "parameter_device": str(next(self.model.parameters()).device),
                "input_device": str(data["angles"].device),
                "gradient_norm_before_clip": float(norm),
                "parameters": sum(p.numel() for p in self.model.parameters()),
            }
            atomic_json(self.output / "cuda_gradient_check.json", self.gradient_check)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite training loss")
        self.optimizer.step()
        torch.cuda.synchronize()
        self.training_seconds += time.perf_counter() - started
        return value

    def finish_epoch(self) -> None:
        self.heartbeat("validation")
        validation = evaluate(
            self.model, self.experiment.data["val"], self.config.eval_batch_size
        )
        count = len(self.experiment.data["train"]["angles"])
        self.progress["epoch"] += 1
        epoch = self.progress["epoch"]
        order = self.progress["order"]
        assert order is not None
        order_hash = array_hash(order.numpy())
        if order_hash != self.experiment.order_hashes[epoch - 1]:
            raise RuntimeError("Actual sample order differs from the paired protocol")
        record = {
            "epoch": epoch,
            "global_step": self.progress["steps"],
            "train_loss": self.progress["loss_sum"] / count,
            "order_sha256": order_hash,
            "validation": validation,
        }
        self.progress["history"].append(record)
        self.progress.update(order=None, offset=0, loss_sum=0.0)
        self.save()
        atomic_json(self.output / "history.json", self.progress["history"])
        score = (
            validation["concept"]["joint_map_accuracy"]
            if self.task == "concept"
            else validation["label"]["accuracy"]
        )
        report(
            f"{self.name} epoch {epoch}/{self.experiment.epochs_for(self.task)} | "
            f"train loss {record['train_loss']:.5f} | "
            f"val loss {validation['loss']:.5f} | "
            f"{'concept MAP' if self.task == 'concept' else 'label accuracy'} "
            f"{score:.2%}"
        )

    def run(self, max_steps: int | None = None) -> dict:
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not (self.output / "initial_validation.json").exists():
            self.heartbeat("initial_validation")
            atomic_json(
                self.output / "initial_validation.json",
                evaluate(
                    self.model, self.experiment.data["val"], self.config.eval_batch_size
                ),
            )
        count = len(self.experiment.data["train"]["angles"])
        while self.progress["epoch"] < self.experiment.epochs_for(self.task):
            if self.stop_requested or (
                max_steps is not None and self.progress["steps"] >= max_steps
            ):
                self.save()
                self.heartbeat("paused")
                return {
                    "status": "paused",
                    "cell": self.name,
                    "global_step": self.progress["steps"],
                }
            if self.progress["order"] is None:
                self.progress["order"] = epoch_order(
                    count, self.config.seed, self.progress["epoch"] + 1
                )
            order = self.progress["order"]
            assert order is not None
            start = self.progress["offset"]
            if start < count:
                indices = order[start : start + self.config.batch_size].cuda()
                value = self.train_step(indices)
                self.progress["steps"] += 1
                self.progress["offset"] += len(indices)
                self.progress["loss_sum"] += value * len(indices)
                self.heartbeat("training", batch_loss=value)
                if self.progress["steps"] % self.config.checkpoint_steps == 0:
                    self.save()
            if self.progress["offset"] == count:
                self.finish_epoch()
        self.save()
        atomic_json(self.output / "history.json", self.progress["history"])
        result = {
            "status": "complete",
            "cell": self.name,
            "task": self.task,
            "learning_rate": self.lr,
            "epochs": self.progress["epoch"],
            "global_step": self.progress["steps"],
            "parameters": PARAMETERS[self.task],
            "model_sha256": state_hash(self.model.state_dict()),
            "initial_model_sha256": self.experiment.initial[self.task]["model_sha256"],
            "validation": self.progress["history"][-1]["validation"],
            "manifest_sha256": self.experiment.manifest_hash,
            "checkpoint_sha256": sha256(self.output / "resume.pt"),
            "history_sha256": sha256(self.output / "history.json"),
            "training_seconds": self.training_seconds,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "test_evaluated": False,
            "evidence_role": "fixed-final epoch development validation",
        }
        atomic_json(self.output / "result.json", result)
        self.heartbeat("complete", validation=result["validation"])
        return result


def verify_complete(experiment: Experiment, task: str, lr: float) -> dict:
    output = experiment.output / cell_name(task, lr)
    result = read_json(output / "result.json")
    history = read_json(output / "history.json")
    initialization = read_json(output / "initialization.json")
    epochs = experiment.epochs_for(task)
    count = len(experiment.data["train"]["angles"])
    if (
        result["status"] != "complete"
        or result["task"] != task
        or result["learning_rate"] != lr
        or result["parameters"] != PARAMETERS[task]
        or result["epochs"] != epochs
        or result["manifest_sha256"] != experiment.manifest_hash
        or result["checkpoint_sha256"] != sha256(output / "resume.pt")
        or result["history_sha256"] != sha256(output / "history.json")
        or result["initial_model_sha256"] != experiment.initial[task]["model_sha256"]
        or initialization["model_sha256"] != result["initial_model_sha256"]
        or initialization["manifest_sha256"] != experiment.manifest_hash
        or result["global_step"]
        != epochs * math.ceil(count / experiment.config.batch_size)
        or [row["epoch"] for row in history] != list(range(1, epochs + 1))
        or [row["order_sha256"] for row in history] != experiment.order_hashes[:epochs]
        or history[-1]["validation"] != result["validation"]
        or result["test_evaluated"]
    ):
        raise ValueError(
            f"Completed cell failed provenance/budget/order check: {output}"
        )
    checkpoint = torch.load(
        output / "resume.pt", weights_only=False, map_location="cpu"
    )
    if (
        state_hash(checkpoint["model"]) != result["model_sha256"]
        or checkpoint["manifest_sha256"] != experiment.manifest_hash
        or checkpoint["cell"] != cell_name(task, lr)
        or sum(value.numel() for value in checkpoint["model"].values())
        != PARAMETERS[task]
        or checkpoint["progress"]["history"] != history
    ):
        raise ValueError("Checkpoint model or history differs from final result")
    return result
