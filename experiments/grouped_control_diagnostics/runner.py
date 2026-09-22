"""Label-only optimization with complete CUDA checkpoints and frozen-head controls."""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

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
from experiments.grouped_dynamic_vqc.train import ProgressState
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .diagnostics import control_output, evaluate_control
from .protocol import Experiment, module_hash, read_json


class RouteRun:
    """Head routes share the frozen frontend and untouched initial head."""

    def __init__(self, shared: Experiment, route: str, resume: bool = False) -> None:
        if route not in shared.routes:
            raise ValueError(f"Unknown training route: {route}")
        self.shared, self.config, self.route = shared, shared.config, route
        self.output = shared.output / route
        self.output.mkdir(parents=True, exist_ok=True)
        self.epochs = (
            self.config.epochs if route == "standard" else self.config.head_epochs
        )
        self.epoch_offset = 0 if route == "standard" else shared.start_epoch
        self.mode = {"head_true": "both", "head_zero": "zero"}.get(route, "measured")
        initial = shared.initial if route == "standard" else shared.source
        self.model = shared.make_model(initial["model"])
        restore_rng(initial["rng"])
        self.model.frontend.requires_grad_(route == "standard")
        self.model.label_head.requires_grad_(True)
        self.initial_hash = state_hash(self.model.state_dict())
        self.frontend_hash = state_hash(self.model.frontend.state_dict())
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )
        self.progress: ProgressState = {
            "completed_epoch": 0,
            "global_step": 0,
            "order": None,
            "offset": 0,
            "loss_sums": [0.0, 0.0],
            "history": [],
        }
        self.gradient_checks: dict = {}
        self.training_seconds = 0.0
        self.stop_requested = False
        if resume:
            checkpoint = torch.load(
                self.output / "resume.pt", weights_only=False, map_location="cpu"
            )
            if (
                checkpoint["manifest_sha256"] != shared.manifest_hash
                or checkpoint["route"] != route
            ):
                raise ValueError("Checkpoint manifest or route mismatch")
            if checkpoint["initial_model_sha256"] != self.initial_hash:
                raise ValueError("Checkpoint has different initialization")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.progress = checkpoint["progress"]
            self.gradient_checks = checkpoint["gradient_checks"]
            self.training_seconds = checkpoint["training_seconds"]
            restore_rng(checkpoint["rng"])
        else:
            if (self.output / "resume.pt").exists():
                raise FileExistsError("Route exists; use --resume")
            atomic_json(
                self.output / "initialization.json",
                {
                    "initial_model_sha256": self.initial_hash,
                    "manifest_sha256": shared.manifest_hash,
                    "initial_head_sha256": state_hash(
                        self.model.label_head.state_dict()
                    ),
                    "initial_frontend_sha256": self.frontend_hash,
                    "fresh_adam": not self.optimizer.state,
                },
            )
            self.save_checkpoint()
        self.verify_frozen()
        torch.cuda.reset_peak_memory_stats()

    def verify_frozen(self) -> None:
        if (
            self.route != "standard"
            and state_hash(self.model.frontend.state_dict()) != self.frontend_hash
        ):
            raise RuntimeError("Frozen frontend was modified")

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True

    def checkpoint(self) -> dict:
        self.verify_frozen()
        return {
            "manifest_sha256": self.shared.manifest_hash,
            "route": self.route,
            "initial_model_sha256": self.initial_hash,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": rng_state(),
            "progress": self.progress,
            "gradient_checks": self.gradient_checks,
            "training_seconds": self.training_seconds,
            "control_mode": self.mode,
        }

    def save_checkpoint(self) -> None:
        atomic_checkpoint(self.output / "resume.pt", self.checkpoint())

    def heartbeat(self, status: str, **details: object) -> None:
        content = {
            "updated_at": utc_now(),
            "pid": os.getpid(),
            "status": status,
            "cell": self.route,
            "epoch_completed": self.progress["completed_epoch"],
            "epochs_total": self.epochs,
            "global_step": self.progress["global_step"],
            "offset": self.progress["offset"],
            "device": self.shared.runtime["device"],
            "completed_cells": sum(
                (self.shared.output / r / "result.json").exists()
                for r in self.shared.routes
            ),
            "total_cells": len(self.shared.routes),
            "training_seconds": self.training_seconds,
            **details,
        }
        atomic_json(self.output / "heartbeat.json", content)
        atomic_json(
            self.shared.output / "heartbeat.json",
            {**content, "status": "between_routes" if status == "complete" else status},
        )

    def check_gradients(self) -> None:
        norms = {}
        for name, module in (
            ("frontend", self.model.frontend),
            ("label_head", self.model.label_head),
        ):
            parameters = list(module.parameters())
            if any(not p.is_cuda for p in parameters):
                raise RuntimeError("Model parameter not on CUDA")
            if parameters[0].requires_grad:
                if any(
                    p.grad is None
                    or not p.grad.is_cuda
                    or not torch.isfinite(p.grad).all()
                    for p in parameters
                ):
                    raise RuntimeError(
                        f"Missing/nonfinite/non-CUDA active gradient: {name}"
                    )
                norm = math.sqrt(
                    sum(
                        float(p.grad.square().sum())
                        for p in parameters
                        if p.grad is not None
                    )
                )
                norms[name] = norm
                if norm <= 0:
                    raise RuntimeError(f"Zero module gradient: {name}")
            else:
                if any(p.grad is not None for p in parameters):
                    raise RuntimeError("Frozen frontend received gradients")
                norms[name] = None
        self.gradient_checks = {
            "device": str(next(self.model.parameters()).device),
            "input_device": str(self.shared.data["train"]["angles"].device),
            "gradient_l2": norms,
            "objective": "label BCE only",
        }
        atomic_json(self.output / "gradient_checks.json", self.gradient_checks)

    def train_step(self, indices: torch.Tensor) -> tuple[float, float]:
        started = time.perf_counter()
        data = self.shared.data["train"]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(self.route == "standard"):
            states = self.model.frontend(data["angles"][indices])
        output = control_output(
            self.model, states, data["concepts"][indices], self.mode
        )
        loss = F.binary_cross_entropy(
            output["label_prob"].clamp(1e-7, 1 - 1e-7), data["labels"][indices]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config.grad_clip,
            error_if_nonfinite=True,
        )
        if not self.gradient_checks:
            self.check_gradients()
        with torch.no_grad():
            codes = data["concepts"][indices, 0] * 8 + data["concepts"][indices, 1]
            nll = (
                -output["concept_probs"]
                .gather(1, codes[:, None])
                .clamp_min(1e-7)
                .log()
                .mean()
            )
        numbers = float(loss.detach()), float(nll)
        if not all(math.isfinite(v) for v in numbers):
            raise RuntimeError("Nonfinite loss")
        self.optimizer.step()
        torch.cuda.synchronize()
        self.training_seconds += time.perf_counter() - started
        return numbers

    def finish_epoch(self, count: int) -> None:
        self.verify_frozen()
        self.heartbeat("validation")
        validation = evaluate_control(
            self.model, self.shared.data["val"], self.config.eval_batch_size, self.mode
        )
        epoch = self.progress["completed_epoch"] + 1
        order = self.progress["order"]
        assert order is not None
        self.progress["history"].append(
            {
                "epoch": epoch,
                "reference_order_epoch": epoch + self.epoch_offset,
                "global_step": self.progress["global_step"],
                "order_sha256": array_hash(order.numpy()),
                "train_label_bce": self.progress["loss_sums"][0] / count,
                "train_concept_nll_monitor_only": self.progress["loss_sums"][1] / count,
                "validation": validation,
            }
        )
        self.progress.update(
            completed_epoch=epoch, order=None, offset=0, loss_sums=[0.0, 0.0]
        )
        self.save_checkpoint()
        atomic_json(self.output / "history.json", self.progress["history"])
        report(
            f"{self.route} {epoch}/{self.epochs} | Label ({self.mode}) "
            f"{validation['label']['accuracy']:.2%} | "
            f"BCE {validation['label']['bce']:.4f}"
        )

    def run(self, max_steps: int | None = None) -> dict:
        count = len(self.shared.data["train"]["angles"])
        while self.progress["completed_epoch"] < self.epochs:
            if self.stop_requested or (
                max_steps is not None and self.progress["global_step"] >= max_steps
            ):
                self.save_checkpoint()
                self.heartbeat("paused")
                return {"status": "paused", "route": self.route}
            if self.progress["order"] is None:
                self.progress["order"] = epoch_order(
                    count,
                    self.config.seed,
                    self.progress["completed_epoch"] + 1 + self.epoch_offset,
                )
            order = self.progress["order"]
            assert order is not None
            start = self.progress["offset"]
            if start < count:
                indices = order[start : start + self.config.batch_size].cuda()
                losses = self.train_step(indices)
                self.progress["offset"] += len(indices)
                self.progress["global_step"] += 1
                for i, loss in enumerate(losses):
                    self.progress["loss_sums"][i] += loss * len(indices)
                self.heartbeat("training", batch_loss=losses[0])
                if self.progress["global_step"] % self.config.checkpoint_steps == 0:
                    self.save_checkpoint()
            if self.progress["offset"] == count:
                self.finish_epoch(count)
        self.save_checkpoint()
        atomic_json(self.output / "history.json", self.progress["history"])
        atomic_checkpoint(self.output / "endpoint.pt", self.checkpoint())
        result = {
            "status": "complete",
            "route": self.route,
            "epoch": self.epochs,
            "global_step": self.progress["global_step"],
            "control_mode": self.mode,
            "manifest_sha256": self.shared.manifest_hash,
            "model_sha256": state_hash(self.model.state_dict()),
            "frontend_sha256": state_hash(self.model.frontend.state_dict()),
            "head_sha256": state_hash(self.model.label_head.state_dict()),
            "initial_model_sha256": self.initial_hash,
            "module_steps": {
                "frontend": self.progress["global_step"]
                if self.route == "standard"
                else 0,
                "label_head": self.progress["global_step"],
            },
            "training_seconds": self.training_seconds,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "validation": self.progress["history"][-1]["validation"],
            "test_evaluated": False,
            "artifacts": {
                name: sha256(self.output / name)
                for name in (
                    "resume.pt",
                    "endpoint.pt",
                    "history.json",
                    "initialization.json",
                    "gradient_checks.json",
                )
            },
        }
        atomic_json(self.output / "result.json", result)
        self.heartbeat("complete", validation=result["validation"])
        return result


def verify_complete(shared: Experiment, route: str) -> dict:
    directory = shared.output / route
    result = read_json(directory / "result.json")
    epochs = shared.config.epochs if route == "standard" else shared.config.head_epochs
    offset = 0 if route == "standard" else shared.start_epoch
    if (
        result["status"] != "complete"
        or result["epoch"] != epochs
        or result["manifest_sha256"] != shared.manifest_hash
        or result["test_evaluated"]
    ):
        raise ValueError("Completed route protocol mismatch")
    for name, expected in result["artifacts"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Completed artifact changed: {route}/{name}")
    checkpoint = torch.load(
        directory / "endpoint.pt", weights_only=False, map_location="cpu"
    )
    history = read_json(directory / "history.json")
    initial = shared.initial if route == "standard" else shared.source
    count = len(shared.data["train"]["angles"])
    steps = math.ceil(count / shared.config.batch_size) * epochs
    if (
        state_hash(checkpoint["model"]) != result["model_sha256"]
        or result["initial_model_sha256"] != state_hash(initial["model"])
        or len(history) != epochs
        or checkpoint["progress"]["history"] != history
        or checkpoint["progress"]["completed_epoch"] != epochs
        or result["global_step"] != steps
        or checkpoint["progress"]["global_step"] != steps
        or checkpoint["progress"]["order"] is not None
        or checkpoint["progress"]["offset"] != 0
    ):
        raise ValueError("Completed model/history/budget mismatch")
    for epoch, row in enumerate(history, 1):
        expected = array_hash(
            epoch_order(count, shared.config.seed, epoch + offset).numpy()
        )
        if row["epoch"] != epoch or row["order_sha256"] != expected:
            raise ValueError("Completed sample order mismatch")
    if route != "standard" and module_hash(
        checkpoint["model"], "frontend"
    ) != module_hash(shared.source["model"], "frontend"):
        raise ValueError("Completed head route changed its frozen frontend")
    return result
