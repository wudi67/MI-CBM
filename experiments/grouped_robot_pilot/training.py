"""Concept NLL then two paired frozen-frontend BCE routes, with Adam recovery."""

from __future__ import annotations

import math
import time
from typing import Any

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
)
from experiments.grouped_dynamic_vqc.train import ProgressState
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .evaluation import evaluate
from .model import concept_loss, forward_control
from .protocol import CELLS, read_json, verify_files


def load(path) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


class Route:
    def __init__(self, shared: Any, cell: str) -> None:
        if cell not in CELLS:
            raise ValueError("Unknown training cell")
        self.shared, self.config, self.cell = shared, shared.config, cell
        self.output = shared.output / "training" / cell
        self.output.mkdir(parents=True, exist_ok=True)
        self.epochs = (
            self.config.concept_epochs if cell == "concept" else self.config.head_epochs
        )
        self.epoch_offset = 0 if cell == "concept" else self.config.concept_epochs
        initial = shared.initial_for(cell)
        self.model = shared.make_model(initial["model"])
        restore_rng(initial["rng"])
        self.initial_hash = state_hash(initial["model"])
        self.frozen_module = "label_head" if cell == "concept" else "frontend"
        self.frozen_hash = state_hash(
            getattr(self.model, self.frozen_module).state_dict()
        )
        self.model.frontend.requires_grad_(cell == "concept")
        self.model.label_head.requires_grad_(cell != "concept")
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.config.learning_rate)
        self.progress: ProgressState = {
            "completed_epoch": 0,
            "global_step": 0,
            "order": None,
            "offset": 0,
            "loss_sums": [0.0],
            "history": [],
        }
        self.gradient_checks: dict = {}
        self.training_seconds = 0.0
        if (self.output / "resume.pt").exists():
            checkpoint = load(self.output / "resume.pt")
            if any(checkpoint[k] != v for k, v in self.identity().items()):
                raise ValueError("Resume initialization/protocol mismatch")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.progress = checkpoint["progress"]
            self.gradient_checks = checkpoint["gradient_checks"]
            self.training_seconds = checkpoint["training_seconds"]
            restore_rng(checkpoint["rng"])
        else:
            atomic_json(
                self.output / "initialization.json",
                {
                    **self.identity(),
                    "fresh_adam": True,
                    "frontend_sha256": state_hash(self.model.frontend.state_dict()),
                    "head_sha256": state_hash(self.model.label_head.state_dict()),
                },
            )
            self.save()
        self.verify_frozen()
        if cell != "concept":
            shared.prepare_states(self.model)
        torch.cuda.reset_peak_memory_stats()

    def identity(self) -> dict:
        return {
            "manifest_sha256": self.shared.manifest_hash,
            "data_lock_sha256": self.shared.data_hash,
            "cell": self.cell,
            "initial_model_sha256": self.initial_hash,
            "epochs": self.epochs,
            "epoch_offset": self.epoch_offset,
            "training_control": {
                "concept": "none",
                "independent": "true",
                "no_feedback": "zero",
            }[self.cell],
        }

    def verify_frozen(self) -> None:
        if (
            state_hash(getattr(self.model, self.frozen_module).state_dict())
            != self.frozen_hash
        ):
            raise ValueError("Training changed a frozen circuit")

    def checkpoint(self) -> dict:
        self.verify_frozen()
        return {
            **self.identity(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": rng_state(),
            "progress": self.progress,
            "gradient_checks": self.gradient_checks,
            "training_seconds": self.training_seconds,
        }

    def save(self) -> None:
        atomic_checkpoint(self.output / "resume.pt", self.checkpoint())

    def heartbeat(self, status: str, **details) -> None:
        self.shared.heartbeat(
            status,
            epoch_completed=self.progress["completed_epoch"],
            epochs_total=self.epochs,
            global_step=self.progress["global_step"],
            offset=self.progress["offset"],
            **details,
        )

    def check_gradients(self) -> None:
        norms = {}
        for name in ("frontend", "label_head"):
            parameters = list(getattr(self.model, name).parameters())
            if any(not p.is_cuda for p in parameters):
                raise RuntimeError("All quantum parameters must be on CUDA")
            if name == self.frozen_module:
                if any(p.grad is not None for p in parameters):
                    raise RuntimeError("Frozen circuit received gradients")
                norms[name] = None
            else:
                if any(
                    p.grad is None
                    or not p.grad.is_cuda
                    or not torch.isfinite(p.grad).all()
                    for p in parameters
                ):
                    raise RuntimeError("Active gradients must be finite and on CUDA")
                norm = math.sqrt(
                    sum(
                        float(p.grad.square().sum())
                        for p in parameters
                        if p.grad is not None
                    )
                )
                if norm <= 0:
                    raise RuntimeError("Active circuit has zero gradient")
                norms[name] = norm
        self.gradient_checks = {
            "device": str(self.parameters[0].device),
            "gradient_l2": norms,
            "input_device": str(self.shared.data["train"]["angles"].device),
        }
        atomic_json(self.output / "gradient_checks.json", self.gradient_checks)

    def step(self, indices: torch.Tensor) -> float:
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data = self.shared.data["train"]
        if self.cell == "concept":
            states = self.model.frontend(data["angles"][indices])
            loss = concept_loss(
                states.reshape(-1, 32, 32).abs().square().sum(-1),
                data["concepts"][indices],
            )
        else:
            states = self.shared.state_cache["train"][indices]
            output = forward_control(
                self.model,
                states,
                data["concepts"][indices],
                zero=self.cell == "no_feedback",
                mask=31 if self.cell == "independent" else 0,
            )
            loss = F.binary_cross_entropy(
                output["label_prob"].clamp(1e-7, 1 - 1e-7), data["labels"][indices]
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters, self.config.grad_clip, error_if_nonfinite=True
        )
        if not self.gradient_checks:
            self.check_gradients()
        self.optimizer.step()
        torch.cuda.synchronize()
        self.training_seconds += time.perf_counter() - started
        return float(loss.detach())

    def finish_epoch(self, count: int) -> None:
        self.verify_frozen()
        self.heartbeat("validation")
        validation, _ = evaluate(
            self.model,
            self.shared.data["validation"],
            self.config.eval_batch_size,
            zero=self.cell == "no_feedback",
            concept_only=self.cell == "concept",
            states=None
            if self.cell == "concept"
            else self.shared.state_cache["validation"],
        )
        order = self.progress["order"]
        assert order is not None
        epoch = self.progress["completed_epoch"] + 1
        self.progress["history"].append(
            {
                "epoch": epoch,
                "order_epoch": epoch + self.epoch_offset,
                "global_step": self.progress["global_step"],
                "order_sha256": array_hash(order.numpy()),
                "train_loss": self.progress["loss_sums"][0] / count,
                "validation": validation,
            }
        )
        self.progress.update(
            completed_epoch=epoch, order=None, offset=0, loss_sums=[0.0]
        )
        self.save()
        atomic_json(self.output / "history.json", self.progress["history"])
        accuracy = validation["concept"]["all_concepts_accuracy"]
        label = (
            ""
            if self.cell == "concept"
            else f" | label={validation['label']['accuracy']:.2%}"
        )
        report(
            f"{self.cell} {epoch}/{self.epochs} | "
            f"five concepts correct={accuracy:.2%}{label}"
        )
        self.heartbeat("between_epochs", validation=validation)

    def run(self, max_steps: int | None = None) -> dict:
        count = len(self.shared.data["train"]["angles"])
        while self.progress["completed_epoch"] < self.epochs:
            if self.shared.stop_requested or (
                max_steps is not None and self.progress["global_step"] >= max_steps
            ):
                self.save()
                raise InterruptedError("Paused with complete Adam/RNG/minibatch state")
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
                loss = self.step(indices)
                self.progress["offset"] += len(indices)
                self.progress["global_step"] += 1
                self.progress["loss_sums"][0] += loss * len(indices)
                self.heartbeat("training", batch_loss=loss)
                if self.progress["global_step"] % self.config.checkpoint_steps == 0:
                    self.save()
            if self.progress["offset"] == count:
                self.finish_epoch(count)
        self.save()
        atomic_checkpoint(self.output / "endpoint.pt", self.checkpoint())
        result = {
            **self.identity(),
            "status": "complete",
            "test_evaluated": False,
            "model_sha256": state_hash(self.model.state_dict()),
            "frontend_sha256": state_hash(self.model.frontend.state_dict()),
            "head_sha256": state_hash(self.model.label_head.state_dict()),
            "global_step": self.progress["global_step"],
            "training_seconds": self.training_seconds,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "validation": self.progress["history"][-1]["validation"],
            "artifacts": {
                name: sha256(self.output / name)
                for name in (
                    "endpoint.pt",
                    "resume.pt",
                    "history.json",
                    "initialization.json",
                    "gradient_checks.json",
                )
            },
        }
        atomic_json(self.output / "result.json", result)
        return result


def verify_complete(shared: Any, cell: str) -> dict:
    directory = shared.output / "training" / cell
    result = read_json(directory / "result.json")
    verify_files(directory, result["artifacts"])
    checkpoint, initial = load(directory / "endpoint.pt"), shared.initial_for(cell)
    epochs = (
        shared.config.concept_epochs if cell == "concept" else shared.config.head_epochs
    )
    offset = 0 if cell == "concept" else shared.config.concept_epochs
    count = len(shared.data["train"]["angles"])
    steps = math.ceil(count / shared.config.batch_size) * epochs
    history = read_json(directory / "history.json")
    expected = {
        "manifest_sha256": shared.manifest_hash,
        "data_lock_sha256": shared.data_hash,
        "cell": cell,
        "epochs": epochs,
        "epoch_offset": offset,
        "initial_model_sha256": state_hash(initial["model"]),
        "training_control": {
            "concept": "none",
            "independent": "true",
            "no_feedback": "zero",
        }[cell],
    }
    if (
        result["status"] != "complete"
        or result["test_evaluated"]
        or any(result[k] != v or checkpoint[k] != v for k, v in expected.items())
        or state_hash(checkpoint["model"]) != result["model_sha256"]
        or result["global_step"] != steps
        or len(history) != epochs
    ):
        raise ValueError("Completed training protocol/weights/budget mismatch")
    progress = checkpoint["progress"]
    if (
        progress["completed_epoch"] != epochs
        or progress["global_step"] != steps
        or progress["order"] is not None
        or progress["offset"] != 0
        or progress["history"] != history
    ):
        raise ValueError("Completed Adam/epoch boundary mismatch")
    for epoch, row in enumerate(history, 1):
        if row["order_sha256"] != array_hash(
            epoch_order(count, shared.config.seed, epoch + offset).numpy()
        ):
            raise ValueError("Sample order changed")
    prefix = "label_head" if cell == "concept" else "frontend"
    frozen = {
        k: v for k, v in checkpoint["model"].items() if k.startswith(prefix + ".")
    }
    original = {k: v for k, v in initial["model"].items() if k.startswith(prefix + ".")}
    if state_hash(frozen) != state_hash(original):
        raise ValueError("Completed route modified its frozen circuit")
    for name, key in (("frontend", "frontend_sha256"), ("label_head", "head_sha256")):
        module = {
            k.removeprefix(name + "."): v
            for k, v in checkpoint["model"].items()
            if k.startswith(name + ".")
        }
        if state_hash(module) != result[key]:
            raise ValueError("Saved circuit fingerprint differs from actual weights")
    return result
