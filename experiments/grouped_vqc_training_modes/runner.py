"""Resumable phase transitions, paired orders and fixed-epoch measurements."""

from __future__ import annotations

import json
import os
import time

import torch

from experiments.grouped_dynamic_vqc.evaluation import evaluate
from experiments.grouped_dynamic_vqc.model import GroupedDynamicVQC
from experiments.grouped_dynamic_vqc.objectives import loss_function
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

from .protocol import ROUTES, SharedExperiment, epoch_order, state_hash


class RouteRun:
    """Sequential stage one leaves the head at its shared initialization."""

    def __init__(
        self, shared: SharedExperiment, route: str, resume: bool = False
    ) -> None:
        if route not in ROUTES:
            raise ValueError(f"Unknown route: {route}")
        self.shared = shared
        self.config = shared.config
        self.route = route
        self.output = shared.output / route
        self.output.mkdir(parents=True, exist_ok=True)
        self.stop_requested = False
        self.model = GroupedDynamicVQC(
            self.config.front_layers, self.config.label_layers
        ).cuda()
        self.model.load_state_dict(shared.initial["model"])
        restore_rng(shared.initial["rng"])
        self.phase = "joint" if route == "joint" else "concept"
        self.frozen_frontend_hash: str | None = None
        self.initial_head_hash = state_hash(self.model.label_head.state_dict())
        self.gradient_checks: dict = {}
        self.module_steps = {"frontend": 0, "label_head": 0}
        self.training_seconds = 0.0
        self.progress: ProgressState = {
            "completed_epoch": 0,
            "global_step": 0,
            "order": None,
            "offset": 0,
            "loss_sums": [0.0, 0.0, 0.0],
            "history": [],
        }
        if resume:
            checkpoint = torch.load(
                self.output / "resume.pt", weights_only=False, map_location="cpu"
            )
            if (
                checkpoint["manifest_sha256"] != shared.manifest_hash
                or checkpoint["route"] != route
            ):
                raise ValueError("Checkpoint does not belong to this experiment/route")
            self.model.load_state_dict(checkpoint["model"])
            self.phase = checkpoint["phase"]
            self.progress = checkpoint["progress"]
            self.frozen_frontend_hash = checkpoint["frozen_frontend_hash"]
            self.gradient_checks = checkpoint["gradient_checks"]
            self.module_steps = checkpoint["module_steps"]
            self.training_seconds = checkpoint["training_seconds"]
            self.configure_optimizer()
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            restore_rng(checkpoint["rng"])
            self.verify_frozen()
        else:
            if (self.output / "resume.pt").exists():
                raise FileExistsError("Route exists; use --resume")
            self.configure_optimizer()
            atomic_json(
                self.output / "initialization.json",
                {
                    "manifest_sha256": shared.manifest_hash,
                    "initial_model_sha256": state_hash(self.model.state_dict()),
                    "initial_head_sha256": self.initial_head_hash,
                },
            )
            self.save_checkpoint()
        torch.cuda.reset_peak_memory_stats()

    def configure_optimizer(self) -> None:
        self.model.zero_grad(set_to_none=True)
        self.model.frontend.requires_grad_(self.phase in {"joint", "concept"})
        self.model.label_head.requires_grad_(self.phase in {"joint", "label"})
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True

    def verify_frozen(self) -> None:
        if (
            self.phase == "concept"
            and state_hash(self.model.label_head.state_dict()) != self.initial_head_hash
        ):
            raise RuntimeError("Concept-only training changed the label head")
        if (
            self.phase == "label"
            and state_hash(self.model.frontend.state_dict())
            != self.frozen_frontend_hash
        ):
            raise RuntimeError("Label-only training changed the frozen frontend")

    def checkpoint(self) -> dict:
        self.verify_frozen()
        return {
            "manifest_sha256": self.shared.manifest_hash,
            "route": self.route,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "progress": self.progress,
            "phase": self.phase,
            "rng": rng_state(),
            "frozen_frontend_hash": self.frozen_frontend_hash,
            "gradient_checks": self.gradient_checks,
            "module_steps": self.module_steps,
            "training_seconds": self.training_seconds,
        }

    def save_checkpoint(self) -> None:
        atomic_checkpoint(self.output / "resume.pt", self.checkpoint())

    def heartbeat(self, status: str, **details: object) -> None:
        content = {
            "updated_at": utc_now(),
            "pid": os.getpid(),
            "status": status,
            "route": self.route,
            "phase": self.phase,
            "epoch_completed": self.progress["completed_epoch"],
            "epochs_total": self.config.epochs,
            "global_step": self.progress["global_step"],
            "offset": self.progress["offset"],
            "device": self.shared.runtime["device"],
            "training_seconds": self.training_seconds,
            **details,
        }
        atomic_json(self.output / "heartbeat.json", content)
        # A completed child is not a completed paired experiment.
        parent = {
            **content,
            "status": "between_routes" if status == "complete" else status,
        }
        atomic_json(self.shared.output / "heartbeat.json", parent)

    def check_gradients(self) -> None:
        norms = {}
        for name, module in (
            ("frontend", self.model.frontend),
            ("label_head", self.model.label_head),
        ):
            parameters = list(module.parameters())
            if parameters[0].requires_grad:
                if any(
                    p.grad is None or not torch.isfinite(p.grad).all()
                    for p in parameters
                ):
                    raise RuntimeError(f"Missing/nonfinite active gradient: {name}")
                norm = (
                    sum(
                        float(p.grad.square().sum())
                        for p in parameters
                        if p.grad is not None
                    )
                    ** 0.5
                )
                if norm <= 0:
                    raise RuntimeError(f"Zero active gradient: {name}")
                norms[name] = norm
            elif any(p.grad is not None for p in parameters):
                raise RuntimeError(f"Frozen module received a gradient: {name}")
            else:
                norms[name] = None
        self.gradient_checks[self.phase] = {
            "global_step_before_update": self.progress["global_step"],
            "device": str(next(self.model.parameters()).device),
            "gradient_l2": norms,
        }
        atomic_json(self.output / "gradient_checks.json", self.gradient_checks)

    def train_step(self, indices: torch.Tensor) -> tuple[float, float, float]:
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data = self.shared.data["train"]
        if self.phase == "label":
            with torch.no_grad():
                states = self.model.frontend(data["angles"][indices])
            output = self.model.from_state(states)
        else:
            output = self.model(data["angles"][indices])
        losses = loss_function(
            output,
            data["concepts"][indices],
            data["labels"][indices],
            self.config.concept_weight,
            self.config.label_weight,
        )
        if self.phase == "concept":
            total = self.config.concept_weight * losses[1]
        elif self.phase == "label":
            total = self.config.label_weight * losses[2]
        else:
            total = losses[0]
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config.grad_clip,
            error_if_nonfinite=True,
        )
        if self.phase not in self.gradient_checks:
            self.check_gradients()
        numbers = (
            float(total.detach()),
            float(losses[1].detach()),
            float(losses[2].detach()),
        )
        if not all(torch.isfinite(torch.tensor(numbers))):
            raise RuntimeError("Nonfinite training loss")
        self.optimizer.step()
        for name, module in (
            ("frontend", self.model.frontend),
            ("label_head", self.model.label_head),
        ):
            if next(module.parameters()).requires_grad:
                self.module_steps[name] += 1
        torch.cuda.synchronize()
        self.training_seconds += time.perf_counter() - started
        return numbers

    def endpoint(self) -> dict:
        """Idempotent endpoints also recover interruption at a phase boundary."""
        epoch = self.progress["completed_epoch"]
        stem = self.output / "endpoints" / f"epoch_{epoch:04d}"
        record_path, checkpoint_path = (
            stem.with_suffix(".json"),
            stem.with_suffix(".pt"),
        )
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if (
                record["checkpoint_sha256"] != sha256(checkpoint_path)
                or record["model_sha256"] != state_hash(self.model.state_dict())
                or record["manifest_sha256"] != self.shared.manifest_hash
            ):
                raise ValueError("Endpoint content/provenance mismatch")
            return record
        self.heartbeat("endpoint_diagnostics")
        validation = evaluate(
            self.model,
            self.shared.data["val"],
            self.config.eval_batch_size,
            diagnostics=True,
            shots=self.config.shots,
            seed=self.config.seed,
        )
        atomic_checkpoint(checkpoint_path, self.checkpoint())
        record = {
            "route": self.route,
            "phase": self.phase,
            "epoch": epoch,
            "global_step": self.progress["global_step"],
            "module_steps": dict(self.module_steps),
            "model_sha256": state_hash(self.model.state_dict()),
            "frontend_sha256": state_hash(self.model.frontend.state_dict()),
            "head_sha256": state_hash(self.model.label_head.state_dict()),
            "manifest_sha256": self.shared.manifest_hash,
            "checkpoint_sha256": sha256(checkpoint_path),
            "training_seconds": self.training_seconds,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "validation": validation,
            "test_evaluated": False,
            "label_head_trained": self.module_steps["label_head"] > 0,
            "evidence_role": "fixed epoch, development validation only",
        }
        atomic_json(record_path, record)
        return record

    def synchronize_phase(self) -> None:
        if (
            self.progress["completed_epoch"] == self.config.concept_epochs
            and self.progress["order"] is None
        ):
            self.endpoint()
        if (
            self.route == "sequential"
            and self.phase == "concept"
            and self.progress["completed_epoch"] >= self.config.concept_epochs
        ):
            self.verify_frozen()
            self.frozen_frontend_hash = state_hash(self.model.frontend.state_dict())
            self.phase = "label"
            # New label-only Adam; never load concept moments.
            self.configure_optimizer()
            if self.optimizer.state:
                raise RuntimeError("Label optimizer must start with empty Adam state")
            self.save_checkpoint()
            report(
                "[cyan]Sequential: full 10-wire frontend frozen; "
                "fresh label-only Adam.[/cyan]"
            )

    def finish_epoch(self, train_count: int) -> None:
        self.heartbeat("validation")
        validation = evaluate(
            self.model, self.shared.data["val"], self.config.eval_batch_size
        )
        epoch = self.progress["completed_epoch"] + 1
        order = self.progress["order"]
        assert order is not None
        record = {
            "epoch": epoch,
            "phase": self.phase,
            "global_step": self.progress["global_step"],
            "order_sha256": array_hash(order.numpy()),
            "train_loss": self.progress["loss_sums"][0] / train_count,
            "train_concept_nll": self.progress["loss_sums"][1] / train_count,
            "train_label_bce": self.progress["loss_sums"][2] / train_count,
            "label_head_trained": self.module_steps["label_head"] > 0,
            "validation": validation,
        }
        self.progress["history"].append(record)
        self.progress.update(
            completed_epoch=epoch, order=None, offset=0, loss_sums=[0.0, 0.0, 0.0]
        )
        self.save_checkpoint()
        atomic_json(self.output / "history.json", self.progress["history"])
        report(
            f"{self.route}/{self.phase} epoch {epoch}/{self.config.epochs} | "
            f"NLL {validation['concept']['joint_nll']:.4f} | "
            "single-shot "
            f"{validation['concept']['joint_single_shot_probability']:.2%} | "
            f"label {validation['label']['accuracy']:.2%}"
        )

    def run(self, max_steps: int | None = None) -> dict:
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not (self.output / "initial_validation.json").exists():
            self.heartbeat("initial_validation")
            atomic_json(
                self.output / "initial_validation.json",
                evaluate(
                    self.model, self.shared.data["val"], self.config.eval_batch_size
                ),
            )
        train_count = len(self.shared.data["train"]["angles"])
        while self.progress["completed_epoch"] < self.config.epochs:
            if self.stop_requested or (
                max_steps is not None and self.progress["global_step"] >= max_steps
            ):
                self.save_checkpoint()
                self.heartbeat("paused")
                return {
                    "status": "paused",
                    "route": self.route,
                    "global_step": self.progress["global_step"],
                }
            self.synchronize_phase()
            if self.progress["order"] is None:
                self.progress["order"] = epoch_order(
                    train_count, self.config.seed, self.progress["completed_epoch"] + 1
                )
            order = self.progress["order"]
            assert order is not None
            start = self.progress["offset"]
            if start < train_count:
                indices = order[start : start + self.config.batch_size].cuda()
                numbers = self.train_step(indices)
                self.progress["offset"] += len(indices)
                self.progress["global_step"] += 1
                for index, number in enumerate(numbers):
                    self.progress["loss_sums"][index] += number * len(indices)
                self.heartbeat("training", batch_loss=numbers[0])
                if self.progress["global_step"] % self.config.checkpoint_steps == 0:
                    self.save_checkpoint()
            if self.progress["offset"] == train_count:
                self.finish_epoch(train_count)
                if self.progress["completed_epoch"] == self.config.concept_epochs:
                    self.endpoint()
        record = self.endpoint()
        self.save_checkpoint()
        # Restore a missing history export after interruption between atomic writes.
        atomic_json(self.output / "history.json", self.progress["history"])
        result = {
            **record,
            "status": "complete",
            "resume_sha256": sha256(self.output / "resume.pt"),
            "history_sha256": sha256(self.output / "history.json"),
            "endpoint_records_sha256": {
                str(epoch): sha256(
                    self.output / "endpoints" / f"epoch_{epoch:04d}.json"
                )
                for epoch in (self.config.concept_epochs, self.config.epochs)
            },
        }
        atomic_json(self.output / "result.json", result)
        self.heartbeat("complete", validation=record["validation"])
        return result
