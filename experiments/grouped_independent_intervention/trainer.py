"""CUDA training with paired sample order and exact mid-epoch Adam recovery."""

from __future__ import annotations

import math
import time

import torch

from experiments.grouped_control_diagnostics.protocol import module_hash
from experiments.grouped_dynamic_vqc.objectives import loss_function
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
from experiments.grouped_feedback_ablation.protocol import load_checkpoint
from experiments.grouped_sequential_intervention.evaluation import forward_control
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .evaluation import evaluate_exact as evaluate
from .protocol import Experiment


class IndependentRun:
    """True concept records train only the retained-state classification circuit."""

    def __init__(self, shared: Experiment, seed: int) -> None:
        cell = f"independent/seed{seed}"
        self.shared, self.config, self.cell = shared, shared.config, cell
        self.spec = shared.spec(seed)
        self.phase, self.mode = self.spec["phase"], self.spec["control_mode"]
        self.output = shared.output / cell
        self.output.mkdir(parents=True, exist_ok=True)
        initial = shared.initial_for(seed)
        self.initial_hash = state_hash(initial["model"])
        self.initial_frontend = module_hash(initial["model"], "frontend")
        self.initial_head = module_hash(initial["model"], "label_head")
        self.model = shared.make_model(initial["model"])
        restore_rng(initial["rng"])
        self.model.frontend.requires_grad_(False)
        self.model.label_head.requires_grad_(True)
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )
        self.progress: ProgressState = {
            "completed_epoch": 0,
            "global_step": 0,
            "order": None,
            "offset": 0,
            "loss_sums": [0.0, 0.0, 0.0],
            "history": [],
        }
        self.gradient_checks: dict = {}
        self.training_seconds = 0.0
        if (self.output / "resume.pt").exists():
            checkpoint = load_checkpoint(self.output / "resume.pt")
            if (
                checkpoint["manifest_sha256"] != shared.manifest_hash
                or checkpoint["cell"] != cell
                or checkpoint["spec"] != self.spec
                or checkpoint["initial_model_sha256"] != self.initial_hash
            ):
                raise ValueError("Resume checkpoint protocol/initialization mismatch")
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
                    "initial_model_sha256": self.initial_hash,
                    "initial_frontend_sha256": self.initial_frontend,
                    "initial_head_sha256": self.initial_head,
                    "manifest_sha256": shared.manifest_hash,
                    "fresh_adam": True,
                },
            )
            self.save()
        self.verify_frozen()
        shared.prepare_states(self.model)
        torch.cuda.reset_peak_memory_stats()

    def verify_frozen(self) -> None:
        if state_hash(self.model.frontend.state_dict()) != self.initial_frontend:
            raise RuntimeError("Frozen frontend changed")

    def checkpoint(self) -> dict:
        self.verify_frozen()
        return {
            "manifest_sha256": self.shared.manifest_hash,
            "cell": self.cell,
            "spec": self.spec,
            "initial_model_sha256": self.initial_hash,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": rng_state(),
            "progress": self.progress,
            "gradient_checks": self.gradient_checks,
            "training_seconds": self.training_seconds,
        }

    def save(self) -> None:
        atomic_checkpoint(self.output / "resume.pt", self.checkpoint())

    def heartbeat(self, status: str, **details: object) -> None:
        self.shared.heartbeat(
            status,
            cell=self.cell,
            phase=self.phase,
            control_mode=self.mode,
            epoch_completed=self.progress["completed_epoch"],
            epochs_total=self.spec["epochs"],
            global_step=self.progress["global_step"],
            offset=self.progress["offset"],
            training_seconds=self.training_seconds,
            **details,
        )

    def check_gradients(self) -> None:
        norms = {}
        for name, module in (
            ("frontend", self.model.frontend),
            ("label_head", self.model.label_head),
        ):
            parameters = list(module.parameters())
            if any(not p.is_cuda for p in parameters):
                raise RuntimeError("CUDA parameters required")
            if parameters[0].requires_grad:
                if any(
                    p.grad is None
                    or not p.grad.is_cuda
                    or not torch.isfinite(p.grad).all()
                    for p in parameters
                ):
                    raise RuntimeError(f"Missing/nonfinite CUDA gradient: {name}")
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
                    raise RuntimeError(f"Frozen parameter received a gradient: {name}")
                norms[name] = None
        self.gradient_checks = {
            "device": str(next(self.model.parameters()).device),
            "phase": self.phase,
            "control_mode": self.mode,
            "gradient_l2": norms,
        }
        atomic_json(self.output / "gradient_checks.json", self.gradient_checks)

    def train_step(self, indices: torch.Tensor) -> tuple[float, float, float]:
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data = self.shared.data["train"]
        states = self.shared.state_cache["train"][indices]
        output = forward_control(
            self.model, states, self.mode, data["concepts"][indices]
        )
        losses = loss_function(
            output,
            data["concepts"][indices],
            data["labels"][indices],
            1.0,
            1.0,
        )
        total = losses[2]
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config.grad_clip,
            error_if_nonfinite=True,
        )
        if not self.gradient_checks:
            self.check_gradients()
        numbers = (
            float(total.detach()),
            float(losses[1].detach()),
            float(losses[2].detach()),
        )
        if not all(math.isfinite(v) for v in numbers):
            raise RuntimeError("Nonfinite training loss")
        self.optimizer.step()
        torch.cuda.synchronize()
        self.training_seconds += time.perf_counter() - started
        return numbers

    def finish_epoch(self, count: int) -> None:
        self.verify_frozen()
        self.heartbeat("validation")
        validation, _ = evaluate(
            self.model,
            self.shared.data["val"],
            self.config.eval_batch_size,
            self.mode,
            cached_states=self.shared.state_cache["val"],
        )
        unassisted, _ = evaluate(
            self.model,
            self.shared.data["val"],
            self.config.eval_batch_size,
            "measured",
            cached_states=self.shared.state_cache["val"],
        )
        epoch = self.progress["completed_epoch"] + 1
        order = self.progress["order"]
        assert order is not None
        self.progress["history"].append(
            {
                "epoch": epoch,
                "order_epoch": epoch + self.spec["offset"],
                "order_sha256": array_hash(order.numpy()),
                "phase": self.phase,
                "control_mode": self.mode,
                "global_step": self.progress["global_step"],
                "train_loss": self.progress["loss_sums"][0] / count,
                "train_concept_nll": self.progress["loss_sums"][1] / count,
                "train_label_bce": self.progress["loss_sums"][2] / count,
                "validation": validation,
                "validation_unassisted": unassisted,
            }
        )
        self.progress.update(
            completed_epoch=epoch, order=None, offset=0, loss_sums=[0.0, 0.0, 0.0]
        )
        self.save()
        atomic_json(self.output / "history.json", self.progress["history"])
        report(
            f"{self.cell} {epoch}/{self.spec['epochs']} | "
            f"真实概念 Label {validation['label']['accuracy']:.2%} | "
            f"预测概念 Label {unassisted['label']['accuracy']:.2%} | "
            "Concept group exact "
            f"{validation['concept']['group_argmax_exact_accuracy']:.2%}"
        )

    def run(self, max_steps: int | None = None) -> dict:
        count = len(self.shared.data["train"]["angles"])
        while self.progress["completed_epoch"] < self.spec["epochs"]:
            if self.shared.stop_requested or (
                max_steps is not None and self.progress["global_step"] >= max_steps
            ):
                self.save()
                self.heartbeat("paused")
                return {"status": "paused", "cell": self.cell}
            if self.progress["order"] is None:
                self.progress["order"] = epoch_order(
                    count,
                    self.spec["seed"],
                    self.progress["completed_epoch"] + 1 + self.spec["offset"],
                )
            order = self.progress["order"]
            assert order is not None
            start = self.progress["offset"]
            if start < count:
                indices = order[start : start + self.config.batch_size].cuda()
                losses = self.train_step(indices)
                self.progress["offset"] += len(indices)
                self.progress["global_step"] += 1
                for i, value in enumerate(losses):
                    self.progress["loss_sums"][i] += value * len(indices)
                self.heartbeat("training", batch_loss=losses[0])
                if self.progress["global_step"] % self.config.checkpoint_steps == 0:
                    self.save()
            if self.progress["offset"] == count:
                self.finish_epoch(count)
        self.save()
        atomic_json(self.output / "history.json", self.progress["history"])
        atomic_checkpoint(self.output / "endpoint.pt", self.checkpoint())
        result = {
            "status": "complete",
            "origin": "trained",
            "cell": self.cell,
            "spec": self.spec,
            "manifest_sha256": self.shared.manifest_hash,
            "initial_model_sha256": self.initial_hash,
            "model_sha256": state_hash(self.model.state_dict()),
            "frontend_sha256": state_hash(self.model.frontend.state_dict()),
            "head_sha256": state_hash(self.model.label_head.state_dict()),
            "global_step": self.progress["global_step"],
            "training_seconds": self.training_seconds,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "test_evaluated": False,
            "artifacts": {
                name: sha256(self.output / name)
                for name in (
                    "resume.pt",
                    "endpoint.pt",
                    "initialization.json",
                    "gradient_checks.json",
                    "history.json",
                )
            },
        }
        atomic_json(self.output / "result.json", result)
        self.heartbeat("between_cells")
        return result
