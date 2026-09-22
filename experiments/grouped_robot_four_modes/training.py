"""Reuse Robot circuits and the original resumable loop; train both modules."""

import math
import time
from dataclasses import replace

import torch
import torch.nn.functional as F

from experiments.grouped_dynamic_vqc.runtime import (
    array_hash,
    atomic_json,
    report,
    restore_rng,
    sha256,
)
from experiments.grouped_robot_independent.model import make_model, module_state
from experiments.grouped_robot_independent.training import Route as IndependentRoute
from experiments.grouped_robot_pilot.evaluation import evaluate
from experiments.grouped_robot_pilot.model import concept_loss, forward_control
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import Route as PilotRoute
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .protocol import CELLS


class Job:
    def __init__(self, parent, seed: int, cell: str):
        if cell not in CELLS:
            raise ValueError("Unknown full-circuit training cell")
        self.parent, self.seed, self.cell = parent, seed, cell
        self.name = f"seed_{seed}/{cell}"
        self.output = parent.output / f"seed_{seed}"
        self.config = replace(
            parent.pilot_config,
            seed=seed,
            checkpoint_steps=parent.config.checkpoint_steps,
        )
        self.data, self.data_hash = parent.data, parent.data_hash
        self.manifest_hash = parent.manifest_hash

    @property
    def checkpoint_path(self):
        return self.output / f"training/{self.cell}/endpoint.pt"

    @property
    def stop_requested(self) -> bool:
        return self.parent.control["stop"]

    def initial_for(self, cell: str) -> dict:
        if cell != self.cell:
            raise ValueError("Job route mismatch")
        # ORIGINAL uniform frontend, never the concept-trained frontend.
        return self.parent.initial[str(self.seed)]

    def heartbeat(self, status: str, **details) -> None:
        self.parent.heartbeat(status, **details)


def identity(job: Job) -> dict:
    return {
        "manifest_sha256": job.manifest_hash,
        "data_lock_sha256": job.data_hash,
        "cell": job.cell,
        "seed": job.seed,
        "head_layers": 5,
        "initial_model_sha256": state_hash(job.initial_for(job.cell)["model"]),
        "epochs": job.parent.config.epochs,
        "epoch_offset": 0,
        "training_control": "zero" if job.cell == "joint_no_feedback" else "measured",
        "concept_weight": 0.0
        if job.cell == "standard"
        else job.parent.config.concept_weight,
        "label_weight": job.parent.config.label_weight,
        "trainable_modules": ["frontend", "label_head"],
    }


class Route(PilotRoute):
    """Inherit run/save/checkpoint scheduling; no cached/detached training states."""

    verify_progress = IndependentRoute.verify_progress

    def __init__(self, shared: Job):  # pylint: disable=super-init-not-called
        self.shared, self.config, self.cell = shared, shared.config, shared.cell
        self.output = shared.output / f"training/{self.cell}"
        self.output.mkdir(parents=True, exist_ok=True)
        self.epochs, self.epoch_offset = shared.parent.config.epochs, 0
        initial = shared.initial_for(self.cell)
        self.model = make_model(initial["model"])
        self.model.requires_grad_(True)
        restore_rng(initial["rng"])
        self.parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.config.learning_rate)
        self.frozen_module = ""  # Neither module is frozen; used by gradient checker.
        self.progress = {
            "completed_epoch": 0,
            "global_step": 0,
            "order": None,
            "offset": 0,
            "loss_sums": [0.0],
            "history": [],
        }
        self.component_sums = [0.0, 0.0]
        self.gradient_checks: dict = {}
        self.training_seconds = 0.0
        self.protocol_identity = identity(shared)
        if (self.output / "resume.pt").exists():
            saved = load(self.output / "resume.pt")
            if any(saved.get(k) != v for k, v in self.identity().items()):
                raise ValueError("Resume full-circuit identity/control/loss changed")
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.progress, self.gradient_checks = (
                saved["progress"],
                saved["gradient_checks"],
            )
            self.component_sums = saved["component_sums"]
            self.training_seconds = saved["training_seconds"]
            restore_rng(saved["rng"])
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
        self.verify_progress()
        if (
            len(self.component_sums) != 2
            or any(not math.isfinite(v) or v < 0 for v in self.component_sums)
            or (not self.progress["offset"] and self.component_sums != [0.0, 0.0])
        ):
            raise ValueError("Resume component loss accumulators changed")
        torch.cuda.reset_peak_memory_stats()

    def identity(self) -> dict:
        return self.protocol_identity

    def verify_frozen(self) -> None:
        """The inherited checkpoint hook verifies the opposite contract here."""
        if sum(p.numel() for p in self.parameters) != 352 or any(
            not p.requires_grad or not p.is_cuda for p in self.parameters
        ):
            raise ValueError("Both circuits (352 parameters) must train on CUDA")

    def checkpoint(self) -> dict:
        return {**super().checkpoint(), "component_sums": self.component_sums}

    def check_gradients(self) -> None:
        super().check_gradients()
        self.gradient_checks["active_parameters"] = sum(
            p.numel() for p in self.parameters
        )
        self.gradient_checks["layer_gradient_l2"] = {
            name: [float(layer.norm()) for layer in tensor.grad]
            for name, tensor in self.model.named_parameters()
            if tensor.grad is not None and tensor.ndim > 1
        }
        atomic_json(self.output / "gradient_checks.json", self.gradient_checks)

    def step(self, indices: torch.Tensor) -> float:
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data = self.shared.data["train"]
        # Recompute the uploaded state WITH gradients on every minibatch.
        state = self.model.frontend(data["angles"][indices])
        output = forward_control(
            self.model,
            state,
            data["concepts"][indices],
            zero=self.cell == "joint_no_feedback",
        )
        label = F.binary_cross_entropy(
            output["label_prob"].clamp(1e-7, 1 - 1e-7), data["labels"][indices]
        )
        weight = self.protocol_identity["concept_weight"]
        # Standard NLL is logged only: it cannot send concept gradients.
        probabilities = output["concept_probs"]
        concept = concept_loss(
            probabilities if weight else probabilities.detach(),
            data["concepts"][indices],
        )
        loss = self.protocol_identity["label_weight"] * label
        if weight:
            loss = loss + weight * concept
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite full-circuit loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters, self.config.grad_clip, error_if_nonfinite=True
        )
        if not self.gradient_checks:
            self.check_gradients()
        self.optimizer.step()
        torch.cuda.synchronize()
        self.training_seconds += time.perf_counter() - started
        self.component_sums[0] += float(concept.detach()) * len(indices)
        self.component_sums[1] += float(label.detach()) * len(indices)
        return float(loss.detach())

    def finish_epoch(self, count: int) -> None:
        self.verify_frozen()
        self.heartbeat("validation")
        validation, _ = evaluate(
            self.model,
            self.shared.data["validation"],
            self.config.eval_batch_size,
            zero=self.cell == "joint_no_feedback",
            tick=self.shared.heartbeat,
        )
        order = self.progress["order"]
        assert order is not None
        epoch = self.progress["completed_epoch"] + 1
        self.progress["history"].append(
            {
                "epoch": epoch,
                "order_epoch": epoch,
                "global_step": self.progress["global_step"],
                "order_sha256": array_hash(order.numpy()),
                "train_loss": self.progress["loss_sums"][0] / count,
                "train_concept_nll": self.component_sums[0] / count,
                "train_label_bce": self.component_sums[1] / count,
                "concept_supervised": self.cell != "standard",
                "validation": validation,
            }
        )
        self.progress.update(
            completed_epoch=epoch, order=None, offset=0, loss_sums=[0.0]
        )
        self.component_sums = [0.0, 0.0]
        self.save()
        atomic_json(self.output / "history.json", self.progress["history"])
        report(
            f"{self.shared.name} {epoch}/{self.epochs} | "
            f"label={validation['label']['accuracy']:.2%} | "
            f"concept NLL={validation['concept']['joint_nll']:.4f}"
            + (" (unsupervised diagnostic)" if self.cell == "standard" else "")
        )
        self.heartbeat("between_epochs", validation=validation)


def verify_job(job: Job) -> dict:
    directory = job.checkpoint_path.parent
    result = read_json(directory / "result.json")
    verify_files(directory, result["artifacts"])
    saved = load(job.checkpoint_path)
    expected = identity(job)
    p, history = saved["progress"], read_json(directory / "history.json")
    count = len(job.data["train"]["angles"])
    steps_per_epoch = math.ceil(count / job.config.batch_size)
    steps = steps_per_epoch * expected["epochs"]
    if (
        any(saved.get(k) != v or result.get(k) != v for k, v in expected.items())
        or result["status"] != "complete"
        or result["test_evaluated"]
        or result["global_step"] != steps
        or p["global_step"] != steps
        or p["completed_epoch"] != expected["epochs"]
        or p["order"] is not None
        or p["offset"] != 0
        or p["history"] != history
        or len(history) != expected["epochs"]
        or p["loss_sums"] != [0.0]
        or saved["component_sums"] != [0.0, 0.0]
    ):
        raise ValueError("Full-circuit endpoint identity/budget/epoch boundary changed")
    optimizer = saved["optimizer"]
    if {int(v["step"]) for v in optimizer["state"].values()} != {steps} or len(
        optimizer["state"]
    ) != len(saved["model"]):
        raise ValueError("Full-circuit Adam state/budget changed")
    for epoch, row in enumerate(history, 1):
        if (
            row["epoch"] != epoch
            or row["order_epoch"] != epoch
            or row["global_step"] != epoch * steps_per_epoch
            or row["order_sha256"]
            != array_hash(epoch_order(count, job.seed, epoch).numpy())
        ):
            raise ValueError("Full-circuit sample order changed")
    for prefix, key in (("frontend", "frontend_sha256"), ("label_head", "head_sha256")):
        if state_hash(module_state(saved["model"], prefix)) != result[key]:
            raise ValueError("Endpoint circuit weights changed")
    if state_hash(saved["model"]) != result["model_sha256"]:
        raise ValueError("Endpoint model hash changed")
    gradients = saved["gradient_checks"]
    if gradients["active_parameters"] != 352 or any(
        gradients["gradient_l2"][name] is None
        or not math.isfinite(gradients["gradient_l2"][name])
        or gradients["gradient_l2"][name] <= 0
        for name in ("frontend", "label_head")
    ):
        raise ValueError("Both circuit modules must receive finite nonzero gradients")
    reference = job.parent.reference.training
    reference_steps = (
        reference[job.seed, "concept"]["global_step"]
        + reference[job.seed, "independent"]["global_step"]
    )
    if steps != reference_steps:
        raise ValueError("Total optimizer-update budget differs from staged modes")
    record = {
        **expected,
        "reused": False,
        "checkpoint_path": str(job.checkpoint_path),
        "checkpoint_sha256": sha256(job.checkpoint_path),
        "global_step": steps,
        "reference_total_updates": reference_steps,
        "model_sha256": result["model_sha256"],
        "history_order_sha256": [r["order_sha256"] for r in history],
    }
    job.parent.save_once(
        f"seed_{job.seed}/training/{job.cell}/training_reference.json", record
    )
    return record


def verify_pair(left: dict, right: dict) -> None:
    for key in (
        "seed",
        "initial_model_sha256",
        "epochs",
        "epoch_offset",
        "global_step",
        "history_order_sha256",
    ):
        if left[key] != right[key]:
            raise ValueError(f"Full-circuit routes are not paired: {key}")
