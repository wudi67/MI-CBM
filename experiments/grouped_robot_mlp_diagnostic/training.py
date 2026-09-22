"""True-concept-only optimization with exact mid-epoch Adam/RNG recovery."""

import math
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
)
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .model import ConceptMLP, predict, scores


class Route:
    def __init__(self, shared) -> None:
        self.shared, self.config = shared, shared.config
        self.output = shared.output / "training"
        self.output.mkdir(parents=True, exist_ok=True)
        self.model = ConceptMLP().cuda()
        self.model.load_state_dict(shared.initial["model"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.progress: dict = {
            "epoch": 0,
            "step": 0,
            "order": None,
            "offset": 0,
            "loss_sum": 0.0,
            "history": [],
        }
        self.training_seconds = 0.0
        self.gradient_check = {}
        restore_rng(shared.initial["rng"])
        if (self.output / "resume.pt").exists():
            checkpoint = load(self.output / "resume.pt")
            self.check_identity(checkpoint)
            payload = {k: v for k, v in checkpoint.items() if k != "payload_sha256"}
            if tree_hash(payload) != checkpoint["payload_sha256"]:
                raise ValueError("Resume checkpoint payload changed")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.progress = checkpoint["progress"]
            self.training_seconds = checkpoint["training_seconds"]
            self.gradient_check = checkpoint["gradient_check"]
            restore_rng(checkpoint["rng"])
            self.check_progress()

    def identity(self) -> dict:
        return {
            "manifest_sha256": self.shared.manifest_hash,
            "data_lock_sha256": self.shared.data_hash,
            "initial_model_sha256": self.shared.initial_hash,
            "order_seed": self.shared.order_seed,
            "order_epoch_offset": self.shared.order_offset,
        }

    def check_identity(self, checkpoint: dict) -> None:
        if any(checkpoint.get(k) != v for k, v in self.identity().items()):
            raise ValueError("Training checkpoint identity changed")

    def check_progress(self) -> None:
        count = len(self.shared.data["train"]["labels"])
        progress = self.progress
        batches = math.ceil(count / self.config.batch_size)
        if not 0 <= progress["epoch"] <= self.config.epochs:
            raise ValueError("Invalid resumed epoch")
        if progress["step"] != progress["epoch"] * batches + math.ceil(
            progress["offset"] / self.config.batch_size
        ):
            raise ValueError("Inconsistent resumed sample offset / Adam steps")
        if progress["order"] is not None:
            expected = epoch_order(
                count,
                self.shared.order_seed,
                self.shared.order_offset + progress["epoch"] + 1,
            )
            if (
                not torch.equal(progress["order"], expected)
                or not 0 <= progress["offset"] <= count
            ):
                raise ValueError("Resumed sample permutation changed")
        elif progress["offset"] != 0:
            raise ValueError("Sample offset without a permutation")
        if len(progress["history"]) != progress["epoch"]:
            raise ValueError("Incomplete resumed history")
        if progress["step"] and {
            int(v["step"]) for v in self.optimizer.state.values()
        } != {progress["step"]}:
            raise ValueError("Resumed Adam moments have the wrong step count")

    def checkpoint(self) -> dict:
        value = {
            **self.identity(),
            "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "rng": rng_state(),
            "progress": self.progress,
            "training_seconds": self.training_seconds,
            "gradient_check": self.gradient_check,
        }
        return {**value, "payload_sha256": tree_hash(value)}

    def save(self) -> None:
        atomic_checkpoint(self.output / "resume.pt", self.checkpoint())

    def step(self, indices: torch.Tensor) -> float:
        data = self.shared.data["train"]
        self.model.train()
        torch.cuda.synchronize()
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        # Independent: neither predicted concepts nor validation labels enter this loss.
        loss = F.binary_cross_entropy_with_logits(
            self.model(data["concepts"][indices]), data["labels"][indices].float()
        )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True
        )
        if not self.gradient_check:
            if any(p.grad is None for p in self.model.parameters()) or float(norm) <= 0:
                raise RuntimeError("Missing CUDA gradient")
            self.gradient_check = {
                "parameter_device": str(next(self.model.parameters()).device),
                "input_device": str(data["concepts"].device),
                "target_device": str(data["labels"].device),
                "gradient_norm": float(norm),
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "input": "training true binary concepts only",
                "frontend_trained": False,
            }
        value = float(loss.detach())
        if not math.isfinite(value):
            raise ValueError("Nonfinite MLP loss")
        self.optimizer.step()
        torch.cuda.synchronize()
        self.training_seconds += time.perf_counter() - started
        return value

    def finish_epoch(self) -> None:
        count = len(self.shared.data["train"]["labels"])
        epoch = self.progress["epoch"] + 1
        row: dict = {
            "epoch": epoch,
            "step": self.progress["step"],
            "train_batch_bce": self.progress["loss_sum"] / count,
            "order_sha256": array_hash(self.progress["order"].numpy()),
        }
        if epoch % self.config.diagnostic_every == 0 or epoch == self.config.epochs:
            for role in ("train", "validation"):
                row[role] = scores(
                    predict(self.model, self.shared.data[role]),
                    self.shared.cpu_data[role]["labels"],
                )
            report(
                f"MLP epoch {epoch}/{self.config.epochs} | "
                f"val normal {row['validation']['measured']['accuracy']:.2%} | "
                "val all corrected "
                f"{row['validation']['correct_all_five']['accuracy']:.2%} | "
                f"train true BCE {row['train']['direct_true']['bce']:.5f}"
            )
        self.progress["history"].append(row)
        self.progress.update(epoch=epoch, order=None, offset=0, loss_sum=0.0)
        self.save()
        atomic_json(self.output / "history.json", self.progress["history"])
        details = {"validation": row["validation"]} if "validation" in row else {}
        self.shared.heartbeat(
            "training",
            epoch_completed=epoch,
            global_step=self.progress["step"],
            offset=0,
            **details,
        )

    def run(self, max_steps: int | None = None) -> None:
        limit = None if max_steps is None else self.progress["step"] + max_steps
        count = len(self.shared.data["train"]["labels"])
        try:
            while self.progress["epoch"] < self.config.epochs:
                if self.shared.control["stop"] or (
                    limit is not None and self.progress["step"] >= limit
                ):
                    raise InterruptedError("Paused at a safe MLP step; use --resume")
                if self.progress["order"] is None:
                    self.progress["order"] = epoch_order(
                        count,
                        self.shared.order_seed,
                        self.shared.order_offset + self.progress["epoch"] + 1,
                    )
                start = self.progress["offset"]
                indices = self.progress["order"][
                    start : start + self.config.batch_size
                ].cuda()
                if len(indices):
                    value = self.step(indices)
                    self.progress["step"] += 1
                    self.progress["offset"] += len(indices)
                    self.progress["loss_sum"] += value * len(indices)
                    self.shared.heartbeat(
                        "training",
                        epoch_completed=self.progress["epoch"],
                        global_step=self.progress["step"],
                        offset=self.progress["offset"],
                        batch_loss=value,
                    )
                    if self.progress["step"] % self.config.checkpoint_steps == 0:
                        self.save()
                if self.progress["offset"] == count:
                    self.finish_epoch()
        except InterruptedError:
            self.save()
            raise
        self.check_progress()
        self.save()
        atomic_checkpoint(self.output / "endpoint.pt", self.checkpoint())
        atomic_json(self.output / "cuda_gradient_check.json", self.gradient_check)
        atomic_json(
            self.output / "result.json",
            {
                **self.identity(),
                "epochs": self.config.epochs,
                "step": self.progress["step"],
                "model_sha256": state_hash(self.model.state_dict()),
                "parameters": 113,
                "training_seconds": self.training_seconds,
                "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "test_evaluated": False,
            },
        )
        atomic_json(
            self.output / "completion_lock.json",
            {
                **self.identity(),
                "artifacts": {
                    n: sha256(self.output / n)
                    for n in (
                        "endpoint.pt",
                        "resume.pt",
                        "result.json",
                        "history.json",
                        "cuda_gradient_check.json",
                    )
                },
            },
        )


def verify_complete(shared) -> dict:
    directory = shared.output / "training"
    locked = read_json(directory / "completion_lock.json")
    verify_files(directory, locked["artifacts"])
    route = Route(shared)
    route.check_identity(locked)
    endpoint = load(directory / "endpoint.pt")
    if tree_hash(endpoint) != tree_hash(load(directory / "resume.pt")):
        raise ValueError("Endpoint and resume states disagree")
    result = read_json(directory / "result.json")
    history = read_json(directory / "history.json")
    count = len(shared.data["train"]["labels"])
    orders = [
        array_hash(
            epoch_order(count, shared.order_seed, shared.order_offset + e).numpy()
        )
        for e in range(1, shared.config.epochs + 1)
    ]
    if (
        route.progress["epoch"] != shared.config.epochs
        or route.progress["step"]
        != shared.config.epochs * math.ceil(count / shared.config.batch_size)
        or route.progress["history"] != history
        or [r["epoch"] for r in history] != list(range(1, shared.config.epochs + 1))
        or [r["order_sha256"] for r in history] != orders
        or result["model_sha256"] != state_hash(endpoint["model"])
        or result["parameters"] != sum(p.numel() for p in route.model.parameters())
        or result["test_evaluated"]
    ):
        raise ValueError("Completed MLP provenance or training budget changed")
    return result
