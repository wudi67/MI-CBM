"""CUDA-only, resumable dSprites development training for grouped dynamic VQC."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import torch

from .data import load_cached_data, prepare_data
from .evaluation import evaluate
from .model import GroupedDynamicVQC
from .objectives import loss_function
from .runtime import (
    ROOT,
    atomic_checkpoint,
    atomic_json,
    cuda_runtime,
    report,
    restore_rng,
    rng_state,
    sha256,
    source_hashes,
    utc_now,
)


@dataclass(frozen=True)
class TrainConfig:
    dataset: str = str(
        ROOT / "data/dsprites/confirmatory_2027/dsprites_compact_c_32.npz"
    )
    admission: str = str(ROOT / "outputs/dsprites_confirmatory_2027_admission.json")
    front_layers: int = 4
    label_layers: int = 1
    batch_size: int = 1024
    eval_batch_size: int = 2048
    epochs: int = 100
    learning_rate: float = 0.01
    concept_weight: float = 1.0
    label_weight: float = 1.0
    grad_clip: float = 5.0
    train_limit: int = 0
    val_limit: int = 0
    seed: int = 0
    checkpoint_steps: int = 25
    shots: int = 256

    def validate(self) -> None:
        integers = (
            self.front_layers,
            self.label_layers,
            self.batch_size,
            self.eval_batch_size,
            self.epochs,
            self.checkpoint_steps,
            self.shots,
        )
        if min(integers) < 1 or self.seed < 0:
            raise ValueError("Positive depths, batch sizes, epochs and shots required")
        for value in (
            self.learning_rate,
            self.concept_weight,
            self.label_weight,
            self.grad_clip,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Learning rate, loss weights and gradient clip must be positive"
                )
        if any(
            limit < 0 or 0 < limit < 18 for limit in (self.train_limit, self.val_limit)
        ):
            raise ValueError("Subset limits must be zero (all rows) or at least 18")


class ProgressState(TypedDict):
    completed_epoch: int
    global_step: int
    order: torch.Tensor | None
    offset: int
    loss_sums: list[float]
    history: list[dict]


class TrainingRun:
    """Save model, Adam, RNG, permutation, offset and partial epoch totals."""

    def __init__(self, config: TrainConfig, output: Path, resume: bool = False) -> None:
        config.validate()
        self.config = config
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.stop_requested = False
        self.runtime = cuda_runtime(config.seed)
        self.sources = source_hashes()
        self.model = GroupedDynamicVQC(config.front_layers, config.label_layers).cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.progress: ProgressState = {
            "completed_epoch": 0,
            "global_step": 0,
            "order": None,
            "offset": 0,
            "loss_sums": [0.0, 0.0, 0.0],
            "history": [],
        }
        if resume:
            self.data, self.audit = load_cached_data(output)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            if (
                manifest["config"] != asdict(config)
                or manifest["sources"] != self.sources
            ):
                raise ValueError("Resume requires identical config and source hashes")
            if manifest["preprocessing_sha256"] != sha256(
                output / "preprocessing.json"
            ):
                raise ValueError("Preprocessing provenance changed")
            if manifest["runtime"] != self.runtime:
                raise ValueError("Resume requires the original CUDA/software runtime")
            checkpoint = torch.load(
                output / "resume.pt", map_location="cpu", weights_only=False
            )
            if checkpoint["manifest_sha256"] != sha256(output / "manifest.json"):
                raise ValueError("Checkpoint does not belong to this manifest")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.progress = checkpoint["progress"]
            restore_rng(checkpoint["rng"])
        else:
            if (output / "config.json").exists():
                raise FileExistsError(
                    "Run already exists; use --resume or a new output directory"
                )
            self.heartbeat("preprocessing")
            self.data, self.audit = prepare_data(
                Path(config.dataset),
                Path(config.admission),
                output,
                train_limit=config.train_limit,
                val_limit=config.val_limit,
                seed=config.seed,
            )
            atomic_json(output / "config.json", asdict(config))
            manifest = {
                "schema": "grouped_dynamic_vqc.v1",
                "created_at": utc_now(),
                "config": asdict(config),
                "sources": self.sources,
                "runtime": self.runtime,
                "preprocessing_sha256": sha256(output / "preprocessing.json"),
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "qubits": {
                    "uploaded": 10,
                    "measured_concepts": 5,
                    "retained": 5,
                    "readout": 1,
                },
                "concept_encoding": (
                    "MSB q0..q4: shape(2 bits) | scale(3 bits); code=8*shape+scale"
                ),
                "measurement": "exact Born branch mixture; all 32 outcomes retained",
                "loss": "joint 32-code concept NLL + binary label BCE",
                "architecture_boundary": (
                    "retains input-dependent quantum residual pathway"
                ),
                "evidence_role": "development validation; no test evaluation",
            }
            atomic_json(output / "manifest.json", manifest)
        self.data = {
            role: {key: tensor.cuda() for key, tensor in values.items()}
            for role, values in self.data.items()
        }
        self.manifest_hash = sha256(output / "manifest.json")
        if not resume:
            self.save_checkpoint()
        report(
            f"[cyan]Grouped Dynamic VQC[/cyan] | {self.runtime['device']} | "
            f"batch={config.batch_size} | "
            f"Lc={config.front_layers}, Ly={config.label_layers} | "
            f"parameters={sum(p.numel() for p in self.model.parameters())}"
        )

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True

    def heartbeat(self, status: str, **details: object) -> None:
        atomic_json(
            self.output / "heartbeat.json",
            {
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "status": status,
                "epoch_completed": self.progress["completed_epoch"],
                "epochs_total": self.config.epochs,
                "global_step": self.progress["global_step"],
                "offset": self.progress["offset"],
                "process_seconds": time.perf_counter() - self.started,
                "device": self.runtime["device"],
                **details,
            },
        )

    def save_checkpoint(self) -> None:
        atomic_checkpoint(
            self.output / "resume.pt",
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "progress": self.progress,
                "rng": rng_state(),
                "manifest_sha256": self.manifest_hash,
            },
        )

    def train_step(self, indices: torch.Tensor) -> tuple[float, float, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        train = self.data["train"]
        losses = loss_function(
            self.model(train["angles"][indices]),
            train["concepts"][indices],
            train["labels"][indices],
            self.config.concept_weight,
            self.config.label_weight,
        )
        losses[0].backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True
        )
        if self.progress["global_step"] == 0:
            norms = {}
            for name, module in (
                ("frontend", self.model.frontend),
                ("label_head", self.model.label_head),
            ):
                gradients = [parameter.grad for parameter in module.parameters()]
                if any(gradient is None for gradient in gradients):
                    raise RuntimeError(f"Missing {name} gradient")
                squared = sum(
                    gradient.abs().square().sum()
                    for gradient in gradients
                    if gradient is not None
                )
                norms[name] = float(torch.as_tensor(squared).sqrt())
                if not math.isfinite(norms[name]) or norms[name] <= 0:
                    raise RuntimeError(f"No finite nonzero {name} gradient")
            atomic_json(
                self.output / "cuda_gradient_check.json",
                {
                    "parameter_device": str(next(self.model.parameters()).device),
                    "input_device": str(train["angles"].device),
                    "gradient_l2_after_clip": norms,
                },
            )
        numbers = (
            float(losses[0].detach()),
            float(losses[1].detach()),
            float(losses[2].detach()),
        )
        if not all(math.isfinite(value) for value in numbers):
            raise RuntimeError("Nonfinite training loss")
        self.optimizer.step()
        return numbers

    def run(self, max_steps: int | None = None) -> dict:
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive")
        if max_steps is not None and self.progress["global_step"] >= max_steps:
            self.heartbeat("paused")
            return {"status": "paused", "global_step": self.progress["global_step"]}
        train_count = len(self.data["train"]["angles"])
        if not (self.output / "initial_validation.json").exists():
            self.heartbeat("initial_validation")
            initial = evaluate(
                self.model, self.data["val"], self.config.eval_batch_size
            )
            atomic_json(self.output / "initial_validation.json", initial)
        while self.progress["completed_epoch"] < self.config.epochs:
            if self.progress["order"] is None:
                self.progress["order"] = torch.randperm(train_count)
            order = self.progress["order"]
            assert order is not None
            while self.progress["offset"] < train_count:
                start = self.progress["offset"]
                indices = order[start : start + self.config.batch_size].cuda()
                numbers = self.train_step(indices)
                self.progress["offset"] += len(indices)
                self.progress["global_step"] += 1
                for index, value in enumerate(numbers):
                    self.progress["loss_sums"][index] += value * len(indices)
                self.heartbeat(
                    "training", batch_loss=numbers[0], train_rows=train_count
                )
                if self.progress["global_step"] % self.config.checkpoint_steps == 0:
                    self.save_checkpoint()
                if self.stop_requested or (
                    max_steps is not None and self.progress["global_step"] >= max_steps
                ):
                    self.save_checkpoint()
                    self.heartbeat("paused")
                    report(
                        "[yellow]Paused with resumable checkpoint:[/yellow] "
                        f"{self.output}"
                    )
                    return {
                        "status": "paused",
                        "global_step": self.progress["global_step"],
                    }
            self.heartbeat("validation")
            validation = evaluate(
                self.model, self.data["val"], self.config.eval_batch_size
            )
            self.progress["completed_epoch"] += 1
            epoch = self.progress["completed_epoch"]
            record = {
                "epoch": epoch,
                "global_step": self.progress["global_step"],
                "train_loss": self.progress["loss_sums"][0] / train_count,
                "train_concept_nll": self.progress["loss_sums"][1] / train_count,
                "train_label_bce": self.progress["loss_sums"][2] / train_count,
                "validation": validation,
            }
            self.progress["history"].append(record)
            self.progress["order"] = None
            self.progress["offset"] = 0
            self.progress["loss_sums"] = [0.0, 0.0, 0.0]
            self.save_checkpoint()
            atomic_json(self.output / "history.json", self.progress["history"])
            report(
                f"Epoch {epoch:3}/{self.config.epochs} | "
                f"train loss {record['train_loss']:.4f} | "
                f"val concept NLL {validation['concept']['joint_nll']:.4f} | "
                f"joint MAP {validation['concept']['joint_map_accuracy']:.2%} | "
                f"label {validation['label']['accuracy']:.2%}"
            )
        self.heartbeat("final_diagnostics")
        endpoint = evaluate(
            self.model,
            self.data["val"],
            self.config.eval_batch_size,
            diagnostics=True,
            shots=self.config.shots,
            seed=self.config.seed,
        )
        result = {
            "status": "complete",
            "evidence_role": "fixed final epoch, development validation only",
            "test_evaluated": False,
            "epochs": self.progress["completed_epoch"],
            "global_step": self.progress["global_step"],
            "train_rows": train_count,
            "runtime": self.runtime,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "manifest_sha256": self.manifest_hash,
            "checkpoint_sha256": sha256(self.output / "resume.pt"),
            "validation": endpoint,
        }
        atomic_json(self.output / "result.json", result)
        self.heartbeat("complete", validation=endpoint)
        report(f"[green]Completed.[/green] Result: {self.output / 'result.json'}")
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Pause after this total step; for development/recovery checks",
    )
    defaults = asdict(TrainConfig())
    for key, value in defaults.items():
        parser.add_argument(
            "--" + key.replace("_", "-"), type=type(value), default=None
        )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".worker.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another worker already owns this run directory"
            ) from error
        if args.resume:
            values = json.loads((output / "config.json").read_text(encoding="utf-8"))
        else:
            values = asdict(TrainConfig())
        for key in values:
            override = getattr(args, key)
            if override is not None:
                values[key] = override
        config = TrainConfig(**values)
        try:
            runner = TrainingRun(config, output, args.resume)
        except Exception as error:
            # The directory lock is already owned, so this cannot replace the
            # status of a different live worker. Report startup failures too.
            atomic_json(
                output / "heartbeat.json",
                {
                    "updated_at": utc_now(),
                    "pid": os.getpid(),
                    "status": "failed",
                    "epoch_completed": 0,
                    "epochs_total": config.epochs,
                    "global_step": 0,
                    "offset": 0,
                    "device": "initialization",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        previous = {
            signum: signal.signal(signum, runner.request_stop)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            runner.run(args.max_steps)
        except Exception as error:
            runner.heartbeat("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    main()
