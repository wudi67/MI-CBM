"""Configuration, CUDA placement, provenance and atomic experiment files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
CONSOLE = Console(highlight=False, stderr=bool(os.environ.get("DYNAMIC_VQC_LOG")))


def report(*objects, **kwargs) -> None:
    """Send human messages through the live console and a separate plain log."""
    CONSOLE.print(*objects, **kwargs)
    log_path = os.environ.get("DYNAMIC_VQC_LOG")
    if log_path:
        with Path(log_path).open("a", encoding="utf-8") as stream:
            Console(file=stream, highlight=False, width=160).print(*objects, **kwargs)


class RichArgumentParser(argparse.ArgumentParser):
    """Use Rich for argparse help and diagnostics too."""

    def _print_message(self, message, file=None):
        if message:
            console = CONSOLE if file is None else Console(file=file, highlight=False)
            console.print(message, end="", markup=False)


@dataclass(frozen=True)
class E0Config:
    dataset: str = "data/dsprites/confirmatory_2027/dsprites_compact_c_32.npz"
    seeds: tuple[int, ...] = (0, 1, 2)
    learning_rates: tuple[float, ...] = (0.01, 0.03)
    steps: int = 500
    eval_every: int = 10
    checkpoint_every: int = 50
    split_seed: int = 20260906
    selection_fraction: float = 0.5
    device: str = "cuda:0"
    frontend_layers: int = 2
    frontend_direction: str = "alternating"
    persistent_feedback: bool = False
    oracle_accuracy_threshold: float = 0.99

    def validate(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if not self.learning_rates or len(set(self.learning_rates)) != len(
            self.learning_rates
        ):
            raise ValueError("learning rates must be nonempty and unique")
        if any(not np.isfinite(lr) or lr <= 0 for lr in self.learning_rates):
            raise ValueError("Learning rates must be finite and positive")
        if (
            min(
                self.steps, self.eval_every, self.checkpoint_every, self.frontend_layers
            )
            < 1
        ):
            raise ValueError("Training intervals and frontend depth must be positive")
        if not 0 < self.selection_fraction < 1:
            raise ValueError("selection_fraction must lie between zero and one")
        if self.frontend_direction not in {"fixed", "alternating"}:
            raise ValueError("Unsupported CU3 direction")
        if not 0 < self.oracle_accuracy_threshold <= 1:
            raise ValueError("Invalid oracle accuracy threshold")
        if torch.device(self.device).type != "cuda":
            raise ValueError("E0 training requires CUDA; no silent CPU fallback")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


def training_parser() -> RichArgumentParser:
    parser = RichArgumentParser(description="Dynamic VQC E0 oracle-concept experiment")
    parser.add_argument("--config", type=Path, default=PACKAGE / "configs/e0.json")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/dynamic_vqc/e0"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--learning-rates", type=float, nargs="+")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--device")
    parser.add_argument("--frontend-layers", type=int)
    parser.add_argument("--frontend-direction", choices=["fixed", "alternating"])
    parser.add_argument(
        "--persistent-feedback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store the edit switch in the interface checkpoint (E0 trains heads only)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> E0Config:
    values = json.loads(args.config.read_text(encoding="utf-8"))
    known = {item.name for item in fields(E0Config)}
    if set(values) - known:
        raise ValueError(f"Unknown configuration keys: {sorted(set(values) - known)}")
    for name in known:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    for name in ("seeds", "learning_rates"):
        if name in values:
            values[name] = tuple(values[name])
    dataset = Path(values.get("dataset", E0Config.dataset)).expanduser()
    values["dataset"] = str((ROOT / dataset).resolve())
    config = E0Config(**values)
    config.validate()
    return config


def cuda_runtime(device_name: str) -> tuple[torch.device, dict]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if Path(sys.prefix).name != "VQC":
        raise RuntimeError("Run this experiment using the VQC environment")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A working CUDA device is required")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    # Complex64 statevectors / float32 parameters; no AMP/TF32 approximation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return device, {
        "python": sys.executable,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "state_dtype": "complex64",
        "parameter_dtype": "float32",
        "amp": False,
        "tf32": False,
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    paths = (
        sorted(PACKAGE.glob("*.py"))
        + sorted((PACKAGE / "scripts").glob("*.sh"))
        + [
            ROOT / "FusionModel.py",
            ROOT / "torchquantum/functional/u3.py",
            ROOT / "torchquantum/functional/rx.py",
            ROOT / "torchquantum/functional/ry.py",
            ROOT / "torchquantum/functional/rz.py",
            ROOT / "torchquantum/functional/gate_wrapper.py",
            ROOT / "torchquantum/device/devices.py",
            ROOT / "torchquantum/encoding/encodings.py",
        ]
    )
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    torch.save(payload, temporary)
    temporary.replace(path)


def cpu_state(model: torch.nn.Module) -> dict:
    return {
        name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for name, value in model.state_dict().items()
    }
