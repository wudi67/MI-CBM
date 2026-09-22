"""CUDA requirements, atomic artifacts, source provenance and Rich reporting."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
console = Console(highlight=False)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def report(message: str) -> None:
    console.print(message)
    log_path = os.environ.get("GROUPED_VQC_LOG")
    if log_path:
        with Path(log_path).open("a", encoding="utf-8") as stream:
            Console(file=stream, width=150, highlight=False).print(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((contiguous.shape, contiguous.dtype.str)).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    """Pin new implementation and the actual shared quantum implementation."""
    sources = [ROOT / "FusionModel.py"]
    sources.extend(PACKAGE.rglob("*.py"))
    sources.extend((PACKAGE / "scripts").glob("*.sh"))
    sources.extend((ROOT / "torchquantum").rglob("*.py"))
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(sources)
        if "__pycache__" not in path.parts
    }


def atomic_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".json-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_checkpoint(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(content, stream)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def cuda_runtime(seed: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run scripts with the VQC environment")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(properties.name),
        "device_index": 0,
        "total_memory_gib": properties.total_memory / 2**30,
        "state_dtype": "complex64",
        "parameter_dtype": "float32",
        "tf32": False,
        "amp": False,
        "deterministic_algorithms": True,
    }


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
