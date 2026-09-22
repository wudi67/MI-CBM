"""Content locks, source snapshots and deterministic independent sample orders."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch

from ..runtime import ROOT, atomic_json, sha256
from .config import PACKAGE


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, allow_nan=False).encode())
        digest.update(b";")

    visit(value)
    return digest.hexdigest()


def epoch_order(size: int, seed: int, epoch: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 104729 * epoch)
    return torch.randperm(size, generator=generator)


def order_hash(indices: torch.Tensor, seed: int, epochs: int) -> str:
    digest = hashlib.sha256()
    rows = indices.cpu()
    for epoch in range(1, epochs + 1):
        digest.update(rows[epoch_order(len(rows), seed, epoch)].numpy().tobytes())
    return digest.hexdigest()


def verify_pins(directory: Path, pins: dict) -> None:
    for name, expected in pins.items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact changed or missing: {path}")


def lock_files(directory: Path, names: list[str], lock: str = "result_lock.json"):
    pins = {name: sha256(directory / name) for name in names}
    atomic_json(directory / lock, pins)
    return pins


def sources() -> dict:
    shared = PACKAGE.parent
    files = [
        path
        for path in PACKAGE.rglob("*")
        if path.suffix in (".py", ".sh", ".json", ".md")
    ]
    files += [
        shared / name
        for name in (
            "__init__.py",
            "circuits.py",
            "upload_circuit.py",
            "runtime.py",
            "progress.py",
            "status.py",
            "scripts/environment.sh",
        )
    ]
    files += [ROOT / "FusionModel.py", ROOT / "robot_dataset_schema.py"]
    files += list((ROOT / "torchquantum").rglob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(files))}


def manifest(output: Path, contract: dict, resume: bool) -> None:
    path = output / "manifest.json"
    if path.exists():
        if not resume:
            raise FileExistsError(
                "Run exists; use --resume or another output directory"
            )
        if read_json(path) != contract:
            raise ValueError("Resume rejected: config, source or dataset changed")
    elif resume:
        raise FileNotFoundError("--resume requires an existing manifest.json")
    else:
        atomic_json(path, contract)
    # Recover an interrupted initial snapshot without permitting changed sources.
    for name, expected in contract["source_sha256"].items():
        destination = output / "source_snapshot" / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        if sha256(destination) != expected:
            raise ValueError(f"Source snapshot changed: {destination}")
