"""Preserve Robot v5 train/validation rows and fit forty-feature routing on train."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from experiments.dynamic_vqc.robot.data import center_gray, load_tables
from experiments.grouped_dynamic_vqc.data import fit_routing
from experiments.grouped_dynamic_vqc.runtime import (
    array_hash,
    atomic_checkpoint,
    atomic_json,
    sha256,
)
from robot_dataset_schema import SPLIT_FILES

from .model import codes
from .protocol import Config, read_json, verify_files


def table_pins(dataset: Path) -> dict:
    return {
        str((dataset / SPLIT_FILES[role]).resolve()): sha256(
            dataset / SPLIT_FILES[role]
        )
        for role in ("train", "val")
    }


def selected_indices(concepts: torch.Tensor, limit: int, seed: int) -> torch.Tensor:
    if not limit:
        return torch.arange(len(concepts))
    if limit > len(concepts):
        raise ValueError("Development limit exceeds available rows")
    generator = torch.Generator().manual_seed(seed)
    encoded = codes(concepts)
    groups = []
    for code in range(32):
        rows = torch.where(encoded == code)[0]
        groups.append(rows[torch.randperm(len(rows), generator=generator)].tolist())
    selected, position = [], 0
    while len(selected) < limit:
        for group in groups:
            if position < len(group) and len(selected) < limit:
                selected.append(group[position])
        position += 1
    return torch.tensor(sorted(selected))


def pooled(paths: list[str], tick) -> tuple[np.ndarray, dict]:
    chunks, hashes = [], {}
    for start in range(0, len(paths), 512):
        images = []
        for name in paths[start : start + 512]:
            raw = Path(name).read_bytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != (32, 32):
                    raise ValueError(f"Expected Robot 32x32 images: {name}")
                images.append(np.array(image.convert("L"), dtype=np.uint8))
        pixels = torch.from_numpy(np.stack(images)).float()
        foreground = center_gray(1 - pixels / 255)
        chunks.append(
            F.adaptive_avg_pool2d(foreground[:, None], (10, 4)).flatten(1).numpy()
        )
        tick(
            "preprocessing",
            offset=min(start + 512, len(paths)),
            stage_samples=len(paths),
        )
    return np.concatenate(chunks), hashes


def prepare(config: Config, output: Path, tick) -> tuple[dict, dict]:
    groups, _roles, audit = load_tables(Path(config.dataset), smoke=False)
    if not config.development and (
        len(groups["train"]["labels"]),
        len(groups["validation"]["labels"]),
    ) != (18432, 6144):
        raise ValueError("Expected complete Robot v5 training/validation splits")
    chosen = {
        role: selected_indices(group["concepts"], limit, config.seed)
        for (role, group), limit in zip(
            groups.items(), (config.train_limit, config.val_limit), strict=True
        )
    }
    lock_path = output / "data_lock.json"
    preprocessing: dict
    if lock_path.exists():
        lock = read_json(lock_path)
        if lock["config"] != config.to_dict():
            raise ValueError("Cached data configuration changed")
        verify_files(output, lock["artifacts"])
        pins = read_json(output / "image_sha256.json")
        for i, (name, digest) in enumerate(pins.items()):
            verify_files(Path("/"), {name: digest})
            if i % 512 == 0:
                tick("verifying_images", offset=i, stage_samples=len(pins))
        data = torch.load(output / "data.pt", map_location="cpu", weights_only=True)
        preprocessing = read_json(output / "preprocessing.json")
    else:
        train_features, pins = pooled(groups["train"]["paths"], tick)
        permutation, variances = fit_routing(train_features)
        val_paths = [
            groups["validation"]["paths"][i] for i in chosen["validation"].tolist()
        ]
        val_features, val_pins = pooled(val_paths, tick)
        pins.update(val_pins)
        features = {
            "train": train_features[chosen["train"].numpy()],
            "validation": val_features,
        }
        data = {}
        for role, group in groups.items():
            index = chosen[role]
            data[role] = {
                "concepts": group["concepts"][index].long(),
                "labels": group["labels"][index].float(),
                "source_index": group["indices"][index],
                "robot_ids": group["robot_ids"][index],
                "angles": torch.from_numpy(
                    (features[role][:, permutation].reshape(-1, 10, 4) * np.pi).astype(
                        np.float32
                    )
                ),
            }
        preprocessing = {
            "dataset": str(Path(config.dataset).resolve()),
            "data_version": "Robot v5",
            "concept_order": audit["concept_order"],
            "pool": [10, 4],
            "normalization": "1 - gray/255; preserve antialiasing",
            "centering": "nonwhite foreground bbox; integer translation without crop",
            "routing": (
                "training variance descending; stable ties; round-robin over 10 wires"
            ),
            "permutation": permutation.tolist(),
            "variances": variances.tolist(),
            "routing_fit_role": "full train only",
            "routing_fit_rows": len(train_features),
            "used_rows": {role: len(values["labels"]) for role, values in data.items()},
            "angle_scale": float(np.pi),
            "scale_applied_once": True,
            "source_audit": audit,
            "engineering_subset": config.development,
            "test_read": False,
            "test_evaluated": False,
            "selected_indices": {
                role: index.tolist() for role, index in chosen.items()
            },
            "data_hashes": {
                role: {k: array_hash(v.numpy()) for k, v in values.items()}
                for role, values in data.items()
            },
        }
        atomic_checkpoint(output / "data.pt", data)
        atomic_json(output / "preprocessing.json", preprocessing)
        atomic_json(output / "image_sha256.json", pins)
        atomic_json(
            lock_path,
            {
                "config": config.to_dict(),
                "artifacts": {
                    name: sha256(output / name)
                    for name in ("data.pt", "preprocessing.json", "image_sha256.json")
                },
            },
        )
    for role, group in groups.items():
        for target, source in (
            ("concepts", "concepts"),
            ("labels", "labels"),
            ("source_index", "indices"),
            ("robot_ids", "robot_ids"),
        ):
            if not torch.equal(data[role][target], group[source][chosen[role]]):
                raise ValueError("Prepared targets differ from original observed rows")
        for key, tensor in data[role].items():
            if array_hash(tensor.numpy()) != preprocessing["data_hashes"][role][key]:
                raise ValueError("Prepared tensor content changed")
    return {
        role: {k: v.cuda() for k, v in values.items()} for role, values in data.items()
    }, preprocessing
