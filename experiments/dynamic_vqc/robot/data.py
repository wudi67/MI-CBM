"""Original Robot identity splits; grayscale pool features fit on train only."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from robot_dataset_schema import ROBOT_CONCEPT_NAMES, SPLIT_FILES

from ..circuits import bit_indices, bit_table
from ..runtime import atomic_checkpoint, atomic_json
from .common import fingerprint, lock_files, read_json, verify_pins
from .config import Config

CONCEPTS = tuple(ROBOT_CONCEPT_NAMES)


def label_counts(concepts: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Retain both observed labels and their multiplicities for every concept."""
    pairs = 2 * bit_indices(concepts) + labels.long()
    return torch.bincount(pairs, minlength=64).reshape(32, 2)


def load_tables(dataset: Path, smoke: bool = False) -> tuple[dict, dict, dict]:
    groups, roles, audit = {}, {}, {}
    seen_ids: set[int] = set()
    seen_paths: set[str] = set()
    for role, split in (("train", "train"), ("validation", "val")):
        with (dataset / SPLIT_FILES[split]).open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"Empty {role} CSV")
        concepts = torch.tensor([[int(row[key]) for key in CONCEPTS] for row in rows])
        labels = torch.tensor([int(row["label"]) for row in rows])
        identities = torch.tensor([int(row["robot_id"]) for row in rows])
        renders = [int(row["render_id"]) for row in rows]
        paths = [
            str(dataset / "robot_images" / Path(row["image_path"]).name) for row in rows
        ]
        if not bool(((concepts == 0) | (concepts == 1)).all()) or not bool(
            ((labels == 0) | (labels == 1)).all()
        ):
            raise ValueError("Robot concepts and observed labels must be binary")
        if not bool(((identities >= 0) & (identities < 7680)).all()):
            raise ValueError("Invalid Robot v5 identity")
        by_id = defaultdict(list)
        for index, identity in enumerate(identities.tolist()):
            by_id[identity].append(index)
        for indices in by_id.values():
            if sorted(renders[index] for index in indices) != [0, 1, 2, 3]:
                raise ValueError("Each identity must retain all four renders")
            if not torch.equal(concepts[indices], concepts[indices[0]].expand(4, -1)):
                raise ValueError("Concept annotations differ within one identity")
        if seen_ids.intersection(by_id) or seen_paths.intersection(paths):
            raise ValueError("Train and validation identities or image paths overlap")
        if len(set(paths)) != len(rows):
            raise ValueError("Repeated image path within a split")
        seen_ids.update(by_id)
        seen_paths.update(paths)
        counts = label_counts(concepts, labels)
        if bool((counts.sum(1) == 0).any()):
            raise ValueError("Each development split must cover all 32 concepts")
        roles[role] = {
            "csv_rows": list(range(len(rows))),
            "robot_ids": identities.tolist(),
            "render_ids": renders,
            "image_paths": paths,
        }
        audit[role] = {
            "rows": len(rows),
            "identities": len(by_id),
            "concept_label_counts": counts.tolist(),
            "combinations_with_both_observed_labels": int(
                (counts.min(1).values > 0).sum()
            ),
        }
        # Smoke subsetting is explicitly separate from the full-data protocol.
        selected = torch.arange(len(rows))
        if smoke:
            selected = (
                torch.randperm(len(rows), generator=torch.Generator().manual_seed(71))[
                    :128
                ]
                .sort()
                .values
            )
        groups[role] = {
            "concepts": concepts[selected].float(),
            "labels": labels[selected].float(),
            "indices": selected,
            "robot_ids": identities[selected],
            "paths": [paths[index] for index in selected.tolist()],
        }
    audit.update(
        concept_order=list(CONCEPTS),
        train_validation_identity_overlap=0,
        all_training_samples_preserved=not smoke,
        excluded_training_rows=0 if not smoke else None,
        pooled_input_filtering=False,
        pooled_input_statistics_computed=False,
        observed_labels_replaced=False,
        test_read=False,
        smoke_only=smoke,
    )
    return groups, roles, audit


def oracle_data(groups: dict, device: torch.device) -> dict:
    return {
        role: {
            "counts": label_counts(group["concepts"], group["labels"]).to(device),
            "basis": bit_table(5, device).float(),
        }
        for role, group in groups.items()
    }


def center_gray(images: torch.Tensor) -> torch.Tensor:
    """Translate nonwhite foreground without binarizing antialiased grayscale."""
    foreground = images > 0
    batch, height, width = images.shape
    occupied_y, occupied_x = foreground.any(2), foreground.any(1)
    if not bool(occupied_y.any(1).all()):
        raise ValueError("Empty Robot foreground")
    low_y, low_x = occupied_y.int().argmax(1), occupied_x.int().argmax(1)
    high_y = height - 1 - occupied_y.flip(1).int().argmax(1)
    high_x = width - 1 - occupied_x.flip(1).int().argmax(1)
    dy = (height - 1) // 2 - (low_y + high_y) // 2
    dx = (width - 1) // 2 - (low_x + high_x) // 2
    y = torch.arange(height, device=images.device)[None] - dy[:, None]
    x = torch.arange(width, device=images.device)[None] - dx[:, None]
    valid = ((y >= 0) & (y < height))[:, :, None] & ((x >= 0) & (x < width))[:, None]
    result = (
        images[
            torch.arange(batch, device=images.device)[:, None, None],
            y.clamp(0, height - 1)[:, :, None],
            x.clamp(0, width - 1)[:, None],
        ]
        * valid
    )
    if not torch.allclose(result.sum((1, 2)), images.sum((1, 2)), atol=1e-4, rtol=1e-6):
        raise ValueError("Centering cropped Robot foreground")
    return result


def variance_permutation(features: torch.Tensor) -> tuple[list[int], list[float]]:
    variance = features.double().flatten(1).var(0, correction=0).tolist()
    ranked = sorted(range(20), key=lambda index: (-variance[index], index))
    return [index for wire in range(5) for index in ranked[wire::5]], variance


def image_features(groups: dict, output: Path, device: torch.device, update) -> dict:
    """Hash every used image; decode only when the prepared cache is absent."""
    prepared_path = output / "prepared.pt"
    cache_ready = (output / "data_lock.json").exists()
    if cache_ready:
        verify_pins(output, read_json(output / "data_lock.json"))
    total = sum(len(group["paths"]) for group in groups.values())
    done, features, image_pins = 0, {}, {}
    for role, group in groups.items():
        arrays = []
        for path_string in group["paths"]:
            path = Path(path_string)
            raw = path.read_bytes()
            image_pins[path_string] = hashlib.sha256(raw).hexdigest()
            if not cache_ready:
                with Image.open(io.BytesIO(raw)) as image:
                    if image.size != (32, 32):
                        raise ValueError(f"Unexpected Robot image dimensions: {path}")
                    arrays.append(np.array(image.convert("L"), dtype=np.uint8))
            done += 1
            if done % 512 == 0 or done == total:
                update({"step": done, "steps": total})
        if not cache_ready:
            pixels = torch.from_numpy(np.stack(arrays)).to(device).float()
            foreground = 1 - pixels / 255
            features[role] = F.adaptive_avg_pool2d(
                center_gray(foreground)[:, None], (5, 4)
            )[:, 0].cpu()
    if cache_ready:
        if image_pins != read_json(output / "image_sha256.json"):
            raise ValueError("Robot image contents changed after preprocessing")
        prepared = torch.load(prepared_path, map_location="cpu", weights_only=True)
    else:
        permutation, variance = variance_permutation(features["train"])
        prepared = {}
        for role, group in groups.items():
            prepared[role] = {
                key: value for key, value in group.items() if key != "paths"
            }
            prepared[role]["angles"] = (
                features[role].flatten(1)[:, permutation].reshape(-1, 5, 4) * torch.pi
            ).contiguous()
        atomic_checkpoint(prepared_path, prepared)
        atomic_json(output / "image_sha256.json", image_pins)
        atomic_json(
            output / "feature_transform.json",
            {
                "normalization": "1 - gray_uint8 / 255; antialias values retained",
                "centering": "bbox foreground > 0; zero-filled integer translation",
                "pool": [5, 4],
                "pool_function": "torch.nn.functional.adaptive_avg_pool2d",
                "layout": "train_variance_balanced",
                "permutation": permutation,
                "train_variance": variance,
                "angle_scale": float(torch.pi),
                "layout_fit_indices": groups["train"]["indices"].tolist(),
                "validation_or_test_used_for_fit": False,
                "prepared_sha256": fingerprint(prepared),
            },
        )
        lock_files(
            output,
            ["prepared.pt", "image_sha256.json", "feature_transform.json"],
            "data_lock.json",
        )
    for role, group in groups.items():
        for key in ("indices", "concepts", "labels", "robot_ids"):
            if not torch.equal(prepared[role][key], group[key]):
                raise ValueError("Cached preprocessing changed the original data roles")
    return {
        role: {key: value.to(device) for key, value in group.items()}
        for role, group in prepared.items()
    }


def prepare_images(
    config: Config, groups: dict, output: Path, device, progress
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    total = sum(len(group["paths"]) for group in groups.values())
    progress.start_cell("Prepare shared Robot image features", total)
    data = image_features(groups, output, device, progress.update)
    if not config.smoke and len(data["train"]["angles"]) != len(
        groups["train"]["paths"]
    ):
        raise ValueError("Training rows were lost")
    return data
