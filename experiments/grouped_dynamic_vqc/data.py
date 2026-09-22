"""Image-only centering, train-fitted variance routing and preserved dSprites splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .runtime import array_hash, atomic_checkpoint, atomic_json, sha256


def center_images(images: torch.Tensor) -> torch.Tensor:
    """Translate foreground bboxes to the center without resize, wrap or cropping."""
    if images.ndim != 3 or not torch.all((images == 0) | (images == 1)):
        raise ValueError("Expected binary images [N,H,W] with values 0/1")
    foreground = images.bool()
    if not foreground.flatten(1).any(1).all():
        raise ValueError("Empty foreground is unsupported")
    height, width = images.shape[1:]
    occupied_rows = foreground.any(2).int()
    occupied_columns = foreground.any(1).int()
    top = occupied_rows.argmax(1)
    bottom = height - 1 - occupied_rows.flip(1).argmax(1)
    left = occupied_columns.argmax(1)
    right = width - 1 - occupied_columns.flip(1).argmax(1)
    dy = (height - 1) // 2 - (top + bottom) // 2
    dx = (width - 1) // 2 - (left + right) // 2
    source_y = torch.arange(height, device=images.device)[None, :] - dy[:, None]
    source_x = torch.arange(width, device=images.device)[None, :] - dx[:, None]
    valid = ((source_y >= 0) & (source_y < height))[:, :, None] & (
        (source_x >= 0) & (source_x < width)
    )[:, None, :]
    rows = images.gather(1, source_y.clamp(0, height - 1)[:, :, None].expand_as(images))
    result = (
        rows.gather(2, source_x.clamp(0, width - 1)[:, None, :].expand_as(images))
        * valid
    )
    if not torch.equal(result.sum((1, 2)), images.sum((1, 2))):
        raise ValueError("Centering would crop foreground")
    return result


def pooled_features(images: np.ndarray) -> np.ndarray:
    chunks = []
    for start in range(0, len(images), 2048):
        centered = center_images(torch.from_numpy(images[start : start + 2048]))
        chunks.append(F.adaptive_avg_pool2d(centered[:, None].float(), (10, 4)))
    return torch.cat(chunks).reshape(-1, 40).numpy()


def fit_routing(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train_features.ndim != 2 or train_features.shape[1] != 40:
        raise ValueError("Expected forty pooled features")
    variances = train_features.astype(np.float64).var(axis=0)
    ranking = sorted(range(40), key=lambda index: (-variances[index], index))
    permutation = np.array([index for wire in range(10) for index in ranking[wire::10]])
    return permutation, variances


def stratified_indices(concepts: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Optional development subset, balanced over all 18 concept combinations."""
    if limit == 0 or limit >= len(concepts):
        return np.arange(len(concepts))
    if limit < 18:
        raise ValueError("Development subsets must include at least 18 rows")
    generator = np.random.default_rng(seed)
    codes = concepts[:, 0] * 6 + concepts[:, 1]
    groups = [
        generator.permutation(np.flatnonzero(codes == code)) for code in range(18)
    ]
    selected: list[int] = []
    position = 0
    while len(selected) < limit:
        for group in groups:
            if position < len(group) and len(selected) < limit:
                selected.append(int(group[position]))
        position += 1
    return np.sort(np.array(selected))


def validate_targets(concepts: np.ndarray, labels: np.ndarray) -> None:
    if concepts.shape != (len(labels), 2):
        raise ValueError("Expected shape/scale integer columns")
    if not (
        np.issubdtype(concepts.dtype, np.integer)
        and np.all((concepts[:, 0] >= 0) & (concepts[:, 0] < 3))
        and np.all((concepts[:, 1] >= 0) & (concepts[:, 1] < 6))
    ):
        raise ValueError("Invalid shape/scale codes")
    expected = ((concepts[:, 0] == 2) ^ (concepts[:, 1] > 2)).astype(np.int64)
    if not np.array_equal(labels, expected):
        raise ValueError("Expected compact_c label = (shape == 2) XOR (scale > 2)")


def prepare_data(
    dataset: Path,
    admission: Path,
    output: Path,
    *,
    train_limit: int = 0,
    val_limit: int = 0,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Read existing admitted dataset; never optimize/evaluate on test rows."""
    manifest_path = dataset.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = json.loads(admission.read_text(encoding="utf-8"))
    dataset_hash = sha256(dataset)
    if (
        not gate.get("admitted_for_confirmatory_sweep")
        or manifest.get("variant") != "compact_c"
        or manifest.get("npz_sha256") != dataset_hash
        or gate["variants"]["compact_c"]["array_content_fingerprint"]
        != manifest["array_content_fingerprint"]
    ):
        raise ValueError("Dataset hash/manifest/admission mismatch")
    with np.load(dataset, allow_pickle=False) as archive:
        split = np.asarray(archive["split"])
        source = np.asarray(archive["source_index"])
        raster = np.asarray(archive["raster_group_id"])
        if set(np.unique(split)) != {"train", "val", "test"}:
            raise ValueError("Expected preserved train/val/test split")
        if len(np.unique(source)) != len(source):
            raise ValueError("Duplicate source indices")
        roles = {
            role: np.flatnonzero(split == role) for role in ("train", "val", "test")
        }
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            if np.intersect1d(raster[roles[first]], raster[roles[second]]).size:
                raise ValueError("Raster group crosses dataset splits")
        # NPZ members are materialized as arrays; slice train/val before image
        # preprocessing. Test rows are not transformed, fitted, or evaluated.
        images = np.asarray(archive["imgs"])
        concepts = np.asarray(archive["c_int"])
        labels = np.asarray(archive["y_task"])
    features = {role: pooled_features(images[roles[role]]) for role in ("train", "val")}
    permutation, variances = fit_routing(features["train"])
    data = {}
    role_audits = {}
    for role, limit in (("train", train_limit), ("val", val_limit)):
        indices = roles[role]
        validate_targets(concepts[indices], labels[indices])
        selected = stratified_indices(concepts[indices], limit, seed)
        selected_rows = indices[selected]
        angles = (
            features[role][selected][:, permutation].reshape(-1, 10, 4) * np.pi
        ).astype(np.float32)
        data[role] = {
            "angles": torch.from_numpy(angles),
            "concepts": torch.from_numpy(concepts[selected_rows].copy()),
            "labels": torch.from_numpy(labels[selected_rows].copy()).float(),
            "source_index": torch.from_numpy(source[selected_rows].copy()),
        }
        role_audits[role] = {
            "available_rows": len(indices),
            "used_rows": len(selected_rows),
            "selected_npz_rows": selected_rows.tolist(),
            "source_indices_sha256": array_hash(source[selected_rows]),
            "angles_sha256": array_hash(angles),
        }
    audit = {
        "dataset": str(dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "dataset_manifest_sha256": sha256(manifest_path),
        "admission": str(admission.resolve()),
        "admission_sha256": sha256(admission),
        "preprocessing": "binary_0_1 -> bbox integer center -> avg_pool(10,4)",
        "routing": "stable variance descending; round-robin over ten wires",
        "routing_fit_role": "full original train images only",
        "routing_fit_rows": len(roles["train"]),
        "routing_fit_source_indices_sha256": array_hash(source[roles["train"]]),
        "permutation": permutation.tolist(),
        "variances": variances.tolist(),
        "angle_scale": float(np.pi),
        "scale_applied_once": True,
        "roles": role_audits,
        "test_evaluated": False,
        "test_preprocessed": False,
        "test_row_count_metadata": len(roles["test"]),
        "cross_split_raster_group_overlap": 0,
    }
    atomic_checkpoint(output / "data.pt", data)
    audit["cache_sha256"] = sha256(output / "data.pt")
    atomic_json(output / "preprocessing.json", audit)
    return data, audit


def load_cached_data(output: Path) -> tuple[dict, dict]:
    audit = json.loads((output / "preprocessing.json").read_text(encoding="utf-8"))
    if sha256(output / "data.pt") != audit["cache_sha256"]:
        raise ValueError("Cached inputs changed since preprocessing")
    if sha256(Path(audit["dataset"])) != audit["dataset_sha256"]:
        raise ValueError("Source dataset changed since preprocessing")
    return torch.load(output / "data.pt", weights_only=True, map_location="cpu"), audit
