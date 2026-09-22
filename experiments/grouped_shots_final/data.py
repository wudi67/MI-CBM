"""Use frozen training preprocessing; open test values only after the stage seal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiments.grouped_dynamic_vqc.data import (
    pooled_features,
    stratified_indices,
    validate_targets,
)
from experiments.grouped_dynamic_vqc.runtime import array_hash, sha256
from experiments.grouped_feedback_ablation.protocol import read_json


def subset(data: dict, limit: int, seed: int = 0) -> dict:
    if limit > len(data["labels"]):
        raise ValueError("Development subset exceeds validation data")
    indices = stratified_indices(data["concepts"].cpu().numpy(), limit, seed)
    return {k: v[indices].cpu() for k, v in data.items()}


def data_hashes(data: dict) -> dict:
    return {k: array_hash(v.cpu().numpy()) for k, v in data.items()}


def read_test(audit: dict, protocol: Path, access: Path, manifest_hash: str) -> dict:
    """No fitting and no validation fallback in the formal test reader."""
    gate, opened = read_json(protocol), read_json(access)
    if (
        gate["manifest_sha256"] != manifest_hash
        or gate["development"]
        or opened["protocol_sha256"] != sha256(protocol)
        or opened["role"] != "test"
    ):
        raise ValueError(
            "A verified formal validation seal is required before test access"
        )
    dataset = Path(audit["dataset"])
    if sha256(dataset) != audit["dataset_sha256"]:
        raise ValueError("Dataset changed before final test")
    if audit["routing_fit_role"] != "full original train images only":
        raise ValueError("Expected training-only variance routing")
    permutation = np.asarray(audit["permutation"], dtype=np.int64)
    if sorted(permutation.tolist()) != list(range(40)):
        raise ValueError("Invalid frozen feature permutation")
    with np.load(dataset, allow_pickle=False) as archive:
        split = np.asarray(archive["split"])
        indices = np.flatnonzero(split == "test")
        if len(indices) != audit["test_row_count_metadata"]:
            raise ValueError("Test row count differs from original split")
        raster = np.asarray(archive["raster_group_id"])
        if np.intersect1d(raster[indices], raster[split != "test"]).size:
            raise ValueError("Test raster identities overlap development data")
        source = np.asarray(archive["source_index"])
        if len(np.unique(source)) != len(source):
            raise ValueError("Duplicate source indices")
        images = np.asarray(archive["imgs"])[indices]
        concepts = np.asarray(archive["c_int"])[indices]
        labels = np.asarray(archive["y_task"])[indices]
    validate_targets(concepts, labels)
    features = pooled_features(images)
    angles = (
        features[:, permutation].reshape(-1, 10, 4) * audit["angle_scale"]
    ).astype(np.float32)
    return {
        "angles": torch.from_numpy(angles),
        "concepts": torch.from_numpy(concepts.copy()),
        "labels": torch.from_numpy(labels.copy()).float(),
        "source_index": torch.from_numpy(source[indices].copy()),
    }
