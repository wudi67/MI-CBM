"""Frozen Robot preprocessing and a test reader behind the validation seal."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.dynamic_vqc.robot.data import load_tables
from experiments.grouped_dynamic_vqc.runtime import array_hash, sha256
from experiments.grouped_robot_pilot.data import pooled, selected_indices
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from robot_dataset_schema import ROBOT_CONCEPT_NAMES, SPLIT_FILES


def data_hashes(data: dict) -> dict:
    return {k: array_hash(v.cpu().numpy()) for k, v in data.items()}


def subset(data: dict, limit: int, seed: int = 0) -> dict:
    index = selected_indices(data["concepts"].cpu(), limit, seed)
    return {k: v.cpu()[index] for k, v in data.items()}


def verify_test_gate(output: Path, manifest_hash: str) -> None:
    gate = read_json(output / "protocol_lock.json")
    access = read_json(output / "test_access.json")
    validation = output / "validation"
    if (
        gate["manifest_sha256"] != manifest_hash
        or gate["development"]
        or access["protocol_sha256"] != sha256(output / "protocol_lock.json")
        or access["role"] != "test"
        or not access["real_test_values_requested"]
        or gate["validation_result_lock_sha256"]
        != sha256(validation / "result_lock.json")
    ):
        raise ValueError("Verified formal validation seal required before test access")
    lock = read_json(validation / "result_lock.json")
    if lock["manifest_sha256"] != manifest_hash:
        raise ValueError("Validation result identity changed")
    verify_files(validation, lock["artifacts"])
    if read_json(validation / "summary.json")["status"] != "complete":
        raise ValueError("Validation must be complete before test access")


def read_test(
    sources, output: Path, manifest_hash: str, tick, *, expected_rows: int = 6144
) -> tuple[dict, dict]:
    """Open test only here. Reuse centering/pooling and train-fitted permutation."""
    verify_test_gate(output, manifest_hash)
    audit = sources.audit
    if (
        audit["routing_fit_role"] != "full train only"
        or audit["pool"] != [10, 4]
        or audit["concept_order"] != list(ROBOT_CONCEPT_NAMES)
        or not audit["scale_applied_once"]
        or abs(audit["angle_scale"] - np.pi) > 1e-12
    ):
        raise ValueError("Unexpected frozen Robot preprocessing")
    permutation = np.asarray(audit["permutation"], dtype=np.int64)
    if sorted(permutation.tolist()) != list(range(40)):
        raise ValueError("Invalid frozen forty-feature routing")
    dataset = sources.dataset.resolve()
    groups, roles, _ = load_tables(dataset, smoke=False)
    path = dataset / SPLIT_FILES["test"]
    table_hash = sha256(path)
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != expected_rows:
        raise ValueError("Unexpected test row count; no subsetting or fallback allowed")
    concepts = torch.tensor([[int(r[k]) for k in ROBOT_CONCEPT_NAMES] for r in rows])
    labels = torch.tensor([int(r["label"]) for r in rows])
    ids = torch.tensor([int(r["robot_id"]) for r in rows])
    renders = [int(r["render_id"]) for r in rows]
    paths = [
        str((dataset / "robot_images" / Path(r["image_path"]).name).resolve())
        for r in rows
    ]
    if (
        not bool(((concepts == 0) | (concepts == 1)).all())
        or not bool(((labels == 0) | (labels == 1)).all())
        or not bool(((ids >= 0) & (ids < 7680)).all())
    ):
        raise ValueError("Invalid Robot test concepts/labels/identities")
    by_id = defaultdict(list)
    for index, identity in enumerate(ids.tolist()):
        by_id[identity].append(index)
    for indices in by_id.values():
        if sorted(renders[i] for i in indices) != [0, 1, 2, 3] or not torch.equal(
            concepts[indices], concepts[indices[0]].expand(4, -1)
        ):
            raise ValueError(
                "Test identities must preserve four renders and their concepts"
            )
    development_ids = set().union(*(set(v["robot_ids"]) for v in roles.values()))
    development_paths = set().union(
        *({str(Path(p).resolve()) for p in v["image_paths"]} for v in roles.values())
    )
    if development_ids.intersection(by_id) or development_paths.intersection(paths):
        raise ValueError("Test identities/image paths overlap development data")
    if len(set(paths)) != len(rows):
        raise ValueError("Duplicate test image path")
    if expected_rows == 6144 and (
        (len(groups["train"]["labels"]), len(groups["validation"]["labels"]))
        != (18432, 6144)
        or development_ids.union(by_id) != set(range(7680))
    ):
        raise ValueError("Robot v5 full identity/split coverage changed")
    features, image_hashes = pooled(paths, tick)
    if sha256(path) != table_hash:
        raise ValueError("Test table changed during preprocessing")
    data = {
        "angles": torch.from_numpy(
            (features[:, permutation].reshape(-1, 10, 4) * audit["angle_scale"]).astype(
                np.float32
            )
        ),
        "concepts": concepts.long(),
        "labels": labels.float(),
        "source_index": torch.arange(len(rows)),
        "robot_ids": ids,
    }
    inputs = {
        "dataset": str(dataset),
        "rows": len(rows),
        "identities": len(by_id),
        "test_development_identity_overlap": 0,
        "test_development_path_overlap": 0,
        "preprocessing_sha256": sha256(sources.preprocessing_path),
        "permutation": permutation.tolist(),
        "routing_refitted": False,
        "observed_labels_replaced": False,
        "artifacts": {str(path): table_hash, **image_hashes},
    }
    return data, inputs
