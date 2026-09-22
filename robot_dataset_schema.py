"""Shared schema and label rule for the current Robot dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

ROBOT_DATASET_VERSION = "v5"
ROBOT_NUM_IDENTITIES = 7_680
ROBOT_RENDERS_PER_IDENTITY = 4
ROBOT_ID_COLUMN = "robot_id"
ROBOT_RENDER_ID_COLUMN = "render_id"
ROBOT_CONCEPT_NAMES = (
    "head_shape",
    "body_shape",
    "has_antennae",
    "ears_shape",
    "foot_shape",
)
CLASS_NAMES = {0: "drent", 1: "glorp"}
SPLIT_FILES = {
    "train": "robot_images_train_labels.csv",
    "val": "robot_images_val_labels.csv",
    "test": "robot_images_test_labels.csv",
}

ROBOT_LABEL_WEIGHTS = {
    "body_shape": 5.0,
    "foot_shape": 4.0,
    "has_antennae": 3.0,
    "head_shape": 2.0,
    "ears_shape": 1.0,
}
ROBOT_LABEL_INTERCEPT = -7.5
ROBOT_LABEL_TEMPERATURE = 8.4
EXTERNAL_RENDER_PROTOCOL_VERSION = "external_rerender_v1"


def resolve_robot_image_path(data_root: str | Path, csv_image_path: str) -> Path:
    """Resolve a Robot image path, including legacy CSV path prefixes."""

    root = Path(data_root).expanduser().resolve()
    raw = Path(csv_image_path)
    candidates = (raw, root / raw, root / "robot_images" / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve image path {csv_image_path!r} under {root}"
    )


def audit_robot_dataset(data_root: str | Path) -> dict[str, object]:
    """Validate the v5 identity-grouped splits before an experiment starts."""

    root = Path(data_root).expanduser().resolve()
    required_columns = (
        "image_path",
        ROBOT_ID_COLUMN,
        ROBOT_RENDER_ID_COLUMN,
        *ROBOT_CONCEPT_NAMES,
        "label",
        "class",
    )
    split_rows: dict[str, int] = {}
    split_robot_ids: dict[str, set[int]] = {}
    render_ids_by_robot: dict[int, set[int]] = defaultdict(set)
    row_count_by_robot: Counter[int] = Counter()
    concepts_by_robot: dict[int, tuple[int, ...]] = {}
    labels_by_robot: dict[int, set[int]] = defaultdict(set)
    image_paths: set[Path] = set()
    deterministic_label_mismatches = 0

    for split, filename in SPLIT_FILES.items():
        csv_path = root / filename
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing Robot split CSV: {csv_path}")

        robot_ids: set[int] = set()
        row_count = 0
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [
                column
                for column in required_columns
                if column not in (reader.fieldnames or [])
            ]
            if missing:
                raise ValueError(
                    f"{csv_path} is not a {ROBOT_DATASET_VERSION} Robot split; "
                    f"missing required columns: {missing}"
                )

            for line_number, row in enumerate(reader, start=2):
                try:
                    robot_id = int(row[ROBOT_ID_COLUMN])
                    render_id = int(row[ROBOT_RENDER_ID_COLUMN])
                    concept = tuple(int(row[name]) for name in ROBOT_CONCEPT_NAMES)
                    label = int(row["label"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric value at {csv_path}:{line_number}"
                    ) from exc

                if not 0 <= robot_id < ROBOT_NUM_IDENTITIES:
                    raise ValueError(
                        f"robot_id out of range at {csv_path}:{line_number}: {robot_id}"
                    )
                if not 0 <= render_id < ROBOT_RENDERS_PER_IDENTITY:
                    raise ValueError(
                        f"render_id out of range at {csv_path}:{line_number}: {render_id}"
                    )
                if any(value not in (0, 1) for value in concept):
                    raise ValueError(
                        f"Non-binary concept at {csv_path}:{line_number}: {concept}"
                    )
                if label not in CLASS_NAMES:
                    raise ValueError(
                        f"Unexpected label at {csv_path}:{line_number}: {label}"
                    )
                expected_class = CLASS_NAMES[label]
                if row["class"] != expected_class:
                    raise ValueError(
                        f"Class/label mismatch at {csv_path}:{line_number}: "
                        f"{row['class']!r} != {expected_class!r}"
                    )

                previous_concept = concepts_by_robot.setdefault(robot_id, concept)
                if concept != previous_concept:
                    raise ValueError(
                        f"Concepts change across renders of robot_id={robot_id}"
                    )
                image_path = resolve_robot_image_path(root, row["image_path"])
                if image_path in image_paths:
                    raise ValueError(
                        f"Image is referenced more than once: {image_path}"
                    )

                image_paths.add(image_path)
                robot_ids.add(robot_id)
                render_ids_by_robot[robot_id].add(render_id)
                row_count_by_robot[robot_id] += 1
                labels_by_robot[robot_id].add(label)
                score = ROBOT_LABEL_INTERCEPT + sum(
                    ROBOT_LABEL_WEIGHTS[name] * concept[index]
                    for index, name in enumerate(ROBOT_CONCEPT_NAMES)
                )
                deterministic_label_mismatches += int(label != int(score > 0.0))
                row_count += 1

        split_rows[split] = row_count
        split_robot_ids[split] = robot_ids

    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    intersections = {
        f"{left}_{right}": len(split_robot_ids[left] & split_robot_ids[right])
        for left, right in split_pairs
    }
    leaking_pairs = [name for name, count in intersections.items() if count]
    if leaking_pairs:
        raise ValueError(
            "Robot identity leakage across splits: "
            + ", ".join(f"{name}={intersections[name]}" for name in leaking_pairs)
        )

    all_robot_ids = set().union(*split_robot_ids.values())
    expected_robot_ids = set(range(ROBOT_NUM_IDENTITIES))
    if all_robot_ids != expected_robot_ids:
        missing = len(expected_robot_ids - all_robot_ids)
        unexpected = len(all_robot_ids - expected_robot_ids)
        raise ValueError(
            "Robot identity coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    expected_render_ids = set(range(ROBOT_RENDERS_PER_IDENTITY))
    invalid_robots = [
        robot_id
        for robot_id in sorted(all_robot_ids)
        if row_count_by_robot[robot_id] != ROBOT_RENDERS_PER_IDENTITY
        or render_ids_by_robot[robot_id] != expected_render_ids
    ]
    if invalid_robots:
        examples = invalid_robots[:10]
        raise ValueError(
            "Each Robot v5 identity must have exactly one render_id in 0..3; "
            f"invalid robot_ids include {examples}"
        )

    expected_rows = ROBOT_NUM_IDENTITIES * ROBOT_RENDERS_PER_IDENTITY
    total_rows = sum(split_rows.values())
    if total_rows != expected_rows or len(image_paths) != expected_rows:
        raise ValueError(
            f"Robot row/image count mismatch: expected={expected_rows}, "
            f"rows={total_rows}, unique_images={len(image_paths)}"
        )

    image_dir_paths = {path.resolve() for path in (root / "robot_images").glob("*.png")}
    if image_paths != image_dir_paths:
        raise ValueError(
            "Robot CSV/image directory mismatch: "
            f"missing_from_csv={len(image_dir_paths - image_paths)}, "
            f"outside_or_missing_from_directory={len(image_paths - image_dir_paths)}"
        )

    deterministic_label_mismatch_rate = deterministic_label_mismatches / total_rows
    if deterministic_label_mismatch_rate > 0.02:
        raise ValueError(
            "Robot labels do not follow the documented stochastic rule closely "
            f"enough: deterministic mismatch rate={deterministic_label_mismatch_rate:.4f}"
        )

    return {
        "dataset_version": ROBOT_DATASET_VERSION,
        "split_rows": split_rows,
        "split_identities": {
            split: len(robot_ids) for split, robot_ids in split_robot_ids.items()
        },
        "identity_intersections": intersections,
        "total_rows": total_rows,
        "total_identities": len(all_robot_ids),
        "renders_per_identity": ROBOT_RENDERS_PER_IDENTITY,
        "identities_with_label_variation": sum(
            len(labels) > 1 for labels in labels_by_robot.values()
        ),
        "deterministic_label_mismatches": deterministic_label_mismatches,
        "deterministic_label_mismatch_rate": deterministic_label_mismatch_rate,
    }


def robot_label_score(concepts: Mapping[str, int | float]) -> float:
    """Return the documented pre-sigmoid score for one concept vector."""

    return ROBOT_LABEL_INTERCEPT + sum(
        weight * float(concepts[name]) for name, weight in ROBOT_LABEL_WEIGHTS.items()
    )


def robot_dataset_fingerprint(data_root: str | Path) -> str:
    """Hash split metadata and image bytes to identify one exact dataset build."""

    root = Path(data_root).expanduser().resolve()
    digest = hashlib.sha256()
    digest.update(ROBOT_DATASET_VERSION.encode("ascii"))
    for split, filename in SPLIT_FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing Robot split CSV: {path}")
        digest.update(split.encode("ascii"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

    image_dir = root / "robot_images"
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No Robot PNG images found under {image_dir}")
    for path in image_paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def robot_image_content_hashes(data_root: str | Path) -> set[str]:
    """Return byte-level PNG hashes for cross-build overlap audits."""

    root = Path(data_root).expanduser().resolve()
    image_paths = sorted((root / "robot_images").glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No Robot PNG images found under {root / 'robot_images'}")
    hashes: set[str] = set()
    for path in image_paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes.add(digest.hexdigest())
    return hashes


def robot_split_identity_sets(data_root: str | Path) -> dict[str, set[int]]:
    """Read the identity allocation without exposing it in JSON audit output."""

    root = Path(data_root).expanduser().resolve()
    output: dict[str, set[int]] = {}
    for split, filename in SPLIT_FILES.items():
        with (root / filename).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if ROBOT_ID_COLUMN not in (reader.fieldnames or []):
                raise ValueError(f"Missing {ROBOT_ID_COLUMN} in {root / filename}")
            output[split] = {int(row[ROBOT_ID_COLUMN]) for row in reader}
    return output


def robot_split_concept_maps(
    data_root: str | Path,
) -> dict[str, dict[tuple[int, int], tuple[int, ...]]]:
    """Map (identity, render) to its concepts in each split."""

    root = Path(data_root).expanduser().resolve()
    output: dict[str, dict[tuple[int, int], tuple[int, ...]]] = {}
    required = {ROBOT_ID_COLUMN, ROBOT_RENDER_ID_COLUMN, *ROBOT_CONCEPT_NAMES}
    for split, filename in SPLIT_FILES.items():
        rows: dict[tuple[int, int], tuple[int, ...]] = {}
        with (root / filename).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"Missing Robot columns in {root / filename}: {missing}")
            for row in reader:
                key = (int(row[ROBOT_ID_COLUMN]), int(row[ROBOT_RENDER_ID_COLUMN]))
                if key in rows:
                    raise ValueError(f"Duplicate Robot identity/render in {root / filename}")
                rows[key] = tuple(int(row[name]) for name in ROBOT_CONCEPT_NAMES)
        output[split] = rows
    return output


def audit_external_robot_dataset(
    development_root: str | Path,
    external_root: str | Path,
    *,
    require_disjoint_images: bool = True,
) -> dict[str, object]:
    """Audit an independently generated Robot build used only for final testing."""

    development = Path(development_root).expanduser().resolve()
    external = Path(external_root).expanduser().resolve()
    if development == external:
        raise ValueError("External Robot root must differ from the development root")
    development_audit = audit_robot_dataset(development)
    external_audit = audit_robot_dataset(external)
    development_fingerprint = robot_dataset_fingerprint(development)
    external_fingerprint = robot_dataset_fingerprint(external)
    if development_fingerprint == external_fingerprint:
        raise ValueError("External Robot dataset is byte-identical to development data")
    development_hashes = robot_image_content_hashes(development)
    external_hashes = robot_image_content_hashes(external)
    overlap = development_hashes & external_hashes
    if require_disjoint_images and overlap:
        raise ValueError(
            "External Robot dataset contains byte-identical development images: "
            f"overlap={len(overlap)}"
        )
    if development_audit["split_rows"] != external_audit["split_rows"]:
        raise ValueError("External Robot split sizes do not match the v5 protocol")
    development_ids = robot_split_identity_sets(development)
    external_ids = robot_split_identity_sets(external)
    forbidden_identity_overlap = {
        "development_train_external_test": len(
            development_ids["train"] & external_ids["test"]
        ),
        "development_val_external_test": len(
            development_ids["val"] & external_ids["test"]
        ),
    }
    if any(forbidden_identity_overlap.values()):
        raise ValueError(
            "External test identities overlap development train/validation: "
            + ", ".join(
                f"{name}={count}"
                for name, count in forbidden_identity_overlap.items()
                if count
            )
        )
    split_identity_sets_equal = all(
        development_ids[split] == external_ids[split] for split in SPLIT_FILES
    )
    if not split_identity_sets_equal:
        raise ValueError(
            "External build must preserve the development identity allocation; "
            "only nuisance renders and stochastic label draws may change"
        )
    development_concepts = robot_split_concept_maps(development)
    external_concepts = robot_split_concept_maps(external)
    concept_maps_equal = all(
        development_concepts[split] == external_concepts[split]
        for split in SPLIT_FILES
    )
    if not concept_maps_equal:
        raise ValueError("External Robot build changed identity/concept annotations")
    generation_manifest_path = external / "external_generation_manifest.json"
    if not generation_manifest_path.is_file():
        raise FileNotFoundError(
            "External Robot build lacks external_generation_manifest.json"
        )
    with generation_manifest_path.open(encoding="utf-8") as handle:
        generation_manifest = json.load(handle)
    expected_manifest = {
        "transform_version": EXTERNAL_RENDER_PROTOCOL_VERSION,
        "source_fingerprint": development_fingerprint,
        "identity_protocol": "preserve_source_split_and_robot_id",
        "label_protocol": "independent_draw_same_documented_formula",
    }
    manifest_mismatches = [
        name
        for name, expected in expected_manifest.items()
        if generation_manifest.get(name) != expected
    ]
    if manifest_mismatches:
        raise ValueError(
            "External generation manifest mismatch at: "
            + ", ".join(manifest_mismatches)
        )
    return {
        "development_root": str(development),
        "external_root": str(external),
        "development_fingerprint": development_fingerprint,
        "external_fingerprint": external_fingerprint,
        "development_unique_image_hashes": len(development_hashes),
        "external_unique_image_hashes": len(external_hashes),
        "byte_identical_image_overlap": len(overlap),
        "require_disjoint_images": bool(require_disjoint_images),
        "split_identity_sets_equal": split_identity_sets_equal,
        "identity_concept_maps_equal": concept_maps_equal,
        "forbidden_cross_root_identity_overlap": forbidden_identity_overlap,
        "generation_manifest_path": str(generation_manifest_path),
        "generation_manifest_sha256": hashlib.sha256(
            generation_manifest_path.read_bytes()
        ).hexdigest(),
        "generation_seed": generation_manifest.get("seed"),
        "transform_version": generation_manifest.get("transform_version"),
        "external_dataset_audit": external_audit,
    }
