#!/usr/bin/env python3
"""
audit_dsprites_confirmatory_2027.py
===================================
Admission gate for the confirmatory dSprites datasets.

This is deliberately written as an *independent* re-derivation of every claim
the generator makes. It does not import the generator and does not trust the
manifest: raster identity is recomputed here with BLAKE2b over the exact 32x32
bytes (the generator uses `np.unique`, a different mechanism), the label rule is
re-evaluated from `c_int`, and the one-hot encoding is rebuilt from scratch.

Writes `outputs/dsprites_confirmatory_2027_admission.json` and exits non-zero
if ANY check fails. There is no warn-and-continue path.

Usage
-----
  python scripts/audit_dsprites_confirmatory_2027.py \
      --data-dir data/dsprites/confirmatory_2027 \
      --out outputs/dsprites_confirmatory_2027_admission.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

SPLITS = ("train", "val", "test")
EXPECTED_CONCEPT_NAMES = [
    "shape::square", "shape::ellipse", "shape::heart",
    "scale::0", "scale::1", "scale::2", "scale::3", "scale::4", "scale::5",
]
EXPECTED_GROUP_BOUNDS = [[0, 3], [3, 9]]
EXPECTED_GROUP_NAMES = ["shape", "scale"]
EXPECTED_GROUP_SIZES = [3, 6]
EXPECTED_SPLIT_SEED = 2027
EXPECTED_SPLIT_STRATEGY = "grouped_exact_raster"
REQUIRED_FIELDS = [
    "imgs", "c_int", "c_bin", "group_bounds", "concept_names",
    "concept_group_names", "concept_group_sizes", "y_task", "split",
    "source_index", "latents_classes", "raster_group_id",
]
REQUIRED_MANIFEST_KEYS = [
    "dataset_version", "variant", "task_definition", "n_samples", "image_shape",
    "image_dtype", "image_min", "image_max", "concept_names",
    "concept_group_names", "concept_group_sizes", "group_bounds",
    "n_binary_concepts", "n_concept_groups", "n_classes", "split_strategy",
    "split_seed", "split_counts", "class_counts_by_split",
    "concept_tuple_counts_by_split", "source_raw_path", "source_raw_sha256",
    "source_raw_n_rows", "source_raw_is_full_official_set",
    "generator_sha256", "downsample_method", "downsample_factor",
    "ambiguous_raster_groups_removed", "ambiguous_raster_rows_removed",
    "array_content_fingerprint",
]


class Report:
    """Accumulates named boolean checks; any False sinks the whole audit."""

    def __init__(self):
        self.checks: dict = {}
        self.details: dict = {}

    def check(self, name, ok, detail=None):
        ok = bool(ok)
        self.checks[name] = ok
        if detail is not None:
            self.details[name] = detail
        flag = "PASS" if ok else "FAIL"
        extra = "" if detail is None else f"   {detail}"
        print(f"  [{flag}] {name}{extra}")
        return ok

    @property
    def passed(self):
        return all(self.checks.values())


def raster_digests(imgs: np.ndarray) -> np.ndarray:
    """Independent exact-raster identity: BLAKE2b-128 over the raw 32x32 bytes."""
    flat = np.ascontiguousarray(imgs).reshape(imgs.shape[0], -1)
    return np.array([hashlib.blake2b(row.tobytes(), digest_size=16).hexdigest()
                     for row in flat])


def canonical_partition(labels: np.ndarray) -> np.ndarray:
    """Relabel an arbitrary grouping to first-occurrence ids.

    Two groupings are the same *partition* iff their canonical forms are equal.
    This matters because the generator numbers raster groups by the lexicographic
    rank of the raster bytes while the auditor numbers them by digest -- the ids
    legitimately differ, the partition must not."""
    _, first_idx, inv = np.unique(labels, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    order = np.argsort(first_idx)
    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size)
    return remap[inv]


def oracle_labels(variant: str, c_int: np.ndarray) -> np.ndarray:
    shape, scale = c_int[:, 0], c_int[:, 1]
    if variant == "compact_c":
        return ((shape == 2) ^ (scale > 2)).astype(np.int64)
    if variant == "compact_a":
        return (shape * 6 + scale).astype(np.int64)
    raise SystemExit(f"unknown variant {variant!r}")


def rebuild_one_hot(c_int: np.ndarray) -> np.ndarray:
    n = c_int.shape[0]
    out = np.zeros((n, 9), dtype=np.uint8)
    out[np.arange(n), c_int[:, 0]] = 1
    out[np.arange(n), 3 + c_int[:, 1]] = 1
    return out


def audit_variant(data_dir: str, variant: str, rep: Report) -> dict:
    npz_path = os.path.join(data_dir, f"dsprites_{variant}_32.npz")
    json_path = os.path.join(data_dir, f"dsprites_{variant}_32.json")
    print(f"\n=== {variant} ===\n  {npz_path}")

    if not os.path.isfile(npz_path):
        rep.check(f"{variant}/npz_exists", False, npz_path)
        return {}
    rep.check(f"{variant}/npz_exists", True)

    # --- loads without allow_pickle (i.e. no object arrays anywhere) --------
    try:
        d = dict(np.load(npz_path, allow_pickle=False))
        rep.check(f"{variant}/loads_without_allow_pickle", True)
    except Exception as e:                                       # noqa: BLE001
        rep.check(f"{variant}/loads_without_allow_pickle", False, repr(e))
        return {}

    missing = [f for f in REQUIRED_FIELDS if f not in d]
    rep.check(f"{variant}/required_fields_present", not missing,
              f"missing={missing}" if missing else f"{len(REQUIRED_FIELDS)} fields")
    if missing:
        return {}

    imgs = d["imgs"]
    c_int = d["c_int"].astype(np.int64)
    c_bin = d["c_bin"]
    y = d["y_task"].astype(np.int64)
    split = d["split"].astype(str)
    src = d["source_index"].astype(np.int64)
    lat = d["latents_classes"].astype(np.int64)
    rgid = d["raster_group_id"].astype(np.int64)
    n = imgs.shape[0]

    # --- image contract ----------------------------------------------------
    rep.check(f"{variant}/image_shape_32x32", tuple(imgs.shape[1:]) == (32, 32),
              f"shape={tuple(imgs.shape)}")
    rep.check(f"{variant}/image_dtype_uint8", imgs.dtype == np.uint8, str(imgs.dtype))
    uniq = np.unique(imgs)
    rep.check(f"{variant}/image_values_in_0_1",
              np.array_equal(uniq, np.array([0, 1], dtype=np.uint8)),
              f"unique={uniq.tolist()}")

    # --- concept contract --------------------------------------------------
    rep.check(f"{variant}/n_binary_concepts_9", c_bin.shape == (n, 9), f"{c_bin.shape}")
    rep.check(f"{variant}/concept_names_exact",
              d["concept_names"].astype(str).tolist() == EXPECTED_CONCEPT_NAMES)
    rep.check(f"{variant}/concept_names_not_object",
              d["concept_names"].dtype.kind == "U", str(d["concept_names"].dtype))
    rep.check(f"{variant}/group_bounds_exact",
              d["group_bounds"].tolist() == EXPECTED_GROUP_BOUNDS)
    rep.check(f"{variant}/concept_group_names_exact",
              d["concept_group_names"].astype(str).tolist() == EXPECTED_GROUP_NAMES)
    rep.check(f"{variant}/concept_group_sizes_exact",
              d["concept_group_sizes"].tolist() == EXPECTED_GROUP_SIZES)
    rep.check(f"{variant}/c_int_ranges",
              bool(((c_int[:, 0] >= 0) & (c_int[:, 0] <= 2)).all()
                   and ((c_int[:, 1] >= 0) & (c_int[:, 1] <= 5)).all()))

    shape_sum = c_bin[:, 0:3].sum(axis=1)
    scale_sum = c_bin[:, 3:9].sum(axis=1)
    rep.check(f"{variant}/shape_onehot_exactly_one", bool((shape_sum == 1).all()),
              f"violations={int((shape_sum != 1).sum())}")
    rep.check(f"{variant}/scale_onehot_exactly_one", bool((scale_sum == 1).all()),
              f"violations={int((scale_sum != 1).sum())}")
    rep.check(f"{variant}/c_bin_matches_c_int",
              np.array_equal(c_bin, rebuild_one_hot(c_int)))

    # --- label contract ----------------------------------------------------
    y_oracle = oracle_labels(variant, c_int)
    oracle_acc = float((y == y_oracle).mean())
    rep.check(f"{variant}/oracle_accuracy_100", oracle_acc == 1.0,
              f"acc={oracle_acc:.10f}")
    rep.check(f"{variant}/y_dtype_int64", y.dtype == np.int64, str(y.dtype))

    ckey = c_int[:, 0] * 6 + c_int[:, 1]
    order = np.lexsort((y, ckey))
    ck_s, y_s = ckey[order], y[order]
    boundary = np.flatnonzero(np.diff(ck_s) != 0)
    multi = 0
    starts = np.concatenate(([0], boundary + 1))
    ends = np.concatenate((boundary + 1, [len(ck_s)]))
    for a, b in zip(starts, ends):
        if np.unique(y_s[a:b]).size > 1:
            multi += 1
    rep.check(f"{variant}/concept_vectors_with_multiple_labels_0", multi == 0,
              f"count={multi}")
    rep.check(f"{variant}/concept_to_label_deterministic", multi == 0)

    # --- exact-raster integrity (independent BLAKE2b re-derivation) --------
    dig = raster_digests(imgs)

    # ambiguity: same raster, different concept vector or different label
    order = np.argsort(dig, kind="stable")
    dig_s, ck_s, y_s = dig[order], ckey[order], y[order]
    bnd = np.flatnonzero(dig_s[1:] != dig_s[:-1])
    starts = np.concatenate(([0], bnd + 1))
    ends = np.concatenate((bnd + 1, [len(dig_s)]))
    amb_groups = 0
    amb_rows = 0
    for a, b in zip(starts, ends):
        if np.unique(ck_s[a:b]).size > 1 or np.unique(y_s[a:b]).size > 1:
            amb_groups += 1
            amb_rows += b - a
    rep.check(f"{variant}/ambiguous_exact_raster_groups_0", amb_groups == 0,
              f"groups={amb_groups} rows={amb_rows}")

    # `raster_group_id` must induce exactly the same partition as true raster
    # identity -- neither coarser (two different rasters merged) nor finer
    # (identical rasters split apart). Compared as partitions, not as id values.
    rep.check(f"{variant}/raster_group_id_consistent",
              np.array_equal(canonical_partition(rgid), canonical_partition(dig)),
              f"n_groups={np.unique(rgid).size} vs {np.unique(dig).size}")

    # --- split integrity ---------------------------------------------------
    rep.check(f"{variant}/split_values_valid",
              set(np.unique(split).tolist()) == set(SPLITS),
              f"{sorted(set(np.unique(split).tolist()))}")
    counts = {s: int((split == s).sum()) for s in SPLITS}
    rep.check(f"{variant}/splits_partition_all_rows",
              sum(counts.values()) == n, f"{counts} sum={sum(counts.values())} n={n}")

    sets = {s: set(dig[split == s].tolist()) for s in SPLITS}
    overlaps = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    for k, v in overlaps.items():
        rep.check(f"{variant}/exact_raster_overlap_{k}_0", v == 0, f"overlap={v}")

    rep.check(f"{variant}/source_index_unique",
              np.unique(src).size == n, f"unique={np.unique(src).size} n={n}")
    rep.check(f"{variant}/source_index_in_raw_range",
              bool((src >= 0).all() and (src < 737_280).all()))

    # latents must be self-consistent with the concepts and the sub-selection grid
    rep.check(f"{variant}/latents_shape_6", lat.shape == (n, 6), f"{lat.shape}")
    rep.check(f"{variant}/latents_match_c_int",
              np.array_equal(lat[:, 1], c_int[:, 0]) and np.array_equal(lat[:, 2], c_int[:, 1]))
    rep.check(f"{variant}/latents_on_subselection_grid",
              bool(np.isin(lat[:, 3], np.arange(0, 40, 5)).all()
                   and np.isin(lat[:, 4], np.arange(0, 32, 2)).all()
                   and np.isin(lat[:, 5], np.arange(0, 32, 2)).all()))
    # raw enumeration: index = ((((shape*6+scale)*40+orient)*32+posX)*32+posY)
    recomputed = ((((lat[:, 1] * 6 + lat[:, 2]) * 40 + lat[:, 3]) * 32 + lat[:, 4]) * 32
                  + lat[:, 5])
    rep.check(f"{variant}/source_index_matches_latents",
              np.array_equal(recomputed, src))

    # --- per-split coverage ------------------------------------------------
    tuples_by_split = {}
    labels_by_split = {}
    for s in SPLITS:
        t = np.unique(ckey[split == s])
        tuples_by_split[s] = t.size
        labels_by_split[s] = np.unique(y[split == s]).size
    rep.check(f"{variant}/all_splits_have_18_concept_tuples",
              all(v == 18 for v in tuples_by_split.values()), str(tuples_by_split))
    n_expected_labels = 2 if variant == "compact_c" else 18
    rep.check(f"{variant}/all_splits_have_all_labels",
              all(v == n_expected_labels for v in labels_by_split.values()),
              str(labels_by_split))

    # --- manifest ----------------------------------------------------------
    if not os.path.isfile(json_path):
        rep.check(f"{variant}/manifest_exists", False, json_path)
        meta = {}
    else:
        rep.check(f"{variant}/manifest_exists", True)
        with open(json_path) as f:
            meta = json.load(f)
        missing_keys = [k for k in REQUIRED_MANIFEST_KEYS if k not in meta]
        rep.check(f"{variant}/manifest_required_keys", not missing_keys,
                  f"missing={missing_keys}" if missing_keys else
                  f"{len(REQUIRED_MANIFEST_KEYS)} keys")
        rep.check(f"{variant}/manifest_split_seed_2027",
                  meta.get("split_seed") == EXPECTED_SPLIT_SEED, str(meta.get("split_seed")))
        rep.check(f"{variant}/manifest_split_strategy",
                  meta.get("split_strategy") == EXPECTED_SPLIT_STRATEGY,
                  str(meta.get("split_strategy")))
        rep.check(f"{variant}/manifest_split_counts_match",
                  meta.get("split_counts") == counts, str(meta.get("split_counts")))
        rep.check(f"{variant}/manifest_n_samples_match", meta.get("n_samples") == n)
        rep.check(f"{variant}/manifest_downsample_max_pool",
                  meta.get("downsample_method") == "max_pool_2x2"
                  and meta.get("downsample_factor") == 2,
                  f"{meta.get('downsample_method')} x{meta.get('downsample_factor')}")
        rep.check(f"{variant}/manifest_raw_sha256_present",
                  isinstance(meta.get("source_raw_sha256"), str)
                  and len(meta["source_raw_sha256"]) == 64)
        rep.check(f"{variant}/manifest_source_raw_n_rows_official",
                  meta.get("source_raw_n_rows") == 737_280,
                  str(meta.get("source_raw_n_rows")))
        rep.check(f"{variant}/manifest_source_is_full_official_set",
                  meta.get("source_raw_is_full_official_set") is True,
                  str(meta.get("source_raw_is_full_official_set")))
        rep.check(f"{variant}/manifest_generator_sha256_present",
                  isinstance(meta.get("generator_sha256"), str)
                  and len(meta["generator_sha256"]) == 64)
        # fingerprint recomputed from the decompressed arrays
        h = hashlib.sha256()
        for key in sorted(d):
            a = np.ascontiguousarray(d[key])
            h.update(key.encode()); h.update(str(a.dtype).encode())
            h.update(repr(a.shape).encode()); h.update(a.tobytes())
        rep.check(f"{variant}/array_content_fingerprint_matches",
                  h.hexdigest() == meta.get("array_content_fingerprint"),
                  h.hexdigest())

    return dict(
        n_samples=int(n),
        split_counts=counts,
        cross_split_exact_raster_overlaps=overlaps,
        ambiguous_exact_raster_groups=int(amb_groups),
        ambiguous_exact_raster_rows=int(amb_rows),
        n_raster_groups=int(np.unique(dig).size),
        concept_vectors_with_multiple_labels=int(multi),
        concept_to_label_deterministic=bool(multi == 0),
        oracle_accuracy=oracle_acc,
        concept_tuples_by_split=tuples_by_split,
        label_classes_by_split=labels_by_split,
        class_counts_by_split={
            s: {int(k): int(v) for k, v in zip(*np.unique(y[split == s], return_counts=True))}
            for s in SPLITS},
        concept_tuple_counts_by_split={
            s: {f"{int(k)//6}_{int(k)%6}": int(v)
                for k, v in zip(*np.unique(ckey[split == s], return_counts=True))}
            for s in SPLITS},
        ambiguous_raster_groups_removed=meta.get("ambiguous_raster_groups_removed"),
        ambiguous_raster_rows_removed=meta.get("ambiguous_raster_rows_removed"),
        source_raw_path=meta.get("source_raw_path"),
        source_raw_sha256=meta.get("source_raw_sha256"),
        source_raw_n_rows=meta.get("source_raw_n_rows"),
        source_raw_is_full_official_set=meta.get("source_raw_is_full_official_set"),
        generator_sha256=meta.get("generator_sha256"),
        array_content_fingerprint=meta.get("array_content_fingerprint"),
        _arrays=d,
    )


def audit_variant_pair(data_dir, res_c, res_a, rep):
    """compact_a must be row-identical to compact_c except for y_task."""
    print("\n=== compact_c / compact_a alignment ===")
    a, c = res_a.get("_arrays"), res_c.get("_arrays")
    if not a or not c:
        rep.check("pair/both_variants_loaded", False)
        return
    rep.check("pair/both_variants_loaded", True)
    for field in ("imgs", "c_int", "c_bin", "split", "source_index",
                  "latents_classes", "raster_group_id"):
        rep.check(f"pair/identical_{field}", np.array_equal(a[field], c[field]))
    rep.check("pair/y_task_differs", not np.array_equal(a["y_task"], c["y_task"]))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/dsprites/confirmatory_2027")
    p.add_argument("--out", default="outputs/dsprites_confirmatory_2027_admission.json")
    p.add_argument("--variants", nargs="+", default=["compact_c", "compact_a"])
    args = p.parse_args(argv)

    rep = Report()
    print(f"auditing {os.path.abspath(args.data_dir)}")
    results = {v: audit_variant(args.data_dir, v, rep) for v in args.variants}

    if "compact_c" in results and "compact_a" in results:
        audit_variant_pair(args.data_dir, results["compact_c"], results["compact_a"], rep)

    for v in results:
        results[v].pop("_arrays", None)

    admitted = rep.passed
    payload = dict(
        admitted_for_confirmatory_sweep=admitted,
        audited_at_utc=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        data_dir=os.path.abspath(args.data_dir),
        auditor_path=os.path.abspath(__file__),
        auditor_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        n_checks=len(rep.checks),
        n_failed=sum(1 for v in rep.checks.values() if not v),
        failed_checks=[k for k, v in rep.checks.items() if not v],
        checks=rep.checks,
        check_details=rep.details,
        variants=results,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"\n{'-'*70}")
    print(f"checks: {len(rep.checks)}   failed: {payload['n_failed']}")
    print(f"admitted_for_confirmatory_sweep = {str(admitted).lower()}")
    print(f"wrote {os.path.abspath(args.out)}")
    if not admitted:
        print("\nFAILED CHECKS:")
        for k in payload["failed_checks"]:
            print(f"  - {k}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
