"""
test_dsprites_confirmatory_2027.py
==================================
Acceptance tests for data/dsprites/confirmatory_2027.

Covers the minimum acceptance criteria (spec section 12), the admission gate
(section 9), and -- importantly -- negative tests proving the auditor actually
*fails* on corrupted data instead of warning and continuing.

Run:
  python -m pytest tests/test_dsprites_confirmatory_2027.py -v
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "dsprites", "confirmatory_2027")
AUDITOR = os.path.join(ROOT, "scripts", "audit_dsprites_confirmatory_2027.py")
ADMISSION = os.path.join(ROOT, "outputs", "dsprites_confirmatory_2027_admission.json")
SPLITS = ("train", "val", "test")
VARIANTS = ("compact_c", "compact_a")


def npz_path(variant):
    return os.path.join(DATA_DIR, f"dsprites_{variant}_32.npz")


def json_path(variant):
    return os.path.join(DATA_DIR, f"dsprites_{variant}_32.json")


@pytest.fixture(scope="module")
def d_c():
    with np.load(npz_path("compact_c"), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


@pytest.fixture(scope="module")
def d_a():
    with np.load(npz_path("compact_a"), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


@pytest.fixture(scope="module")
def meta_c():
    with open(json_path("compact_c")) as f:
        return json.load(f)


def digests(imgs):
    flat = np.ascontiguousarray(imgs).reshape(imgs.shape[0], -1)
    return np.array([hashlib.blake2b(r.tobytes(), digest_size=16).hexdigest() for r in flat])


# ---------------------------------------------------------------- section 12
def test_files_exist():
    for v in VARIANTS:
        assert os.path.isfile(npz_path(v)), npz_path(v)
        assert os.path.isfile(json_path(v)), json_path(v)


def test_loads_without_allow_pickle():
    """No Python object arrays anywhere -- loaders must not need allow_pickle."""
    for v in VARIANTS:
        with np.load(npz_path(v), allow_pickle=False) as f:
            for k in f.files:
                assert f[k].dtype.kind != "O", k


def test_image_contract(d_c):
    imgs = d_c["imgs"]
    assert imgs.shape[1:] == (32, 32)
    assert imgs.dtype == np.uint8
    assert np.array_equal(np.unique(imgs), np.array([0, 1], dtype=np.uint8))
    # explicitly: not 0/255, so a loader must NOT divide by 255
    assert imgs.max() == 1


def test_concept_contract(d_c):
    assert d_c["c_bin"].shape[1] == 9
    assert d_c["group_bounds"].tolist() == [[0, 3], [3, 9]]
    assert d_c["concept_group_names"].astype(str).tolist() == ["shape", "scale"]
    assert d_c["concept_group_sizes"].tolist() == [3, 6]
    assert d_c["concept_names"].astype(str).tolist() == [
        "shape::square", "shape::ellipse", "shape::heart",
        "scale::0", "scale::1", "scale::2", "scale::3", "scale::4", "scale::5"]


def test_onehot_wellformed(d_c):
    c_bin = d_c["c_bin"]
    assert (c_bin[:, 0:3].sum(axis=1) == 1).all()
    assert (c_bin[:, 3:9].sum(axis=1) == 1).all()


def test_c_bin_matches_c_int(d_c):
    c_int, c_bin = d_c["c_int"], d_c["c_bin"]
    n = c_int.shape[0]
    rebuilt = np.zeros((n, 9), dtype=np.uint8)
    rebuilt[np.arange(n), c_int[:, 0]] = 1
    rebuilt[np.arange(n), 3 + c_int[:, 1]] = 1
    assert np.array_equal(c_bin, rebuilt)


def test_label_rule_compact_c(d_c):
    c_int, y = d_c["c_int"], d_c["y_task"]
    assert y.dtype == np.int64
    expected = ((c_int[:, 0] == 2) ^ (c_int[:, 1] > 2)).astype(np.int64)
    assert np.array_equal(y, expected)          # oracle accuracy == 100%
    assert set(np.unique(y).tolist()) == {0, 1}


def test_label_rule_compact_a(d_a):
    c_int, y = d_a["c_int"], d_a["y_task"]
    assert np.array_equal(y, (c_int[:, 0] * 6 + c_int[:, 1]).astype(np.int64))
    assert np.unique(y).size == 18


def test_label_rule_identical_across_splits(d_c):
    """The rule must not be split-dependent and must carry no label noise."""
    c_int, y, split = d_c["c_int"], d_c["y_task"], d_c["split"].astype(str)
    key = c_int[:, 0] * 6 + c_int[:, 1]
    mapping = {}
    for s in SPLITS:
        m = split == s
        for k in np.unique(key[m]):
            ys = np.unique(y[m][key[m] == k])
            assert ys.size == 1, f"concept {k} has labels {ys} in split {s}"
            mapping.setdefault(k, set()).add(int(ys[0]))
    assert all(len(v) == 1 for v in mapping.values())


def test_concept_to_label_deterministic(d_c):
    c_int, y = d_c["c_int"], d_c["y_task"]
    key = c_int[:, 0] * 6 + c_int[:, 1]
    for k in np.unique(key):
        assert np.unique(y[key == k]).size == 1


def test_split_seed_and_strategy(meta_c):
    assert meta_c["split_seed"] == 2027
    assert meta_c["split_strategy"] == "grouped_exact_raster"


# ---------------------------------------------------------------- section 5/6
def test_no_ambiguous_rasters(d_c):
    dig = digests(d_c["imgs"])
    key = d_c["c_int"][:, 0] * 6 + d_c["c_int"][:, 1]
    y = d_c["y_task"]
    order = np.argsort(dig, kind="stable")
    dig_s, key_s, y_s = dig[order], key[order], y[order]
    bnd = np.flatnonzero(dig_s[1:] != dig_s[:-1])
    starts = np.concatenate(([0], bnd + 1))
    ends = np.concatenate((bnd + 1, [len(dig_s)]))
    for a, b in zip(starts, ends):
        assert np.unique(key_s[a:b]).size == 1
        assert np.unique(y_s[a:b]).size == 1


def test_cross_split_exact_raster_overlap_is_zero(d_c):
    dig = digests(d_c["imgs"])
    split = d_c["split"].astype(str)
    sets = {s: set(dig[split == s].tolist()) for s in SPLITS}
    assert len(sets["train"] & sets["val"]) == 0
    assert len(sets["train"] & sets["test"]) == 0
    assert len(sets["val"] & sets["test"]) == 0


def test_raster_group_never_straddles_a_split(d_c):
    """Stronger than pairwise overlap: every raster group is split-homogeneous."""
    dig = digests(d_c["imgs"])
    split = d_c["split"].astype(str)
    order = np.argsort(dig, kind="stable")
    dig_s, split_s = dig[order], split[order]
    bnd = np.flatnonzero(dig_s[1:] != dig_s[:-1])
    starts = np.concatenate(([0], bnd + 1))
    ends = np.concatenate((bnd + 1, [len(dig_s)]))
    for a, b in zip(starts, ends):
        assert np.unique(split_s[a:b]).size == 1


def test_splits_partition_dataset(d_c):
    split = d_c["split"].astype(str)
    n = d_c["imgs"].shape[0]
    assert set(np.unique(split).tolist()) == set(SPLITS)
    assert sum(int((split == s).sum()) for s in SPLITS) == n


def test_split_proportions_close_to_70_15_15(d_c):
    split = d_c["split"].astype(str)
    n = split.size
    for s, target in zip(SPLITS, (0.70, 0.15, 0.15)):
        frac = (split == s).sum() / n
        assert abs(frac - target) < 0.01, f"{s}={frac:.4f}"


def test_all_splits_cover_18_concept_tuples_and_both_labels(d_c):
    c_int, y, split = d_c["c_int"], d_c["y_task"], d_c["split"].astype(str)
    key = c_int[:, 0] * 6 + c_int[:, 1]
    for s in SPLITS:
        assert np.unique(key[split == s]).size == 18, s
        assert np.unique(y[split == s]).size == 2, s


def test_source_index_unique_and_consistent_with_latents(d_c):
    src, lat = d_c["source_index"], d_c["latents_classes"]
    assert np.unique(src).size == src.size
    assert (src >= 0).all() and (src < 737_280).all()
    recomputed = ((((lat[:, 1] * 6 + lat[:, 2]) * 40 + lat[:, 3]) * 32 + lat[:, 4]) * 32
                  + lat[:, 5])
    assert np.array_equal(recomputed, src)


def test_latents_on_subselection_grid(d_c):
    lat = d_c["latents_classes"]
    assert lat.shape[1] == 6
    assert np.isin(lat[:, 3], np.arange(0, 40, 5)).all()
    assert np.isin(lat[:, 4], np.arange(0, 32, 2)).all()
    assert np.isin(lat[:, 5], np.arange(0, 32, 2)).all()
    assert np.array_equal(lat[:, 1], d_c["c_int"][:, 0])
    assert np.array_equal(lat[:, 2], d_c["c_int"][:, 1])


# ---------------------------------------------------------------- section 10
def test_compact_a_is_row_aligned_with_compact_c(d_c, d_a):
    for field in ("imgs", "c_int", "c_bin", "split", "source_index",
                  "latents_classes", "raster_group_id"):
        assert np.array_equal(d_a[field], d_c[field]), field
    assert not np.array_equal(d_a["y_task"], d_c["y_task"])


# ---------------------------------------------------------------- manifest
def test_manifest_completeness(meta_c):
    required = [
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
    missing = [k for k in required if k not in meta_c]
    assert not missing, missing
    assert meta_c["image_shape"] == [32, 32]
    assert meta_c["downsample_method"] == "max_pool_2x2"
    assert meta_c["n_binary_concepts"] == 9
    assert meta_c["n_classes"] == 2
    assert len(meta_c["source_raw_sha256"]) == 64
    assert meta_c["source_raw_n_rows"] == 737_280
    assert meta_c["source_raw_is_full_official_set"] is True
    assert len(meta_c["generator_sha256"]) == 64


def test_fingerprint_is_over_array_contents_not_the_zip(d_c, meta_c):
    h = hashlib.sha256()
    for k in sorted(d_c):
        a = np.ascontiguousarray(d_c[k])
        h.update(k.encode()); h.update(str(a.dtype).encode())
        h.update(repr(a.shape).encode()); h.update(a.tobytes())
    assert h.hexdigest() == meta_c["array_content_fingerprint"]


# ---------------------------------------------------------------- section 9
def test_admission_json_says_admitted():
    assert os.path.isfile(ADMISSION), ADMISSION
    with open(ADMISSION) as f:
        adm = json.load(f)
    assert adm["admitted_for_confirmatory_sweep"] is True
    assert adm["n_failed"] == 0
    assert adm["failed_checks"] == []
    for key in ("exact_raster_overlap_train_val_0", "exact_raster_overlap_train_test_0",
                "exact_raster_overlap_val_test_0", "ambiguous_exact_raster_groups_0",
                "concept_vectors_with_multiple_labels_0", "concept_to_label_deterministic",
                "oracle_accuracy_100", "shape_onehot_exactly_one",
                "scale_onehot_exactly_one", "c_bin_matches_c_int",
                "source_index_unique", "all_splits_have_18_concept_tuples",
                "all_splits_have_all_labels", "manifest_source_raw_n_rows_official",
                "manifest_source_is_full_official_set"):
        assert adm["checks"][f"compact_c/{key}"] is True, key


def test_auditor_exits_zero_on_good_data(tmp_path):
    r = subprocess.run(
        [sys.executable, AUDITOR, "--data-dir", DATA_DIR,
         "--out", str(tmp_path / "adm.json")],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:]


# ---------------------------------------------------------------- negative tests
def _corrupt_copy(tmp_path, mutate):
    """Copy the dataset dir, mutate compact_c's arrays, return the new dir."""
    d = tmp_path / "corrupt"
    shutil.copytree(DATA_DIR, d)
    p = d / "dsprites_compact_c_32.npz"
    with np.load(p, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    mutate(arrays)
    np.savez_compressed(p, **arrays)
    return d


def _audit_fails(tmp_path, d, expect_substr):
    r = subprocess.run(
        [sys.executable, AUDITOR, "--data-dir", str(d), "--variants", "compact_c",
         "--out", str(tmp_path / "adm.json")],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0, "auditor did not exit non-zero on corrupted data"
    with open(tmp_path / "adm.json") as f:
        adm = json.load(f)
    assert adm["admitted_for_confirmatory_sweep"] is False
    assert any(expect_substr in c for c in adm["failed_checks"]), adm["failed_checks"]


def test_auditor_catches_cross_split_raster_leak(tmp_path):
    def mutate(a):
        # copy a train row's raster onto a test row -> raster now in two splits
        split = a["split"].astype(str)
        tr = int(np.flatnonzero(split == "train")[0])
        te = int(np.flatnonzero(split == "test")[0])
        a["imgs"][te] = a["imgs"][tr]
        a["c_int"][te] = a["c_int"][tr]
        a["c_bin"][te] = a["c_bin"][tr]
        a["y_task"][te] = a["y_task"][tr]
    _audit_fails(tmp_path, _corrupt_copy(tmp_path, mutate), "exact_raster_overlap")


def test_auditor_catches_flipped_label(tmp_path):
    def mutate(a):
        a["y_task"][0] = 1 - a["y_task"][0]
    _audit_fails(tmp_path, _corrupt_copy(tmp_path, mutate), "oracle_accuracy_100")


def test_auditor_catches_nondeterministic_concept_to_label(tmp_path):
    def mutate(a):
        key = a["c_int"][:, 0] * 6 + a["c_int"][:, 1]
        rows = np.flatnonzero(key == key[0])
        a["y_task"][rows[0]] = 1 - a["y_task"][rows[0]]
    _audit_fails(tmp_path, _corrupt_copy(tmp_path, mutate),
                 "concept_vectors_with_multiple_labels")


def test_auditor_catches_broken_onehot(tmp_path):
    def mutate(a):
        a["c_bin"][0, 0:3] = 0
    _audit_fails(tmp_path, _corrupt_copy(tmp_path, mutate), "shape_onehot_exactly_one")


def test_auditor_catches_rescaled_pixels(tmp_path):
    def mutate(a):
        a["imgs"] = (a["imgs"] * 255).astype(np.uint8)
    _audit_fails(tmp_path, _corrupt_copy(tmp_path, mutate), "image_values_in_0_1")


def test_auditor_catches_duplicate_source_index(tmp_path):
    def mutate(a):
        a["source_index"][1] = a["source_index"][0]
    _audit_fails(tmp_path, _corrupt_copy(tmp_path, mutate), "source_index")


def test_auditor_catches_missing_concept_tuple_in_a_split(tmp_path):
    def mutate(a):
        split = a["split"].astype(str)
        key = a["c_int"][:, 0] * 6 + a["c_int"][:, 1]
        victim = (split == "test") & (key == 0)
        split[victim] = "train"
        a["split"] = split.astype("<U5")
    _audit_fails(tmp_path, _corrupt_copy(tmp_path, mutate),
                 "all_splits_have_18_concept_tuples")


def test_auditor_catches_stale_fingerprint(tmp_path):
    def mutate(a):
        a["imgs"][0, 0, 0] = 1 - a["imgs"][0, 0, 0]
    _audit_fails(tmp_path, _corrupt_copy(tmp_path, mutate),
                 "array_content_fingerprint_matches")
