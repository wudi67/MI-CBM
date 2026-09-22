#!/usr/bin/env python3
"""
make_dsprites_confirmatory.py
=============================
Generate the *confirmatory* dSprites datasets for the Quantum CBM experiments.

This regenerates everything from the official raw dSprites NPZ. It deliberately
does NOT read `data/dsprites/dsprites_compact/*.npz` -- those are the old
sample-level splits that leak identical rasters across splits and contain
image/annotation ambiguities.

Pipeline
--------
  1. load raw NPZ                     (737,280 x 64 x 64, uint8 {0,1})
  2. CME sub-selection                (3 x 6 x 8 x 16 x 16 = 36,864 rows,
                                       raw enumeration order preserved)
  3. 2x2 max-pool  64x64 -> 32x32     (uint8, values stay in {0,1})
  4. exact-raster grouping            (bit-exact, via np.unique(axis=0))
  5. ambiguity removal                (drop *every* row of any raster group
                                       carrying >1 concept vector or >1 label)
  6. grouped stratified split         (raster group is atomic; stratified over
                                       the full 18 (shape,scale) tuples)
  7. write NPZ + provenance manifest

Variants
--------
  compact_c   y = (shape == heart) XOR (scale > 2)      2 classes   [primary]
  compact_a   y = shape * 6 + scale                    18 classes   [auxiliary]

compact_a is emitted from the *same* rows, in the *same* order, with the *same*
concepts and the *same* split as compact_c; only `y_task` differs. That makes
one X->C checkpoint reusable across both C->Y decoders.

Usage
-----
  python scripts/make_dsprites_confirmatory.py \
      --raw /path/to/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
      --out data/dsprites/confirmatory_2027 \
      --variants compact_c compact_a
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

import numpy as np

# --------------------------------------------------------------------------- constants
DATASET_VERSION = "confirmatory_2027.v1"
SPLIT_SEED = 2027
SPLIT_FRACS = (0.70, 0.15, 0.15)
SPLIT_STRATEGY = "grouped_exact_raster"

RAW_FACTOR_NAMES = ["color", "shape", "scale", "orientation", "posX", "posY"]
RAW_FACTOR_SIZES = [1, 3, 6, 40, 32, 32]
RAW_N = 737_280

SHAPE_NAMES = ["square", "ellipse", "heart"]
CONCEPT_GROUP_NAMES = ["shape", "scale"]
CONCEPT_GROUP_SIZES = [3, 6]
GROUP_BOUNDS = [[0, 3], [3, 9]]
CONCEPT_NAMES = [f"shape::{s}" for s in SHAPE_NAMES] + [f"scale::{i}" for i in range(6)]

# CME sub-selection grid (indices into the *raw* latent classes)
SEL_ORIENTATION = np.arange(0, 40, 5)   # 0,5,...,35   -> 8
SEL_POSX = np.arange(0, 32, 2)          # 0,2,...,30   -> 16
SEL_POSY = np.arange(0, 32, 2)          # 0,2,...,30   -> 16
EXPECTED_PRE_AMBIGUITY_N = 3 * 6 * 8 * 16 * 16   # 36,864

DOWNSAMPLE_METHOD = "max_pool_2x2"
DOWNSAMPLE_FACTOR = 2

TASK_DEFINITIONS = {
    "compact_c": "y_task = int((shape == 2) ^ (scale > 2))   # (shape==heart) XOR (scale>2)",
    "compact_a": "y_task = shape * 6 + scale                 # 18 classes",
}
N_CLASSES = {"compact_c": 2, "compact_a": 18}


# --------------------------------------------------------------------------- hashing
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def array_content_fingerprint(arrays: dict) -> str:
    """SHA256 over the *decompressed* contents of the core arrays.

    Covers dtype, shape and raw bytes of every array, in sorted key order, so it
    is independent of NPZ compression settings, zip timestamps and member order.
    """
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(repr(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- steps
def raw_row_index(lat: np.ndarray) -> np.ndarray:
    """Canonical dSprites row index implied by a latent-class row.

    Raw dSprites is a complete lexicographic enumeration over
    (color, shape, scale, orientation, posX, posY) with sizes
    [1, 3, 6, 40, 32, 32], so the row index is fully determined by the latents:

        idx = ((((shape*6 + scale)*40 + orient)*32 + posX)*32 + posY)

    Deriving `source_index` this way (rather than from the row's position in the
    file) makes it verifiable and keeps it meaningful even when the generator is
    exercised against a reduced test fixture."""
    return ((((lat[:, 1] * 6 + lat[:, 2]) * 40 + lat[:, 3]) * 32 + lat[:, 4]) * 32
            + lat[:, 5]).astype(np.int64)


def load_raw(path: str):
    """Load raw dSprites. `allow_pickle` is needed only for the raw file's
    own metadata field; we read strictly `imgs` and `latents_classes`."""
    d = np.load(path, allow_pickle=True, encoding="latin1")
    if "imgs" not in d or "latents_classes" not in d:
        raise SystemExit(f"raw NPZ missing 'imgs'/'latents_classes': got {list(d.keys())}")
    imgs = np.asarray(d["imgs"])
    lat = np.asarray(d["latents_classes"]).astype(np.int64)

    if imgs.ndim != 3 or imgs.shape[1:] != (64, 64):
        raise SystemExit(f"raw imgs must be (N,64,64), got {imgs.shape}")
    if lat.shape != (imgs.shape[0], 6):
        raise SystemExit(f"raw latents must be (N,6) matching imgs, got {lat.shape}")
    vals = np.unique(imgs)
    if not np.array_equal(vals, np.array([0, 1], dtype=vals.dtype)):
        raise SystemExit(f"raw imgs must be binary {{0,1}}, found values {vals[:10]}")
    for col, size in enumerate(RAW_FACTOR_SIZES):
        if lat[:, col].min() < 0 or lat[:, col].max() >= size:
            raise SystemExit(f"latent column {col} ({RAW_FACTOR_NAMES[col]}) out of range "
                             f"[0,{size}); got [{lat[:, col].min()},{lat[:, col].max()}]")
    if np.unique(raw_row_index(lat)).size != lat.shape[0]:
        raise SystemExit("raw latents contain duplicate (shape,scale,orient,posX,posY) rows")

    is_full = imgs.shape[0] == RAW_N
    if is_full and not np.array_equal(raw_row_index(lat), np.arange(RAW_N)):
        raise SystemExit("full-size raw is not in canonical lexicographic latent order")
    return imgs.astype(np.uint8, copy=False), lat, is_full


def select_rows(lat: np.ndarray) -> np.ndarray:
    """Boolean CME sub-selection mask over the raw rows (order preserved)."""
    keep = (
        np.isin(lat[:, 3], SEL_ORIENTATION)
        & np.isin(lat[:, 4], SEL_POSX)
        & np.isin(lat[:, 5], SEL_POSY)
    )
    n = int(keep.sum())
    if n != EXPECTED_PRE_AMBIGUITY_N:
        raise SystemExit(f"sub-selection produced {n} rows, expected {EXPECTED_PRE_AMBIGUITY_N} "
                         f"(3x6x8x16x16); the raw file does not cover the required grid")
    return keep


def max_pool_2x2(imgs: np.ndarray) -> np.ndarray:
    """64x64 -> 32x32 via 2x2 max-pooling. Max-pool (not average) keeps thin
    shapes from vanishing and keeps pixel values in {0,1} without rescaling."""
    n, h, w = imgs.shape
    f = DOWNSAMPLE_FACTOR
    out = imgs.reshape(n, h // f, f, w // f, f).max(axis=(2, 4)).astype(np.uint8)
    vals = np.unique(out)
    if not np.array_equal(vals, np.array([0, 1], dtype=np.uint8)):
        raise SystemExit(f"downsampled imgs must be binary {{0,1}}, found {vals}")
    return out


def raster_groups(imgs: np.ndarray):
    """Bit-exact raster grouping. Returns (group_id per row, n_groups).

    Uses np.unique on the flattened rasters -- exact equality, no hash
    collisions. Group ids are the lexicographic rank of the raster content,
    hence deterministic and independent of row order."""
    flat = np.ascontiguousarray(imgs).reshape(imgs.shape[0], -1)
    _, inverse = np.unique(flat, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel().astype(np.int64)
    return inverse, int(inverse.max()) + 1


def find_ambiguous_groups(gid: np.ndarray, c_int: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Raster groups whose rows do not agree on the concept vector or the label.

    Returns a boolean mask over group ids."""
    n_groups = int(gid.max()) + 1
    # encode (shape, scale) into one integer; 6 scale values
    ckey = c_int[:, 0] * 6 + c_int[:, 1]

    ambiguous = np.zeros(n_groups, dtype=bool)
    for arr in (ckey, y):
        lo = np.full(n_groups, np.iinfo(np.int64).max, dtype=np.int64)
        hi = np.full(n_groups, np.iinfo(np.int64).min, dtype=np.int64)
        np.minimum.at(lo, gid, arr.astype(np.int64))
        np.maximum.at(hi, gid, arr.astype(np.int64))
        ambiguous |= lo != hi
    return ambiguous


def grouped_stratified_split(gid: np.ndarray, strata: np.ndarray, seed: int) -> np.ndarray:
    """70/15/15 split where the atomic unit is a raster group.

    Every copy of a raster lands in exactly one split (no cross-split raster
    overlap by construction). Stratification is over the full (shape, scale)
    concept tuple, not just the binary label, so all 18 tuples appear in all
    three splits."""
    rng = np.random.default_rng(seed)
    n = gid.shape[0]
    split = np.empty(n, dtype="<U5")

    # rows per group, and the stratum of each group (constant within a group,
    # guaranteed because ambiguous groups were already removed)
    n_groups = int(gid.max()) + 1
    counts = np.bincount(gid, minlength=n_groups)
    g_stratum = np.full(n_groups, -1, dtype=np.int64)
    g_stratum[gid] = strata

    group_split = np.empty(n_groups, dtype="<U5")

    for s in np.unique(strata):                       # deterministic: sorted
        gids = np.flatnonzero(g_stratum == s)
        if gids.size < 3:
            raise SystemExit(f"stratum {s} has only {gids.size} raster groups; "
                             "cannot guarantee non-empty train/val/test")
        gids = gids[rng.permutation(gids.size)]
        sizes = counts[gids]
        total = int(sizes.sum())
        cum = np.cumsum(sizes)

        i1 = int(np.searchsorted(cum, SPLIT_FRACS[0] * total, side="left")) + 1
        i2 = int(np.searchsorted(cum, (SPLIT_FRACS[0] + SPLIT_FRACS[1]) * total, side="left")) + 1
        # keep every split non-empty inside every stratum
        i1 = min(max(i1, 1), gids.size - 2)
        i2 = min(max(i2, i1 + 1), gids.size - 1)

        group_split[gids[:i1]] = "train"
        group_split[gids[i1:i2]] = "val"
        group_split[gids[i2:]] = "test"

    split[:] = group_split[gid]
    return split


# --------------------------------------------------------------------------- output
def build_payload(imgs, c_int, y, split, source_index, latents, raster_gid):
    return dict(
        imgs=imgs.astype(np.uint8, copy=False),
        c_int=c_int.astype(np.int64, copy=False),
        c_bin=one_hot(c_int),
        group_bounds=np.array(GROUP_BOUNDS, dtype=np.int64),
        concept_names=np.array(CONCEPT_NAMES, dtype=np.str_),
        concept_group_names=np.array(CONCEPT_GROUP_NAMES, dtype=np.str_),
        concept_group_sizes=np.array(CONCEPT_GROUP_SIZES, dtype=np.int64),
        y_task=y.astype(np.int64, copy=False),
        split=split.astype("<U5", copy=False),
        source_index=source_index.astype(np.int64, copy=False),
        latents_classes=latents.astype(np.int64, copy=False),
        raster_group_id=raster_gid.astype(np.int64, copy=False),
    )


def one_hot(c_int: np.ndarray) -> np.ndarray:
    n = c_int.shape[0]
    out = np.zeros((n, int(sum(CONCEPT_GROUP_SIZES))), dtype=np.uint8)
    off = 0
    for j, s in enumerate(CONCEPT_GROUP_SIZES):
        out[np.arange(n), off + c_int[:, j]] = 1
        off += s
    return out


def write_variant(out_dir, variant, payload, extra_meta, split_seed):
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"dsprites_{variant}_32.npz")
    json_path = os.path.join(out_dir, f"dsprites_{variant}_32.json")

    np.savez_compressed(npz_path, **payload)

    # round-trip without allow_pickle: proves no object arrays got in
    with np.load(npz_path, allow_pickle=False) as chk:
        for k in payload:
            if k not in chk:
                raise SystemExit(f"{npz_path}: field {k} missing after round-trip")
            if chk[k].dtype.kind == "O":
                raise SystemExit(f"{npz_path}: field {k} is an object array")

    split = payload["split"]
    c_int = payload["c_int"]
    y = payload["y_task"]
    tup = [f"{int(a)}_{int(b)}" for a, b in c_int]
    tup = np.array(tup)

    meta = dict(
        dataset_version=DATASET_VERSION,
        variant=variant,
        task_definition=TASK_DEFINITIONS[variant],
        n_samples=int(payload["imgs"].shape[0]),
        image_shape=[int(payload["imgs"].shape[1]), int(payload["imgs"].shape[2])],
        image_dtype=str(payload["imgs"].dtype),
        image_min=int(payload["imgs"].min()),
        image_max=int(payload["imgs"].max()),
        image_unique_values=[int(v) for v in np.unique(payload["imgs"])],
        concept_names=CONCEPT_NAMES,
        concept_group_names=CONCEPT_GROUP_NAMES,
        concept_group_sizes=CONCEPT_GROUP_SIZES,
        group_bounds=GROUP_BOUNDS,
        n_binary_concepts=int(payload["c_bin"].shape[1]),
        n_concept_groups=len(CONCEPT_GROUP_SIZES),
        n_classes=N_CLASSES[variant],
        split_strategy=SPLIT_STRATEGY,
        split_seed=int(split_seed),
        split_fractions=list(SPLIT_FRACS),
        split_counts={s: int((split == s).sum()) for s in ("train", "val", "test")},
        class_counts_by_split={
            s: {int(k): int(v) for k, v in
                zip(*np.unique(y[split == s], return_counts=True))}
            for s in ("train", "val", "test")
        },
        concept_tuple_counts_by_split={
            s: {str(k): int(v) for k, v in
                zip(*np.unique(tup[split == s], return_counts=True))}
            for s in ("train", "val", "test")
        },
        n_raster_groups=int(np.unique(payload["raster_group_id"]).size),
        downsample_method=DOWNSAMPLE_METHOD,
        downsample_factor=DOWNSAMPLE_FACTOR,
        array_content_fingerprint=array_content_fingerprint(payload),
        npz_sha256=sha256_file(npz_path),
    )
    meta.update(extra_meta)

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=False)
        f.write("\n")
    return npz_path, json_path, meta


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True,
                   help="path to dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    p.add_argument("--out", default="data/dsprites/confirmatory_2027")
    p.add_argument("--variants", nargs="+", default=["compact_c", "compact_a"],
                   choices=["compact_c", "compact_a"])
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p.add_argument("--raw-provenance-note", default="",
                   help="free-text note recorded in the manifest (e.g. when the raw "
                        "file is a verified reconstruction rather than the download)")
    args = p.parse_args(argv)

    raw_path = os.path.abspath(args.raw)
    gen_path = os.path.abspath(__file__)

    print(f"[1/7] loading raw       {raw_path}")
    imgs64, lat, raw_is_full = load_raw(raw_path)
    raw_sha = sha256_file(raw_path)
    print(f"      raw sha256        {raw_sha}")
    print(f"      raw rows          {lat.shape[0]:,}"
          f"{'  (full official set)' if raw_is_full else '  (REDUCED FIXTURE)'}")

    print("[2/7] CME sub-selection  3x6x8x16x16")
    keep = select_rows(lat)
    latents = lat[keep]
    imgs64 = imgs64[keep]
    source_index = raw_row_index(latents)
    if raw_is_full and not np.array_equal(source_index, np.flatnonzero(keep)):
        raise SystemExit("source_index disagrees with the boolean mask on the full raw set")
    print(f"      rows              {source_index.size:,}")

    print("[3/7] downsample         64x64 -> 32x32, max-pool 2x2")
    imgs = max_pool_2x2(imgs64)
    del imgs64

    c_int = np.stack([latents[:, 1], latents[:, 2]], axis=1).astype(np.int64)
    y_c = ((c_int[:, 0] == 2) ^ (c_int[:, 1] > 2)).astype(np.int64)

    print("[4/7] exact-raster grouping")
    gid, n_groups = raster_groups(imgs)
    print(f"      raster groups     {n_groups:,}")

    print("[5/7] ambiguity removal")
    amb = find_ambiguous_groups(gid, c_int, y_c)
    amb_groups = int(amb.sum())
    row_mask = ~amb[gid]
    amb_rows = int((~row_mask).sum())
    print(f"      removed           {amb_groups} groups / {amb_rows} rows")

    imgs = imgs[row_mask]
    c_int = c_int[row_mask]
    y_c = y_c[row_mask]
    latents = latents[row_mask]
    source_index = source_index[row_mask]

    # recompute compacted, content-derived group ids on the surviving rows
    gid, n_groups = raster_groups(imgs)
    print(f"      surviving         {imgs.shape[0]:,} rows / {n_groups:,} raster groups")

    print(f"[6/7] grouped stratified split  seed={args.split_seed}")
    strata = (c_int[:, 0] * 6 + c_int[:, 1]).astype(np.int64)
    split = grouped_stratified_split(gid, strata, args.split_seed)
    for s in ("train", "val", "test"):
        print(f"      {s:<5}             {int((split == s).sum()):,}")

    provenance = dict(
        source_raw_path=raw_path,
        source_raw_sha256=raw_sha,
        source_raw_n_rows=int(lat.shape[0]),
        source_raw_is_full_official_set=bool(raw_is_full),
        source_raw_note=args.raw_provenance_note,
        generator_path=gen_path,
        generator_sha256=sha256_file(gen_path),
        generated_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        numpy_version=np.__version__,
        python_version=sys.version.split()[0],
        subselection={
            "shape": "all 3",
            "scale": "all 6",
            "orientation_raw_indices": [int(v) for v in SEL_ORIENTATION],
            "posX_raw_indices": [int(v) for v in SEL_POSX],
            "posY_raw_indices": [int(v) for v in SEL_POSY],
        },
        n_before_ambiguity_removal=EXPECTED_PRE_AMBIGUITY_N,
        ambiguous_raster_groups_removed=amb_groups,
        ambiguous_raster_rows_removed=amb_rows,
        raw_factor_names=RAW_FACTOR_NAMES,
        raw_factor_sizes=RAW_FACTOR_SIZES,
        latents_classes_columns=RAW_FACTOR_NAMES,
    )

    print("[7/7] writing")
    written = []
    for variant in args.variants:
        y = y_c if variant == "compact_c" else (c_int[:, 0] * 6 + c_int[:, 1]).astype(np.int64)
        payload = build_payload(imgs, c_int, y, split, source_index, latents, gid)
        npz_path, json_path, meta = write_variant(
            args.out, variant, payload, provenance, args.split_seed
        )
        print(f"      {npz_path}  ({os.path.getsize(npz_path)/1e6:.2f} MB)")
        print(f"      {json_path}")
        print(f"      fingerprint       {meta['array_content_fingerprint']}")
        written.append(npz_path)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
