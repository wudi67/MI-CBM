# dSprites — confirmatory_2027

Regenerated from raw dSprites for the confirmatory Quantum CBM sweep. Replaces
`data/dsprites/dsprites_compact/` (old sample-level split: identical rasters
leaked across splits, and some rasters carried conflicting annotations).
The old directory is untouched.

## Run it on the real raw file

```bash
python scripts/make_dsprites_confirmatory.py \
    --raw /path/to/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
    --out data/dsprites/confirmatory_2027

python scripts/audit_dsprites_confirmatory_2027.py \
    --data-dir data/dsprites/confirmatory_2027 \
    --out outputs/dsprites_confirmatory_2027_admission.json

python -m pytest tests/test_dsprites_confirmatory_2027.py -v
```

The auditor exits non-zero if any check fails. Do not train on data whose
admission JSON does not say `admitted_for_confirmatory_sweep: true`.

## Contents

| file | rows | y |
|---|---|---|
| `dsprites_compact_c_32.npz` | 36,549 | `(shape==heart) XOR (scale>2)` — 2 classes |
| `dsprites_compact_a_32.npz` | 36,549 | `shape*6 + scale` — 18 classes |

`compact_a` is row-identical to `compact_c` — same images, same order, same
concepts, same split — only `y_task` differs, so one `X→C` checkpoint serves
both `C→Y` decoders.

### NPZ fields

| field | shape | dtype | notes |
|---|---|---|---|
| `imgs` | `[N,32,32]` | `uint8` | values strictly `{0,1}`. **Do not divide by 255.** |
| `c_int` | `[N,2]` | `int64` | `[:,0]` shape 0–2, `[:,1]` scale 0–5 |
| `c_bin` | `[N,9]` | `uint8` | shape one-hot(3) ‖ scale one-hot(6) |
| `group_bounds` | `[2,2]` | `int64` | `[[0,3],[3,9]]` |
| `concept_names` | `[9]` | `<U14` | `shape::square` … `scale::5` |
| `concept_group_names` | `[2]` | `<U5` | `shape`, `scale` |
| `concept_group_sizes` | `[2]` | `int64` | `[3,6]` |
| `y_task` | `[N]` | `int64` | |
| `split` | `[N]` | `<U5` | `train` / `val` / `test` |
| `source_index` | `[N]` | `int64` | row in the 737,280-row raw file; unique |
| `latents_classes` | `[N,6]` | `int64` | raw `(color,shape,scale,orientation,posX,posY)` |
| `raster_group_id` | `[N]` | `int64` | exact-raster equivalence class |

No object arrays — everything loads with `allow_pickle=False`.

## Pipeline

1. **Sub-select** all 3 shapes × all 6 scales × orientation `0,5,…,35` ×
   posX `0,2,…,30` × posY `0,2,…,30` = **36,864** rows.
2. **Downsample** 64×64 → 32×32 by **2×2 max-pool** (not average; max-pool keeps
   thin shapes from vanishing and keeps values in `{0,1}` without rescaling).
3. **Group** rows by bit-exact 32×32 raster → 34,753 groups.
4. **Drop ambiguity**: any raster group carrying more than one concept vector or
   more than one label is deleted *entirely* — no arbitrary
   keep-one-annotation. Removed **75 groups / 315 rows**, all squares (2×2
   max-pool makes some square rasters identical across scales). → **36,549 rows**
   in **34,678 groups**.
5. **Split** 70/15/15 with `split_seed=2027`, strategy `grouped_exact_raster`:
   the raster group is atomic, so identical images can never straddle a split.
   Stratified over the full 18 `(shape,scale)` tuples rather than the binary
   label; every stratum lands within 0.1 pp of 70/15/15.

Steps 4 and 5 run *after* downsampling, so ambiguity is judged on the pixels the
model actually sees.

## Split

| split | n | % | y=0 | y=1 | concept tuples |
|---|---|---|---|---|---|
| train | 25,593 | 70.02% | 12,771 | 12,822 | 18/18 |
| val | 5,479 | 14.99% | 2,733 | 2,746 | 18/18 |
| test | 5,477 | 14.99% | 2,733 | 2,744 | 18/18 |

Cross-split exact-raster overlap: train–val **0**, train–test **0**, val–test **0**.

## Provenance

Every manifest records the raw path and SHA256, the generator's own SHA256,
the split seed, generation time, numpy/python versions, the sub-selection grid,
the downsample method, the ambiguity counts, and an
`array_content_fingerprint` — a SHA256 over the *decompressed* array contents
(dtype + shape + bytes, sorted by key), so it is independent of zip
compression and timestamps and will match after any lossless repack.

## Loader note

```python
d = np.load(path, allow_pickle=False)
x = torch.from_numpy(d["imgs"]).float().unsqueeze(1)   # already 0/1 — no /255
```
