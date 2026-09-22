# Development verification — 2026-09-14

The formal two-stage run has not been launched. Real dSprites test values were not preprocessed or evaluated by these checks.

## Quality and regression checks

`scripts/check.sh` passed Ruff check/format, ty, Pylint E/F/W, and five pytest tests (29.68 seconds in the final quality run). One pre-existing PyTorch/pynvml deprecation warning remains.

The tests cover:

- Real CUDA joint sampling, preservation of concept/label correlation and invalid measurement codes, independent repetitions, nested shot budgets and deterministic replay.
- Two real training seeds, all nine conditions, validation followed by a validation-data test proxy. Pausing after one condition, pausing again at the validation boundary, resuming into stage two, and comparison against an uninterrupted run produced identical saved sampling observations and summaries.
- Completed condition files retained their timestamps across resume. Changed configurations, modified observation files and mismatched Sequential initialization were rejected.
- Historical source artifacts remained byte-identical throughout the CUDA recovery test.
- A synthetic dataset exercised the actual test reader, including the frozen training permutation, held-out row selection and required protocol seal. Development end-to-end tests reject any call to the real test reader.
- Seed-level aggregation preserves negative effects and uses training seeds as the replication unit.

All five shell scripts also passed `bash -n`.

## Actual tmux lifecycle

Output: `outputs/grouped_shots_final/development_tmux_20260914/`.

Configuration: seeds 0,1; 18 validation images; shots 8,16; two sampling repetitions; CUDA batch 18.

1. The first launch paused at 18/36 conditions after validation, with no `protocol_lock.json` or `test_access.json`; the tmux session exited automatically.
2. A `--resume` launch verified the completed validation, created the protocol seal, evaluated `test_proxy`, and completed 36/36 conditions at `2026-09-14T13:07:46.286908+00:00`.
3. The resumed tmux session also exited automatically. `test_evaluated=false`, `real_test_values_requested=false`, and no real `test/` directory was created.
4. All stage CSVs and PNG/PDF/SVG figures were generated; the intervention figure was visually inspected.

## Default-budget CUDA check

Output: `outputs/grouped_shots_final/development_fullbudget_20260914/`.

Seed 0, Independent normal prediction, all 5,479 validation images, batch 2048, shots 64/128/256/512/1024, and ten repetitions completed on NVIDIA GeForce RTX 4090. Frozen exact probabilities reproduced the historical source within the enforced 2e-6 tolerance; exact label accuracy was 70.7063%. The process then paused after the requested one new condition, before any test access. Total command wall time was 12.171 seconds, including initialization and source verification; this is not a timing measurement for all 90 formal conditions.

Default source preflight verifies the five-seed model sources and CUDA without creating the formal output directory. The user-facing `scripts/launch.sh` command starts the actual validation -> protocol seal -> test sequence.
