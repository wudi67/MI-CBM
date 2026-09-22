# MI-CBM: Measurement-Induced Concept Bottleneck Models

Code and data for the paper **"Measurement-Induced Concept Bottlenecks: Quantum
Ante-Hoc Interpretability for Variational Classifiers"** (submitted to ICASSP 2027).

MI-CBM is a concept bottleneck model realised entirely within a variational
quantum circuit, with no trainable classical parameters:

- **Concept qubits C** are measured mid-circuit. The classical measurement
  outcome, the *record* **m**, is supervised with the ground-truth concept code.
- **Classical feedforward:** each record bit *m_j* controls an *X* gate on its
  paired **working qubit** in W.
- A star-topology **label circuit** acts on W and a **prediction qubit P**,
  whose measurement gives the label.
- A **test-time intervention** replaces bits of the record with the true
  concepts before the feedforward; the measured branch, its Born weight and the
  working-qubit state are left unchanged.

All circuits are simulated exactly with TorchQuantum: all 2^n_c records are
enumerated, so training backpropagates through exact Born probabilities and no
measurement outcome is sampled.

## Repository layout

```
MI-CBM/
├── experiments/
│   ├── grouped_dynamic_vqc/   # MI-CBM model, data pipeline, losses, evaluation, training loop
│   ├── dynamic_vqc/           # Robot image loading utilities
│   └── grouped_*/             # one package per experiment stage (see below)
├── FusionModel.py             # encoder layer: data re-uploading, U3, ring of controlled-U3
├── robot_dataset_schema.py    # Robot dataset schema and file checks
├── torchquantum/              # vendored TorchQuantum 0.2.0 (MIT License)
├── data/
│   ├── dsprites/confirmatory_2027/   # dSprites task used in the paper
│   └── robot/                        # Robot Classification benchmark
├── scripts/                   # dSprites generation and audit
├── analysis/                  # reproduce Tables 1 and 3 and Figs. 2 and 3
└── tests/
```

The model is `GroupedDynamicVQC` in `experiments/grouped_dynamic_vqc/model.py`.
Each experiment package contains `scripts/{launch,run,status,check}.sh`;
`bash experiments/<package>/scripts/run.sh --help` lists the options of a stage.

## Installation

Requirements: Python >= 3.10, a CUDA GPU (training refuses to fall back to the
CPU) and `tmux` (the launch scripts run each stage in a tmux session).

```bash
pip install -r requirements.txt
```

The shell scripts take Python and tmux from `$VQC_PREFIX/bin`. Point
`VQC_PREFIX` at your environment, for example a conda environment in which
tmux is also installed:

```bash
export VQC_PREFIX=/path/to/your/env      # default: /root/miniforge3/envs/VQC
```

Run every command from the repository root.

## Data

### dSprites

`data/dsprites/confirmatory_2027/` holds the task used in the paper: 36,549
binary 32×32 images from dSprites with concepts shape (3 values) and scale
(6 values), and the label `(shape == heart) XOR (scale > 2)` (`compact_c`).
The images are all shapes and scales, every fifth orientation and every second
x- and y-position of dSprites, max-pooled from 64×64 to 32×32; images whose
pooled raster occurs with different concept values are removed, and identical
images never cross the 70/15/15 split. `compact_a` has the same images with an
18-class label and is needed only by the audit.

Before training, run the audit. It writes the admission file that the training
code checks:

```bash
python scripts/audit_dsprites_confirmatory_2027.py \
    --data-dir data/dsprites/confirmatory_2027 \
    --out outputs/dsprites_confirmatory_2027_admission.json
```

To regenerate the data from the official dSprites file
(`dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`, from
<https://github.com/google-deepmind/dsprites-dataset>):

```bash
python scripts/make_dsprites_confirmatory.py \
    --raw /path/to/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
    --out data/dsprites/confirmatory_2027
```

### Robot

`data/robot/` holds 30,720 grayscale 32×32 images of 7,680 robots, each
rendered four times, and one CSV per split. Each image has five binary
concepts: `head_shape` (square/round), `body_shape` (square/round),
`has_antennae` (no/yes), `ears_shape` (square/triangle) and `foot_shape`
(flat/pointy). The label is drawn with probability
`sigmoid(8.4 * score)`, where
`score = 5*body + 4*foot + 3*antennae + 2*head + ears - 7.5`, so it is
nearly deterministic. The split (18,432 / 6,144 / 6,144) is grouped by robot
identity (`robot_id`) and stratified by label. The data were generated with the
Robot Classification benchmark of Skirzynski et al.
(<https://github.com/ustunb/concept-benchmark>); `python data/robot/make_dataset.py`
regenerates them and needs the `concept_benchmark` package.

## Reproducing the paper

Results are written to `outputs/`, which is not part of this repository. Each
stage records hashes of its inputs and reads the results of earlier stages from
their default paths, so run the stages in the order below. `status.sh` shows
the progress of a stage, and `launch.sh --resume` continues an interrupted run.

### dSprites

| # | Stage | Command |
|---|---|---|
| 0 | Data audit | see [Data](#dsprites) |
| 1 | Joint and Sequential, seed 0 | `bash experiments/grouped_vqc_training_modes/scripts/launch.sh --session grouped_modes_l4_seed0 --out outputs/grouped_vqc_training_modes/dsprites_l4_seed0` |
| 2 | Standard and label-circuit diagnostics, seed 0 | `bash experiments/grouped_control_diagnostics/scripts/launch.sh` |
| 3 | Sequential (five seeds) and Joint (seed 0), with and without feedforward | `bash experiments/grouped_feedback_ablation/scripts/launch.sh` |
| 4 | Interventions on Sequential | `bash experiments/grouped_sequential_intervention/scripts/launch.sh` |
| 5 | Independent, five seeds, with interventions | `bash experiments/grouped_independent_intervention/scripts/launch.sh` |
| 6 | Independent with and without feedforward | `bash experiments/grouped_independent_feedback_ablation/scripts/launch.sh` |
| 7 | Test-set evaluation | `bash experiments/grouped_shots_final/scripts/launch.sh` |
| 8 | Standard and Joint for all seeds; four-regime summary | `bash experiments/grouped_four_modes/scripts/launch.sh` |

### Robot

| # | Stage | Command |
|---|---|---|
| 1 | Independent with and without feedforward, seed 0 | `bash experiments/grouped_robot_pilot/scripts/launch.sh` |
| 2 | Longer concept and label training, seed 0 | `bash experiments/grouped_robot_continuation/scripts/launch.sh` |
| 3 | Longer label training, seed 0 | `bash experiments/grouped_robot_label_continuation/scripts/launch.sh` |
| 4 | One- vs. five-layer label circuit (Table 2) | `bash experiments/grouped_robot_label_depth/scripts/launch.sh --session robot_label_depth` |
| 5 | Label-circuit initialisation | `bash experiments/grouped_robot_label_init/scripts/launch.sh --session robot_label_init` |
| 6 | Encoder initialisation | `bash experiments/grouped_robot_concept_init/scripts/launch.sh --session robot_concept_init` |
| 7 | Independent, five seeds | `bash experiments/grouped_robot_independent/scripts/launch.sh` |
| 8 | Sequential, five seeds | `bash experiments/grouped_robot_sequential/scripts/launch.sh` |
| 9 | Standard and Joint with and without feedforward, five seeds | `bash experiments/grouped_robot_four_modes/scripts/launch.sh` |
| 10 | Test-set evaluation | `bash experiments/grouped_robot_shots_final/scripts/launch.sh` |
| 11 | Interventions on 0–5 concepts | `bash experiments/grouped_robot_intervention_curve/scripts/launch.sh` |

### Tables and figures

Once both pipelines have finished:

```bash
python -m analysis.tables     # Tables 1 and 3
python -m analysis.figures    # Figs. 2 and 3, saved to figures/
```

Table 2 is written by Robot stage 4 to
`outputs/grouped_robot_label_depth/robot_v5_l4_seed0/summary.md`.

| Paper | Source in `outputs/` |
|---|---|
| Table 1, Table 3 | `grouped_four_modes/dsprites_l4_five_seeds/test/`, `grouped_robot_shots_final/robot_v5_l4_head5_five_seeds/test/` |
| Table 2 | `grouped_robot_label_depth/robot_v5_l4_seed0/` |
| Fig. 2 | `grouped_four_modes/dsprites_l4_five_seeds/test/correction_count_results.csv`, `grouped_robot_intervention_curve/robot_v5_l4_head5_five_seeds/curve_summary.csv` |
| Fig. 3 | per-seed `joint.pt` files in the two `test/` directories above |

## Third-party components

- `torchquantum/`: TorchQuantum 0.2.0, MIT License (`torchquantum/LICENSE`).
- dSprites: Apache License 2.0, <https://github.com/google-deepmind/dsprites-dataset>.
- Robot Classification benchmark: MIT License, <https://github.com/ustunb/concept-benchmark>.

## Citation

A BibTeX entry will be added once the paper is published.
