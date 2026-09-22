"""Generate robot dataset v5: leak-free, group-based splits.

v5 keeps v4's end-to-end generation and label design unchanged (identical
images, concepts and labels, same seed) and fixes ONE thing: the train/val/test
split. v4 split by render-row, so the 4 renders of one robot identity (and
near-duplicate neighbours) leaked across splits. v5 splits by *robot identity*
so no identity ever appears in more than one split, while still keeping the
classes balanced.

Why v4's documented fix did not work
------------------------------------
DATASET.md (v4) suggested ``sample(groups=dataset.meta["robot_ids"])``. That is
a no-op: ``meta["robot_ids"]`` is ``catalog_df["id"]`` == the row index
(0..30719), i.e. UNIQUE per row, so "grouping" by it puts every render in its
own group and prevents nothing. The catalog is 4 concatenated repetitions of the
same 7,680 unique feature combinations, so the true identity of row r is

    identity = robot_ids % num_unique_robots        # 0..7679, each seen 4x

v5 groups on that key and stratifies on the label.

Output (self-contained, no dependency on v4/Robot.zip)
------------------------------------------------------
- 30,720 images (7,680 identities x 4 renders), 32x32 grayscale, in
  data/robot/robot_images/  (byte-identical to v4: same pipeline + seed)
- 5 annotated concepts, all causal (weights 5/4/3/2/1, intercept -7.5)
- stochastic labels: P(glorp) = sigmoid(8.4 * score)
- splits: 60/20/20, grouped by identity (no leakage) + stratified by label
- CSVs add a `robot_id` (0..7679) and `render_id` (0..3) column so the split is
  auditable and can be reproduced exactly.

Run from the repo root:
    python data/robot/make_dataset.py

For the confirmation protocol, preserve held-out identities and independently
rerender them instead:
    python data/robot/make_external_test_dataset.py --seed 2027 \
        --out data/robot_external_2027
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parent

SEED = 1014
CONCEPT_COLS = ["head_shape", "body_shape", "has_antennae", "ears_shape", "foot_shape"]

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Self-contained output root containing CSVs and robot_images/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from concept_benchmark.config import RobotBenchmarkConfig
        from concept_benchmark.formula import F, LabelFormula
        from concept_benchmark.synthetic.robot import create_synthetic_dataset
        from concept_benchmark.utils import set_deterministic_seed
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Exact source-dataset regeneration requires the optional "
            "concept-benchmark package. For the confirmatory external render "
            "set, use data/robot/make_external_test_dataset.py instead."
        ) from exc
    formula = LabelFormula(
        score=5 * F("body_shape").round
        + 4 * F("foot_shape").pointy
        + 3 * F("has_antennae").true
        + 2 * F("head_shape").round
        + 1 * F("ears_shape").triangle
        - 7.5,
        temperature=8.4,
        stochastic=True,
    )
    output_root = args.out.expanduser().resolve()
    image_dir = output_root / "robot_images"
    image_path_prefix = "robot_images"
    config = RobotBenchmarkConfig(
        seed=args.seed,
        image_size="medium",
        color_mode="grayscale",
        renders_per_robot=4,
        label_formula=formula,
    )

    set_deterministic_seed(args.seed)
    settings = config.to_dict()
    settings["output_directory"] = image_dir
    settings["draw"] = True

    dataset = create_synthetic_dataset(**settings)

    # --- true robot identity: same feature combo, differing only by render ---
    robot_ids = np.asarray(dataset.meta["robot_ids"])          # 0..30719 (row index)
    num_unique = int(dataset.meta["num_unique_robots"])        # 7680
    identity = robot_ids % num_unique                          # 0..7679, each x renders
    renders_per_robot = len(robot_ids) // num_unique
    y_full = np.asarray(dataset.y).astype(int)

    # sanity: identity really does group the renders (exactly renders_per_robot each)
    counts = np.bincount(identity)
    assert identity.max() == num_unique - 1
    assert set(counts.tolist()) == {renders_per_robot}, counts

    # group split (no identity across splits) + label stratification (balance)
    dataset.sample(
        test_size=0.2,
        val_size=0.2,
        groups=identity,
        stratify=y_full,
        seed=args.seed,
    )

    names = list(dataset.meta["concepts"])
    col_idx = [names.index(c) for c in CONCEPT_COLS]
    fname_to_identity = dict(zip(np.asarray(dataset.inputs), identity))
    fname_to_render = dict(zip(np.asarray(dataset.inputs), robot_ids // num_unique))

    split_identities = {}
    splits = [
        ("train", dataset.train),
        ("val", dataset.validation),
        ("test", dataset.test),
    ]
    for split_name, split in splits:
        filenames = np.asarray(split.inputs)
        C = np.asarray(split.C)[:, col_idx]
        y = np.asarray(split.y).astype(int)
        rid = np.array([fname_to_identity[f] for f in filenames])
        rend = np.array([fname_to_render[f] for f in filenames])
        split_identities[split_name] = set(rid.tolist())

        df = pd.DataFrame(
            {"image_path": [f"{image_path_prefix}/{f}" for f in filenames]}
        )
        df["robot_id"] = rid
        df["render_id"] = rend
        for j, col in enumerate(CONCEPT_COLS):
            df[col] = C[:, j]
        df["label"] = y
        df["class"] = np.where(y == 1, "glorp", "drent")
        df.to_csv(
            output_root / f"robot_images_{split_name}_labels.csv", index=False
        )
        print(f"{split_name}: {len(df)} rows | glorp share = {y.mean():.4f}")

    # --- leakage check: no identity may appear in more than one split ---
    tr, va, te = (
        split_identities["train"],
        split_identities["val"],
        split_identities["test"],
    )
    assert not (tr & va) and not (tr & te) and not (va & te), "identity leakage!"
    assert len(tr | va | te) == num_unique
    print(
        f"leakage check OK: train/val/test identities disjoint, "
        f"cover all {num_unique} identities"
    )

    n_png = len(list(image_dir.glob("*.png")))
    print(f"images rendered in {image_dir}: {n_png}")


if __name__ == "__main__":
    main()
