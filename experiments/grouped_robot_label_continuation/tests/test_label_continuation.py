"""CUDA equivalence, correct source selection, resumability and data isolation."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.grouped_dynamic_vqc.runtime import sha256
from experiments.grouped_robot_continuation.protocol import Config as PreviousConfig
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_continuation.runner import (
    Experiment as PreviousExperiment,
)
from experiments.grouped_robot_continuation.runner import run_experiment as run_previous
from experiments.grouped_robot_continuation.training import Job
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import Route as PilotRoute
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .. import results
from ..protocol import CELLS, Config, check_output
from ..runner import Experiment, run_experiment
from ..training import Route

pytest_plugins = ["experiments.grouped_robot_continuation.tests.test_continuation"]


def snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file() and p.name != "heartbeat.json"
    }


@pytest.fixture(name="source")
def completed_source(baseline, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "plots", lambda *_: None)
    cfg = PreviousConfig(
        reference=str(baseline.output),
        concept_epochs=4,
        head_epochs=4,
        development=True,
    )
    shared = PreviousExperiment(cfg, tmp_path / "previous")
    run_previous(shared)
    return shared


def config_for(source) -> Config:
    return Config(
        reference=str(source.output),
        head_epochs=5,
        diagnostic_every=2,
        development=True,
    )


def comparable_progress(checkpoint: dict) -> dict:
    progress = deepcopy(checkpoint["progress"])
    for row in progress["history"]:
        row.pop("validation_true", None)
    return progress


def test_matches_continuous_training_and_uses_long_concept(source, tmp_path):
    before, pilot_before = snapshot(source.output), snapshot(source.reference.output)
    shared = Experiment(config_for(source), tmp_path / "extended")
    run_experiment(shared)
    for cell in CELLS:
        directory = shared.output / "long_label/training" / cell
        endpoint = load(directory / "endpoint.pt")
        imported = load(directory / "imported.pt")
        origin = load(source.output / "long_concept/training" / cell / "endpoint.pt")
        for key in ("model", "optimizer", "progress", "rng", "training_seconds"):
            assert tree_hash(imported[key]) == tree_hash(origin[key])
        assert state_hash(origin["model"]) != state_hash(
            load(source.output / "long_label/training" / cell / "endpoint.pt")["model"]
        )
        assert {int(v["step"]) for v in imported["optimizer"]["state"].values()} == {4}
        assert {int(v["step"]) for v in endpoint["optimizer"]["state"].values()} == {10}
        # Five uninterrupted epochs from the ORIGINAL label initialization, using
        # the already four-epoch frontend and original two-epoch shuffle offset.
        job = Job(shared, "long_label", cell)
        job.output = tmp_path / "continuous"
        PilotRoute(job, cell).run()
        gold = load(job.output / "training" / cell / "endpoint.pt")
        assert state_hash(endpoint["model"]) == state_hash(gold["model"])
        assert tree_hash(endpoint["optimizer"]) == tree_hash(gold["optimizer"])
        assert tree_hash(comparable_progress(endpoint)) == tree_hash(gold["progress"])
        assert tree_hash(endpoint["rng"]) == tree_hash(gold["rng"])
        assert endpoint["gradient_checks"]["device"] == "cuda:0"
        assert endpoint["gradient_checks"]["gradient_l2"]["frontend"] is None
        assert endpoint["gradient_checks"]["gradient_l2"]["label_head"] > 0
        assert endpoint["epoch_offset"] == 2
        for role, condition in (
            ("validation", "correct_all_five" if cell == "independent" else "zero"),
            ("train", "measured" if cell == "independent" else "zero"),
        ):
            left = load(
                shared.output / "baseline" / role / cell / condition / "predictions.pt"
            )
            right = load(
                shared.output
                / "long_label"
                / role
                / cell
                / condition
                / "predictions.pt"
            )
            torch.testing.assert_close(
                left["concept_probabilities"],
                right["concept_probabilities"],
                rtol=0,
                atol=0,
            )
    summary = read_json(shared.output / "summary.json")
    assert summary["new_adam_updates"] == 12
    assert summary["concept_epochs"] == 4
    assert summary["completed_training_cells"] == 2
    assert summary["completed_conditions"] == 22
    assert not summary["test_read"] and not summary["test_evaluated"]
    points = [
        r
        for r in summary["intervention_learning_curve"]
        if r["condition"] == "correct_all_five"
    ]
    assert [r["epoch"] for r in points] == [2, 4, 5]
    assert snapshot(source.output) == before
    assert snapshot(source.reference.output) == pilot_before
    assert not (
        Path(source.reference.config.dataset) / "robot_images_test_labels.csv"
    ).exists()


def test_mid_epoch_resume_diagnostics_and_completed_reuse(source, tmp_path):
    cfg = config_for(source)
    shared = Experiment(cfg, tmp_path / "paused")
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_steps=1)
    checkpoint = load(shared.output / "long_label/training/independent/resume.pt")
    assert checkpoint["progress"]["completed_epoch"] == 2
    assert checkpoint["progress"]["global_step"] == 5
    assert checkpoint["progress"]["offset"] == 32
    shared = Experiment(cfg, shared.output, resume=True)
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_conditions=1)
    raw = shared.output / "long_label/validation/independent/measured/predictions.pt"
    before = raw.stat().st_mtime_ns
    shared = Experiment(cfg, shared.output, resume=True)
    run_experiment(shared)
    assert raw.stat().st_mtime_ns == before
    full = Experiment(cfg, tmp_path / "full")
    run_experiment(full)
    for cell in CELLS:
        relative = Path("long_label/training") / cell / "endpoint.pt"
        left, right = load(shared.output / relative), load(full.output / relative)
        for key in ("model", "optimizer", "progress", "rng"):
            assert tree_hash(left[key]) == tree_hash(right[key])
    before = snapshot(shared.output)
    run_experiment(Experiment(cfg, shared.output, resume=True))
    assert snapshot(shared.output) == before
    with pytest.raises(ValueError, match="unchanged"):
        Experiment(replace(cfg, head_epochs=6), shared.output, resume=True)
    raw.write_bytes(b"tampered predictions")
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(cfg, shared.output, resume=True)


def test_rejects_wrong_resume_order_and_protected_outputs(source, tmp_path):
    cfg = config_for(source)
    for path in (
        source.output,
        source.output / "new",
        source.output.parent,
        source.reference.output / "new",
        Path(source.reference.config.dataset) / "new",
    ):
        with pytest.raises(ValueError):
            check_output(cfg, path)
    with pytest.raises(ValueError):
        replace(cfg, head_epochs=2).sources()
    with pytest.raises(ValueError):
        replace(cfg, development=False).sources()
    shared = Experiment(cfg, tmp_path / "bad_order")
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_steps=1)
    path = shared.output / "long_label/training/independent/resume.pt"
    raw = load(path)
    raw["progress"]["order"] = raw["progress"]["order"].flip(0)
    torch.save(raw, path)
    shared = Experiment(cfg, shared.output, resume=True)
    with pytest.raises(ValueError, match="order"):
        Route(Job(shared, "long_label", "independent"))
