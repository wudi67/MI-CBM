"""Compare resumed Adam trajectories to continuous training on actual CUDA circuits."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.grouped_dynamic_vqc.runtime import sha256
from experiments.grouped_robot_pilot import results as pilot_results
from experiments.grouped_robot_pilot.protocol import Config as PilotConfig
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.runner import Experiment as PilotExperiment
from experiments.grouped_robot_pilot.runner import run_experiment as run_pilot
from experiments.grouped_robot_pilot.training import Route as PilotRoute
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .. import results
from ..protocol import JOBS, Config, check_output, tree_hash
from ..runner import Experiment, run_experiment
from ..training import Job, Route

pytest_plugins = ["experiments.grouped_robot_pilot.tests.test_pilot"]


@pytest.fixture(name="baseline")
def completed_pilot(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(pilot_results, "plots", lambda *_: [])
    monkeypatch.setattr(results, "plots", lambda *_: [])
    config = PilotConfig(
        dataset=str(dataset),
        development=True,
        concept_epochs=2,
        head_epochs=2,
        train_limit=64,
        val_limit=32,
        batch_size=32,
        eval_batch_size=64,
        checkpoint_steps=1,
    )
    shared = PilotExperiment(config, tmp_path / "pilot")
    run_pilot(shared)
    return shared


def config_for(baseline) -> Config:
    return Config(
        reference=str(baseline.output),
        concept_epochs=4,
        head_epochs=4,
        development=True,
    )


def snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file() and p.name != "heartbeat.json"
    }


def assert_same_training(left: dict, right: dict) -> None:
    assert state_hash(left["model"]) == state_hash(right["model"])
    assert tree_hash(left["optimizer"]) == tree_hash(right["optimizer"])
    assert tree_hash(left["progress"]) == tree_hash(right["progress"])


def test_continuation_matches_continuous_adam_and_preserves_reference(
    baseline, tmp_path
):
    before = snapshot(baseline.output)
    cfg = config_for(baseline)
    shared = Experiment(cfg, tmp_path / "continuation")
    run_experiment(shared)
    # Independent continuous concept training from the original random initialization.
    gold = PilotExperiment(
        replace(baseline.config, concept_epochs=4), tmp_path / "gold_concept"
    )
    PilotRoute(gold, "concept").run()
    assert_same_training(
        load(gold.output / "training/concept/endpoint.pt"),
        load(shared.output / "long_concept/training/concept/endpoint.pt"),
    )
    for cell in ("independent", "no_feedback"):
        # The old pilot loop trains four uninterrupted label epochs on the ORIGINAL
        # two-epoch frontend. The new continuation must land on identical Adam states.
        job = Job(shared, "long_label", cell)
        job.output = tmp_path / "gold_label"
        PilotRoute(job, cell).run()
        assert_same_training(
            load(job.output / "training" / cell / "endpoint.pt"),
            load(shared.output / "long_label/training" / cell / "endpoint.pt"),
        )
        source_init = read_json(
            baseline.output / "training" / cell / "initialization.json"
        )
        fresh_init = read_json(
            shared.output / "long_concept/training" / cell / "initialization.json"
        )
        assert fresh_init["fresh_adam"]
        assert fresh_init["head_sha256"] == source_init["head_sha256"]
        old_order = [
            v["order_sha256"]
            for v in read_json(baseline.output / "training" / cell / "history.json")
        ]
        new_order = [
            v["order_sha256"]
            for v in read_json(
                shared.output / "long_concept/training" / cell / "history.json"
            )
        ]
        assert old_order == new_order
    for arm, cell in JOBS:
        checkpoint = load(shared.output / arm / "training" / cell / "endpoint.pt")
        active = "frontend" if cell == "concept" else "label_head"
        frozen = "label_head" if cell == "concept" else "frontend"
        assert checkpoint["gradient_checks"]["device"] == "cuda:0"
        assert checkpoint["gradient_checks"]["gradient_l2"][active] > 0
        assert checkpoint["gradient_checks"]["gradient_l2"][frozen] is None
    summary = read_json(shared.output / "summary.json")
    assert summary["completed_conditions"] == 33
    assert summary["completed_training_cells"] == 5
    assert summary["new_adam_updates"] == 20
    assert not summary["test_read"] and not summary["test_evaluated"]
    assert not (Path(baseline.config.dataset) / "robot_images_test_labels.csv").exists()
    assert snapshot(baseline.output) == before


def test_mid_epoch_resume_and_completed_evaluation_reuse(baseline, tmp_path):
    cfg = config_for(baseline)
    shared = Experiment(cfg, tmp_path / "paused")
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_steps=1)
    partial = load(shared.output / "long_concept/training/concept/resume.pt")
    assert partial["progress"]["completed_epoch"] == 2
    assert partial["progress"]["global_step"] == 5
    assert partial["progress"]["offset"] == 32
    shared = Experiment(cfg, shared.output, resume=True)
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_conditions=1)
    predicted = (
        shared.output / "long_concept/validation/independent/measured/predictions.pt"
    )
    timestamp = predicted.stat().st_mtime_ns
    shared = Experiment(cfg, shared.output, resume=True)
    run_experiment(shared)
    assert predicted.stat().st_mtime_ns == timestamp
    full = Experiment(cfg, tmp_path / "full")
    run_experiment(full)
    for arm, cell in JOBS:
        relative = Path(arm) / "training" / cell / "endpoint.pt"
        assert_same_training(
            load(shared.output / relative), load(full.output / relative)
        )
    for p in shared.output.glob("*/*/*/*/predictions.pt"):
        assert tree_hash(load(p)) == tree_hash(
            load(full.output / p.relative_to(shared.output))
        )
    before = snapshot(shared.output)
    run_experiment(Experiment(cfg, shared.output, resume=True))
    assert before == snapshot(shared.output)
    with pytest.raises(ValueError, match="unchanged"):
        Experiment(replace(cfg, head_epochs=5), shared.output, resume=True)
    with predicted.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(cfg, shared.output, resume=True)


def test_resume_rejects_changed_sample_order_and_unsafe_outputs(baseline, tmp_path):
    cfg = config_for(baseline)
    shared = Experiment(cfg, tmp_path / "order")
    job = Job(shared, "long_concept", "concept")
    route = Route(job)
    original = load(job.origin)
    imported = load(route.output / "imported.pt")
    for key in ("model", "optimizer", "rng", "progress"):
        assert tree_hash(original[key]) == tree_hash(imported[key])
    with pytest.raises(InterruptedError):
        route.run(max_steps=5)
    state = load(route.output / "resume.pt")
    state["progress"]["order"] = state["progress"]["order"].flip(0)
    torch.save(state, route.output / "resume.pt")
    with pytest.raises(ValueError, match="minibatch order"):
        Route(job)
    for path in (
        baseline.output,
        baseline.output / "child",
        baseline.output.parent,
        Path("outputs/grouped_robot_pilot/new"),
        Path("experiments/new"),
    ):
        with pytest.raises(ValueError):
            check_output(cfg, path, baseline.config)
    with pytest.raises(ValueError, match="targets"):
        replace(cfg, concept_epochs=2).validate(baseline.config)
