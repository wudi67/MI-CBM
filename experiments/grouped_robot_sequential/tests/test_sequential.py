"""Verify the real measured-branch circuit, exact Adam recovery and paired reuse."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from experiments.grouped_dynamic_vqc.runtime import atomic_checkpoint, sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_independent import results as independent_results
from experiments.grouped_robot_independent.model import make_model
from experiments.grouped_robot_independent.protocol import Config as IndependentConfig
from experiments.grouped_robot_independent.runner import (
    Experiment as IndependentExperiment,
)
from experiments.grouped_robot_independent.runner import (
    run_experiment as run_independent,
)
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .. import results
from ..protocol import Config, check_output
from ..runner import Experiment, run_experiment
from ..training import Job, Route

pytest_plugins = ["experiments.grouped_robot_continuation.tests.test_continuation"]


@pytest.fixture(name="source_run")
def completed_independent(baseline, tmp_path, monkeypatch):
    monkeypatch.setattr(independent_results, "plots", lambda *_: None)
    cfg = IndependentConfig(
        reference=str(baseline.output),
        seeds="0,1",
        concept_epochs=2,
        head_epochs=2,
        head_order_offset=2,
        checkpoint_steps=1,
        fresh_all=True,
        development=True,
    )
    source = IndependentExperiment(cfg, tmp_path / "independent")
    run_independent(source)
    return source


def config_for(source) -> Config:
    return Config(
        reference=str(source.output),
        seeds="0,1",
        head_epochs=2,
        checkpoint_steps=1,
        development=True,
    )


def snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_cuda_pipeline_and_control_models_are_read_only(source_run, tmp_path):
    original = snapshot(source_run.output)
    cfg = config_for(source_run)
    shared = Experiment(cfg, tmp_path / "sequential")
    run_experiment(shared)
    summary = read_json(shared.output / "summary.json")
    assert summary["completed_training_cells"] == 2
    assert summary["completed_conditions"] == 20
    assert summary["reused_evaluations"] == 12
    assert summary["new_adam_updates"] == 8
    assert not summary["test_read"] and not summary["test_evaluated"]
    assert not (
        Path(source_run.reference.config.dataset) / "robot_images_test_labels.csv"
    ).exists()
    assert (shared.output / "validation_results.pdf").stat().st_size > 1000
    for seed in cfg.seed_values:
        job = Job(shared, seed)
        saved = load(job.checkpoint_path)
        assert saved["cell"] == "sequential" and saved["training_control"] == "measured"
        assert saved["gradient_checks"]["active_parameters"] == 112
        assert saved["gradient_checks"]["device"] == "cuda:0"
        assert saved["gradient_checks"]["gradient_l2"]["frontend"] is None
        assert saved["gradient_checks"]["gradient_l2"]["label_head"] > 0
        for cell in ("independent", "no_feedback"):
            old = shared.reference.training[seed, cell]
            assert saved["initial_model_sha256"] == old["initial_scope_sha256"]
            assert [r["order_sha256"] for r in saved["progress"]["history"]] == old[
                "history_order_sha256"
            ]
        a = load(
            shared.output / f"seed_{seed}/validation/sequential/measured/predictions.pt"
        )
        b = load(
            shared.output
            / f"seed_{seed}/validation/sequential/correct_all_five/predictions.pt"
        )
        old = load(
            source_run.output
            / f"seed_{seed}/validation/independent/measured/predictions.pt"
        )
        assert torch.equal(a["concept_probabilities"], b["concept_probabilities"])
        assert torch.equal(a["concept_probabilities"], old["concept_probabilities"])
        marker = read_json(
            shared.output
            / f"seed_{seed}/validation/no_feedback/zero/evaluation_reference.json"
        )
        assert marker["reused"] and Path(marker["predictions_path"]).is_relative_to(
            source_run.output
        )
    for row in summary["paired_rows"]:
        assert row["accuracy_gain_pp"] == 100 * (
            row["left_accuracy"] - row["right_accuracy"]
        )
    assert snapshot(source_run.output) == original
    assert not list(shared.output.glob("seed_*/training/independent/endpoint.pt"))
    assert not list(shared.output.glob("seed_*/training/concept/endpoint.pt"))


def test_step_uses_measured_records_and_not_true_concepts(source_run, tmp_path):
    shared = Experiment(config_for(source_run), tmp_path / "semantics")
    job = Job(shared, 0)
    route = Route(job)
    indices = epoch_order(64, 0, 3)[:32].cuda()
    initial = job.initial_for("sequential")
    manual = make_model(initial["model"])
    manual.frontend.requires_grad_(False)
    optimizer = torch.optim.Adam(
        manual.label_head.parameters(), lr=job.config.learning_rate
    )
    states = shared.state_cache["train"][indices]
    # Explicitly map each physical branch to its own measured bit record.
    records = torch.arange(32, device="cuda").expand(len(indices), -1)
    probability = manual.from_state(states, records)["label_prob"]
    loss = F.binary_cross_entropy(
        probability.clamp(1e-7, 1 - 1e-7), job.data["train"]["labels"][indices]
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        manual.label_head.parameters(), job.config.grad_clip, error_if_nonfinite=True
    )
    optimizer.step()
    actual_loss = route.step(indices)
    assert actual_loss == float(loss.detach())
    assert state_hash(route.model.state_dict()) == state_hash(manual.state_dict())
    assert tree_hash(route.optimizer.state_dict()) == tree_hash(optimizer.state_dict())
    alternate = Job(shared, 0)
    alternate.output = tmp_path / "flipped_concepts"
    alternate.data = {role: dict(values) for role, values in shared.data.items()}
    alternate.data["train"]["concepts"] = 1 - alternate.data["train"]["concepts"]
    second = Route(alternate)
    assert second.step(indices) == actual_loss
    assert state_hash(second.model.state_dict()) == state_hash(route.model.state_dict())
    assert tree_hash(second.optimizer.state_dict()) == tree_hash(
        route.optimizer.state_dict()
    )


def test_resume_and_provenance_guards(source_run, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "plots", lambda *_: None)
    cfg = config_for(source_run)
    shared = Experiment(cfg, tmp_path / "paused")
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_steps=1)
    path = shared.output / "seed_0/training/sequential/resume.pt"
    partial = load(path)
    assert partial["progress"]["offset"] == 32
    corrupted = deepcopy(partial)
    corrupted["progress"]["order"] = corrupted["progress"]["order"].flip(0)
    atomic_checkpoint(path, corrupted)
    with pytest.raises(ValueError, match="sample order"):
        Route(Job(shared, 0))
    atomic_checkpoint(path, partial)
    # Finish seed zero and its evaluation, pause within seed one's training.
    with pytest.raises(InterruptedError):
        run_experiment(Experiment(cfg, shared.output, resume=True), max_steps=4)
    predicted = shared.output / "seed_0/validation/sequential/measured/predictions.pt"
    timestamp = predicted.stat().st_mtime_ns
    run_experiment(Experiment(cfg, shared.output, resume=True))
    assert predicted.stat().st_mtime_ns == timestamp
    full = Experiment(cfg, tmp_path / "continuous")
    run_experiment(full)
    for seed in cfg.seed_values:
        left = load(Job(shared, seed).checkpoint_path)
        right = load(Job(full, seed).checkpoint_path)
        for key in ("model", "optimizer", "progress", "rng"):
            assert tree_hash(left[key]) == tree_hash(right[key]), key
    before = {k: v for k, v in snapshot(shared.output).items() if k != "heartbeat.json"}
    run_experiment(Experiment(cfg, shared.output, resume=True))
    assert {
        k: v for k, v in snapshot(shared.output).items() if k != "heartbeat.json"
    } == before
    with pytest.raises(ValueError, match="budgets must match"):
        replace(cfg, head_epochs=3).sources()
    with pytest.raises(ValueError, match="overlaps"):
        check_output(cfg, source_run.output / "child")
    with pytest.raises(ValueError, match="unchanged"):
        Experiment(replace(cfg, seeds="0"), shared.output, resume=True)
    with predicted.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(cfg, shared.output, resume=True)
    with (source_run.output / "seed_0/validation/no_feedback/zero/predictions.pt").open(
        "ab"
    ) as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(cfg, tmp_path / "bad_reference")
