"""Real CUDA circuits: loss/control semantics, paired budgets and exact recovery."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

import pytest
import torch
import torch.nn.functional as F

from experiments.grouped_dynamic_vqc.runtime import atomic_checkpoint, sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_independent.model import make_model
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_robot_sequential import results as sequential_results
from experiments.grouped_robot_sequential.protocol import Config as SequentialConfig
from experiments.grouped_robot_sequential.runner import (
    Experiment as SequentialExperiment,
)
from experiments.grouped_robot_sequential.runner import run_experiment as run_sequential
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .. import results
from ..protocol import CELLS, Config, check_output
from ..runner import Experiment, run_experiment
from ..training import Job, Route

pytest_plugins = ["experiments.grouped_robot_sequential.tests.test_sequential"]


@pytest.fixture(name="reference")
def completed_sequential(source_run, tmp_path, monkeypatch):
    monkeypatch.setattr(sequential_results, "plots", lambda *_: None)
    config = SequentialConfig(
        reference=str(source_run.output),
        seeds="0,1",
        head_epochs=2,
        checkpoint_steps=1,
        development=True,
    )
    shared = SequentialExperiment(config, tmp_path / "source_sequential")
    run_sequential(shared)
    return shared


def config_for(reference) -> Config:
    return Config(
        reference=str(reference.output),
        seeds="0,1",
        epochs=4,
        checkpoint_steps=1,
        development=True,
    )


def snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_cuda_pipeline_metrics_and_read_only_references(reference, tmp_path):
    before = snapshot(reference.output)
    before_independent = snapshot(reference.reference.output)
    shared = Experiment(config_for(reference), tmp_path / "full")
    run_experiment(shared)
    summary = read_json(shared.output / "summary.json")
    assert summary["completed_training_cells"] == 6
    assert summary["completed_conditions"] == 36
    assert summary["reused_evaluations"] == 20
    assert summary["new_adam_updates"] == 48
    assert not summary["test_read"] and not summary["test_evaluated"]
    assert (shared.output / "validation_results.pdf").stat().st_size > 1000
    dataset = Path(shared.pilot_config.dataset)
    assert not (dataset / "robot_images_test_labels.csv").exists()
    for seed in shared.config.seed_values:
        initial = shared.initial[str(seed)]
        assert tree_hash(initial) == tree_hash(
            shared.reference.reference.initial[str(seed)]
        )
        assert state_hash(initial["model"]) != state_hash(
            shared.reference.initial[str(seed)]["model"]
        )
        for cell in CELLS:
            saved = load(Job(shared, seed, cell).checkpoint_path)
            assert saved["initial_model_sha256"] == state_hash(initial["model"])
            assert saved["gradient_checks"]["active_parameters"] == 352
            assert saved["gradient_checks"]["device"] == "cuda:0"
            assert all(v > 0 for v in saved["gradient_checks"]["gradient_l2"].values())
            assert saved["progress"]["global_step"] == 8
            assert saved["training_control"] == (
                "zero" if cell == "joint_no_feedback" else "measured"
            )
            assert saved["concept_weight"] == (0 if cell == "standard" else 1)
        a = load(
            shared.output / f"seed_{seed}/validation/joint/measured/predictions.pt"
        )
        b = load(
            shared.output
            / f"seed_{seed}/validation/joint/correct_all_five/predictions.pt"
        )
        assert torch.equal(a["concept_probabilities"], b["concept_probabilities"])
        assert not torch.equal(a["branch_label_mass"], b["branch_label_mass"])
    for row in summary["paired_rows"]:
        assert row["accuracy_gain_pp"] == 100 * (
            row["left_accuracy"] - row["right_accuracy"]
        )
    for aggregate in summary["aggregates"]:
        selected = [
            r[aggregate["metric"]]
            for r in summary["rows"]
            if (r["role"], r["cell"], r["condition"])
            == (aggregate["role"], aggregate["cell"], aggregate["condition"])
        ]
        assert aggregate["mean"] == mean(selected)
        assert aggregate["sample_std"] == stdev(selected)
    for row in summary["rows"]:
        if row["cell"] == "standard":
            assert row["mean_bit_accuracy"] is None
            assert not row["concept_supervised"]
            assert row["condition"] == "measured"
    assert snapshot(reference.output) == before
    assert snapshot(reference.reference.output) == before_independent
    assert not list(shared.output.glob("seed_*/training/independent/endpoint.pt"))
    assert not list(shared.output.glob("seed_*/training/sequential/endpoint.pt"))


def test_training_losses_and_measured_vs_zero_control(reference, tmp_path):
    shared = Experiment(config_for(reference), tmp_path / "semantics")
    indices = epoch_order(64, 0, 1)[:8].cuda()
    data = shared.data["train"]
    concept_codes = (
        data["concepts"][indices].long() * torch.tensor([16, 8, 4, 2, 1], device="cuda")
    ).sum(1)
    for cell in CELLS:
        job = Job(shared, 0, cell)
        route = Route(job)
        manual = make_model(job.initial_for(cell)["model"])
        optimizer = torch.optim.Adam(manual.parameters(), lr=job.config.learning_rate)
        states = manual.frontend(data["angles"][indices])
        records = torch.arange(32, device="cuda").expand(len(indices), -1)
        if cell == "joint_no_feedback":
            records = torch.zeros_like(records)
        output = manual.from_state(states, records)
        # Independent direct branch evolution agrees with the optimized backend.
        direct = manual.from_state(states, records, direct=True)
        torch.testing.assert_close(
            output["label_prob"], direct["label_prob"], atol=2e-6, rtol=2e-6
        )
        label = F.binary_cross_entropy(
            output["label_prob"].clamp(1e-7, 1 - 1e-7), data["labels"][indices]
        )
        loss = label
        if cell != "standard":
            nll = (
                -output["concept_probs"]
                .gather(1, concept_codes[:, None])
                .clamp_min(1e-7)
                .log()
                .mean()
            )
            loss = label + nll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            manual.parameters(), job.config.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        assert route.step(indices) == float(loss.detach())
        assert state_hash(route.model.state_dict()) == state_hash(manual.state_dict())
        assert tree_hash(route.optimizer.state_dict()) == tree_hash(
            optimizer.state_dict()
        )
        alternate = Job(shared, 0, cell)
        alternate.output = tmp_path / f"flipped_{cell}"
        alternate.data = {role: dict(value) for role, value in shared.data.items()}
        alternate.data["train"]["concepts"] = 1 - data["concepts"]
        second = Route(alternate)
        alternate_loss = second.step(indices)
        if cell == "standard":
            assert alternate_loss == float(loss.detach())
            assert state_hash(second.model.state_dict()) == state_hash(
                route.model.state_dict()
            )
        else:
            assert state_hash(second.model.state_dict()) != state_hash(
                route.model.state_dict()
            )


def test_resume_exactness_and_provenance_guards(reference, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "plots", lambda *_: None)
    config = config_for(reference)
    shared = Experiment(config, tmp_path / "paused")
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_steps=1)
    path = shared.output / "seed_0/training/standard/resume.pt"
    partial = load(path)
    assert partial["progress"]["offset"] == 32
    corrupted = deepcopy(partial)
    corrupted["progress"]["order"] = corrupted["progress"]["order"].flip(0)
    atomic_checkpoint(path, corrupted)
    with pytest.raises(ValueError, match="sample order"):
        Route(Job(shared, 0, "standard"))
    atomic_checkpoint(path, partial)
    # Pause again in Joint, after Standard has completed.
    with pytest.raises(InterruptedError):
        run_experiment(Experiment(config, shared.output, resume=True), max_steps=8)
    # Complete seed zero/evaluations, then pause within the next seed.
    with pytest.raises(InterruptedError):
        run_experiment(Experiment(config, shared.output, resume=True), max_steps=16)
    predicted = shared.output / "seed_0/validation/joint/measured/predictions.pt"
    timestamp = predicted.stat().st_mtime_ns
    run_experiment(Experiment(config, shared.output, resume=True))
    assert predicted.stat().st_mtime_ns == timestamp
    full = Experiment(config, tmp_path / "continuous")
    run_experiment(full)
    for seed in config.seed_values:
        for cell in CELLS:
            left, right = (
                load(Job(shared, seed, cell).checkpoint_path),
                load(Job(full, seed, cell).checkpoint_path),
            )
            for key in ("model", "optimizer", "progress", "rng", "component_sums"):
                assert tree_hash(left[key]) == tree_hash(right[key]), (seed, cell, key)
    before = {k: v for k, v in snapshot(shared.output).items() if k != "heartbeat.json"}
    run_experiment(Experiment(config, shared.output, resume=True))
    assert {
        k: v for k, v in snapshot(shared.output).items() if k != "heartbeat.json"
    } == before
    with pytest.raises(ValueError, match="budgets"):
        replace(config, epochs=3).sources()
    with pytest.raises(ValueError, match="overlaps"):
        check_output(config, reference.output / "child")
    with pytest.raises(ValueError, match="unchanged"):
        Experiment(replace(config, seeds="0"), shared.output, resume=True)
    with predicted.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(config, shared.output, resume=True)
    source = reference.output / "seed_0/validation/sequential/measured/predictions.pt"
    with source.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(config, tmp_path / "bad_reference")
