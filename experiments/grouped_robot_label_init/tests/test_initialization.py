"""Verify paired initialization, untouched baselines, full runs and exact recovery."""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.grouped_dynamic_vqc.runtime import sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_label_depth.protocol import Config as DepthConfig
from experiments.grouped_robot_label_depth.runner import Experiment as DepthExperiment
from experiments.grouped_robot_label_depth.runner import run_experiment as run_depth
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .. import results
from ..model import verify_initializations
from ..operations import status_table
from ..protocol import CELLS, Config, cell_name, check_output
from ..runner import Experiment, run_experiment
from ..training import Job, Route

pytest_plugins = ["experiments.grouped_robot_label_depth.tests.test_depth"]


def snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file() and p.name != "heartbeat.json"
    }


@pytest.fixture(name="depth_source")
def completed_depth(reference, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "plots", lambda *_: None)
    shared = DepthExperiment(
        DepthConfig(
            reference=str(reference.output),
            head_epochs=5,
            diagnostic_every=2,
            checkpoint_steps=1,
            development=True,
        ),
        tmp_path / "depth_source",
    )
    run_depth(shared)
    return shared


def config_for(source) -> Config:
    return Config(
        reference=str(source.output),
        head_epochs=5,
        diagnostic_every=2,
        checkpoint_steps=1,
        development=True,
    )


def test_full_cuda_comparison_preserves_baselines_and_only_changes_initialization(
    depth_source, tmp_path
):
    before = snapshot(depth_source.output)
    shared = Experiment(config_for(depth_source), tmp_path / "comparison")
    assert shared.reuse_uniform
    assert shared.config.sigma == 0.1 / 30
    verify_initializations(shared.initial, shared.reference, shared.config)
    for index in (0, 1, 2):
        uniform, gaussian, shifted = (
            shared.initial[cell_name(method, index)]
            for method in ("uniform", "eft_gaussian", "eft_readout")
        )
        assert state_hash(uniform["model"]) == state_hash(
            depth_source.initial[f"L5/init_{index}"]["model"]
        )
        assert (
            tree_hash(uniform["rng"])
            == tree_hash(gaussian["rng"])
            == tree_hash(shifted["rng"])
        )
        for name, value in gaussian["model"].items():
            if name.startswith("frontend."):
                assert torch.equal(value, uniform["model"][name])
            if name == "label_head.readout":
                assert torch.equal(value[:1], shifted["model"][name][:1])
                torch.testing.assert_close(
                    shifted["model"][name][1], value[1] + math.pi / 2, atol=0, rtol=0
                )
            else:
                assert torch.equal(value, shifted["model"][name])
        draws = torch.cat(
            [
                v.flatten()
                for k, v in gaussian["model"].items()
                if k.startswith("label_head.")
            ]
        )
        assert draws.numel() == 112 and 0.002 < draws.std().item() < 0.005
    diag = read_json(shared.output / "initial_diagnostics.json")
    assert len(diag["rows"]) == 9 and diag["role"] == "train"
    for row in diag["rows"]:
        assert row["optimizer_updates"] == 0 and row["device"] == "cuda:0"
        assert row["gradient_l2_before_clip"] > 0
        if row["method"] == "eft_gaussian":
            assert row["p_mean"] < 0.001
        if row["method"] == "eft_readout":
            assert 0.48 < row["p_mean"] < 0.52
    run_experiment(shared)
    summary = read_json(shared.output / "summary.json")
    assert summary["completed_training_cells"] == 9
    assert summary["completed_conditions"] == 36
    assert summary["new_adam_updates"] == 60
    assert len(summary["rows"]) == 36 and len(summary["paired_vs_uniform"]) == 8
    assert summary["same_sample_orders_verified"]
    assert not summary["test_read"] and not summary["test_evaluated"]
    assert {row["condition"] for row in summary["rows"]} == {
        "measured",
        "correct_all_five",
    }
    for method, index in CELLS:
        job = Job(shared, method, index)
        checkpoint = load(job.checkpoint_path)
        assert checkpoint["progress"]["global_step"] == 10
        assert {int(v["step"]) for v in checkpoint["optimizer"]["state"].values()} == {
            10
        }
        assert checkpoint["gradient_checks"]["gradient_l2"]["frontend"] is None
        assert len(checkpoint["gradient_checks"]["label_layer_gradient_l2"]) == 5
        if method == "uniform":
            assert not (job.output / "training").exists()
            fresh = load(job.output / "validation/correct_all_five/predictions.pt")
            old = shared.reference.uniform_raw(index, "validation", "correct_all_five")
            torch.testing.assert_close(
                fresh["branch_label_mass"], old["branch_label_mass"], atol=0, rtol=0
            )
    assert snapshot(depth_source.output) == before
    before_completed = snapshot(shared.output)
    run_experiment(Experiment(shared.config, shared.output, resume=True))
    assert snapshot(shared.output) == before_completed
    _, status = status_table(shared.output)
    assert status == "complete"
    path = shared.output / "eft_readout/init_2/validation/measured/predictions.pt"
    path.write_bytes(b"tampered predictions")
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(shared.config, shared.output, resume=True)


def test_mid_batch_resume_matches_continuous_training(depth_source, tmp_path):
    config = config_for(depth_source)
    shared = Experiment(config, tmp_path / "resumed")
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_steps=1)
    job = Job(shared, "eft_gaussian", 0)
    partial = load(job.output / "training/independent/resume.pt")
    assert partial["progress"]["global_step"] == 1
    assert partial["progress"]["offset"] == 32
    shared = Experiment(config, shared.output, resume=True)
    run_experiment(shared)
    continuous = Job(shared, "eft_gaussian", 0)
    continuous.output = tmp_path / "continuous"
    Route(continuous).run()
    actual, expected = load(job.checkpoint_path), load(continuous.checkpoint_path)
    for key in ("model", "optimizer", "progress", "rng"):
        assert tree_hash(actual[key]) == tree_hash(expected[key])
    with pytest.raises(ValueError, match="unchanged"):
        Experiment(replace(config, init_kappa=1.0), shared.output, resume=True)


def test_short_budget_and_provenance_guards(depth_source, tmp_path):
    config = replace(config_for(depth_source), head_epochs=2)
    for path in (
        depth_source.output,
        depth_source.output / "new",
        depth_source.output.parent,
        Path(depth_source.reference.reference.config.dataset) / "new",
    ):
        with pytest.raises(ValueError):
            check_output(config, path)
    for value in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replace(config, init_kappa=value).sources()
    with pytest.raises(ValueError):
        replace(config, development=False).sources()
    shared = Experiment(config, tmp_path / "short")
    assert not shared.reuse_uniform
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_steps=1)
    job = Job(shared, "uniform", 0)
    path = job.output / "training/independent/resume.pt"
    checkpoint = load(path)
    checkpoint["progress"]["order"] = checkpoint["progress"]["order"].flip(0)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="order"):
        Route(job)
