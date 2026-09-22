"""Actual CUDA branches, matched comparisons, immutable sources and exact resume."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.grouped_dynamic_vqc.model import GroupedDynamicVQC
from experiments.grouped_dynamic_vqc.runtime import cuda_runtime, sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_label_continuation.protocol import Config as LabelConfig
from experiments.grouped_robot_label_continuation.runner import (
    Experiment as LabelExperiment,
)
from experiments.grouped_robot_label_continuation.runner import (
    run_experiment as run_labels,
)
from experiments.grouped_robot_pilot.model import controls
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .. import results
from ..model import verify_initializations
from ..operations import status_table
from ..protocol import CELLS, Config, cell_name, check_output
from ..runner import Experiment, run_experiment
from ..training import Job, Route

pytest_plugins = [
    "experiments.grouped_robot_label_continuation.tests.test_label_continuation"
]


def snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file() and p.name != "heartbeat.json"
    }


@pytest.fixture(name="reference")
def completed_labels(source, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "plots", lambda *_: None)
    shared = LabelExperiment(
        LabelConfig(
            reference=str(source.output),
            head_epochs=5,
            diagnostic_every=2,
            development=True,
        ),
        tmp_path / "latest_labels",
    )
    run_labels(shared)
    return shared


def config_for(reference) -> Config:
    return Config(
        reference=str(reference.output),
        head_epochs=5,
        diagnostic_every=2,
        checkpoint_steps=1,
        development=True,
    )


@pytest.mark.parametrize("mask", [0, 1, 31])
def test_five_layers_match_direct_branch_evolution_and_gradients(mask):
    cuda_runtime(7)
    model = GroupedDynamicVQC(front_layers=4, label_layers=5).cuda()
    model.frontend.requires_grad_(False)
    states = torch.randn(2, 1024, device="cuda", dtype=torch.complex64)
    states = states / states.norm(dim=1, keepdim=True)
    concepts = torch.tensor([[0, 1, 0, 1, 1], [1, 0, 1, 0, 0]], device="cuda")
    records = controls(concepts, mask=mask)
    fast = model.from_state(states, records)
    fast["label_prob"].square().sum().backward()
    gradients = []
    for parameter in model.label_head.parameters():
        assert parameter.grad is not None
        gradients.append(parameter.grad.clone())
    model.zero_grad(set_to_none=True)
    direct = model.from_state(states, records, direct=True)
    direct["label_prob"].square().sum().backward()
    for key in fast:
        torch.testing.assert_close(fast[key], direct[key], atol=2e-6, rtol=2e-5)
    for parameter, gradient in zip(
        model.label_head.parameters(), gradients, strict=True
    ):
        torch.testing.assert_close(parameter.grad, gradient, atol=2e-6, rtol=2e-4)
        assert parameter.grad is not None and parameter.grad.is_cuda
    assert all(p.grad is None for p in model.frontend.parameters())
    born = states.reshape(2, 32, 32).abs().square().sum(-1)
    torch.testing.assert_close(fast["concept_probs"], born, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(fast["label_prob"], fast["branch_label_mass"].sum(1))


def test_all_six_cells_reuse_a0_and_preserve_sources(reference, tmp_path):
    before = snapshot(reference.output)
    shared = Experiment(config_for(reference), tmp_path / "complete")
    assert shared.reuse_a0
    verify_initializations(shared.initial, shared.reference.frontend_hash)
    assert (
        len({state_hash(shared.initial[cell_name(1, i)]["model"]) for i in range(3)})
        == 3
    )
    run_experiment(shared)
    summary = read_json(shared.output / "summary.json")
    assert len(summary["training"]) == 6 and len(summary["rows"]) == 36
    assert len(summary["aggregates"]) == 12
    assert len(summary["paired_B_minus_A"]) == 6
    assert all(row["n_initializations"] == 3 for row in summary["aggregates"])
    assert sum(row["reused_historical_training"] for row in summary["training"]) == 1
    assert summary["same_sample_orders_verified"]
    assert not summary["test_read"] and not summary["test_evaluated"]
    assert not (shared.output / "L1/init_0/training").exists()
    for depth, index in CELLS:
        job = Job(shared, depth, index)
        checkpoint = load(job.checkpoint_path)
        assert checkpoint["progress"]["global_step"] == 10
        assert checkpoint["epoch_offset"] == 2
        assert {int(v["step"]) for v in checkpoint["optimizer"]["state"].values()} == {
            10
        }
        if not job.reused:
            gradient = checkpoint["gradient_checks"]
            assert gradient["device"] == "cuda:0"
            assert gradient["gradient_l2"]["frontend"] is None
            assert gradient["label_parameters"] == 22 * depth + 2
            assert len(gradient["label_layer_gradient_l2"]) == depth
            assert min(gradient["label_layer_gradient_l2"]) > 0
    raw = load(shared.output / "L1/init_0/validation/measured/predictions.pt")
    original = shared.reference.quantum_raw["validation"]["measured"]
    torch.testing.assert_close(
        raw["branch_label_mass"], original["branch_label_mass"], rtol=0, atol=0
    )
    assert snapshot(reference.output) == before
    before_completed = snapshot(shared.output)
    run_experiment(Experiment(shared.config, shared.output, resume=True))
    assert snapshot(shared.output) == before_completed
    _, status = status_table(shared.output)
    assert status == "complete"
    raw_path = shared.output / "L5/init_2/validation/measured/predictions.pt"
    raw_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(shared.config, shared.output, resume=True)


def test_mid_epoch_resume_matches_continuous_adam_and_rng(reference, tmp_path):
    config = config_for(reference)
    shared = Experiment(config, tmp_path / "resumed")
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_steps=1)
    job = Job(shared, 5, 0)
    partial = load(job.output / "training/independent/resume.pt")
    assert partial["progress"]["global_step"] == 1
    assert partial["progress"]["offset"] == 32
    assert partial["progress"]["completed_epoch"] == 0
    shared = Experiment(config, shared.output, resume=True)
    run_experiment(shared)
    continuous = Job(shared, 5, 0)
    continuous.output = tmp_path / "continuous"
    Route(continuous).run()
    actual, expected = load(job.checkpoint_path), load(continuous.checkpoint_path)
    for key in ("model", "optimizer", "progress", "rng"):
        assert tree_hash(actual[key]) == tree_hash(expected[key])
    with pytest.raises(ValueError, match="unchanged"):
        Experiment(replace(config, head_epochs=6), shared.output, resume=True)


def test_development_budget_and_output_and_resume_order_guards(reference, tmp_path):
    config = replace(config_for(reference), head_epochs=2)
    for path in (
        reference.output,
        reference.output / "new",
        reference.output.parent,
        Path(reference.reference.config.dataset) / "new",
    ):
        with pytest.raises(ValueError):
            check_output(config, path)
    with pytest.raises(ValueError):
        replace(config, development=False).sources()
    shared = Experiment(config, tmp_path / "shortened")
    assert not shared.reuse_a0
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_steps=1)
    job = Job(shared, 1, 0)
    path = job.output / "training/independent/resume.pt"
    checkpoint = load(path)
    checkpoint["progress"]["order"] = checkpoint["progress"]["order"].flip(0)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="order"):
        Route(job)
