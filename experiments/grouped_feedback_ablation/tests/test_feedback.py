"""Scientific controls, physical sampling, historical reuse and CUDA recovery."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from experiments.grouped_dynamic_vqc.model import GroupedDynamicVQC, controls_for
from experiments.grouped_dynamic_vqc.runtime import cuda_runtime
from experiments.grouped_dynamic_vqc.tests.test_training import assert_tree_equal
from experiments.grouped_vqc_training_modes.protocol import (
    ExperimentConfig,
    SharedExperiment,
    state_hash,
)
from experiments.grouped_vqc_training_modes.runner import RouteRun as HistoricalRun

from ..evaluation import evaluate, forward_control
from ..protocol import Config, Experiment, cells, load_checkpoint
from ..results import evaluate_cell, statistics_for
from ..runner import RouteRun, import_historical, verify_complete
from ..train import run_experiment


def small_config() -> Config:
    return Config(
        seeds="0",
        joint_epochs=2,
        concept_epochs=2,
        head_epochs=2,
        train_limit=36,
        val_limit=18,
        batch_size=18,
        eval_batch_size=18,
        shots=32,
        checkpoint_steps=1,
    )


def test_finite_shots_use_zero_controls_and_keep_invalid_measurement_records():
    cuda_runtime(43)
    model = GroupedDynamicVQC().cuda()
    states = torch.zeros(32, 32, 32, dtype=torch.complex64, device="cuda")
    states[torch.arange(32), torch.arange(32), 0] = 1
    states = states.flatten(1)
    data = {
        "angles": torch.zeros(32, 10, 4, device="cuda"),
        "concepts": torch.zeros(32, 2, dtype=torch.long, device="cuda"),
        "labels": torch.zeros(32, device="cuda"),
        "source_index": torch.arange(32, device="cuda"),
    }
    outputs = {}
    for mode in ("measured", "zero"):
        metrics, raw = evaluate(
            model, data, 32, mode, shots=32768, seed=17, cached_states=states
        )
        direct = model.from_state(
            states, controls_for(32, states.device, mode), direct=True
        )
        torch.testing.assert_close(
            raw["label_probabilities"], direct["label_prob"].cpu()
        )
        assert (
            raw["shot_label_probabilities"] - raw["label_probabilities"]
        ).abs().max() < 0.02
        assert raw["concept_probabilities"][31, 31] == 1
        assert metrics["finite_shots"]["control_mode"] == mode
        outputs[mode] = raw
    assert torch.equal(outputs["zero"]["concept_probabilities"], torch.eye(32))
    assert (
        outputs["zero"]["label_probabilities"]
        - outputs["measured"]["label_probabilities"]
    ).abs().max() > 0.05


@pytest.mark.parametrize("mode", ["measured", "zero"])
def test_branch_kernel_matches_direct_quantum_evolution_and_gradients(mode):
    cuda_runtime(44)
    model = GroupedDynamicVQC().cuda()
    angles = torch.randn(2, 10, 4, device="cuda")
    states = model.frontend(angles)
    fast = forward_control(model, states, mode)
    direct = model.from_state(states, controls_for(2, states.device, mode), direct=True)
    for key in fast:
        torch.testing.assert_close(fast[key], direct[key], atol=2e-6, rtol=2e-6)
    a = torch.autograd.grad(
        fast["label_prob"].sum(), tuple(model.parameters()), retain_graph=True
    )
    b = torch.autograd.grad(direct["label_prob"].sum(), tuple(model.parameters()))
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-6)
        assert x.is_cuda and torch.isfinite(x).all()


def test_joint_feedback_matches_historical_training_step(tmp_path):
    shared = Experiment(small_config(), tmp_path / "new")
    current = RouteRun(shared, "joint/seed0/feedback")
    proxy = SimpleNamespace(
        config=ExperimentConfig(**shared.old),
        output=tmp_path / "old",
        initial=shared.initial_for("joint/seed0/feedback"),
        data=shared.data,
        runtime=shared.runtime,
        manifest_hash=shared.manifest_hash,
    )
    original = HistoricalRun(cast(SharedExperiment, proxy), "joint")
    indices = torch.arange(18, device="cuda")
    assert current.train_step(indices) == original.train_step(indices)
    assert_tree_equal(current.model.state_dict(), original.model.state_dict())
    assert_tree_equal(current.optimizer.state_dict(), original.optimizer.state_dict())


@pytest.mark.parametrize("cell", cells(small_config()))
def test_each_phase_resumes_mid_epoch_exactly(tmp_path, cell):
    config = small_config()
    uninterrupted = Experiment(config, tmp_path / "full")
    interrupted = Experiment(config, tmp_path / "resumed")
    if cell.startswith("sequential") and not cell.endswith("concept"):
        for shared in (uninterrupted, interrupted):
            RouteRun(shared, "sequential/seed0/concept").run()
    full = RouteRun(uninterrupted, cell)
    full.run()
    paused = RouteRun(interrupted, cell)
    assert paused.run(max_steps=1)["status"] == "paused"
    assert paused.progress["offset"] == 18
    del paused
    again = Experiment(config, interrupted.output, resume=True)
    resumed = RouteRun(again, cell)
    resumed.run()
    assert_tree_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_tree_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_tree_equal(full.progress, resumed.progress)
    verify_complete(again, cell)
    gradients = resumed.gradient_checks["gradient_l2"]
    assert (
        gradients["frontend"] is None
        if resumed.phase == "label"
        else gradients["frontend"] > 0
    )
    assert (
        gradients["label_head"] is None
        if resumed.phase == "concept"
        else gradients["label_head"] > 0
    )
    with (again.output / cell / "endpoint.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_complete(again, cell)


def test_full_two_seed_suite_pairing_stages_and_evaluation_locks(tmp_path):
    config = replace(
        small_config(), seeds="0,1", concept_epochs=1, head_epochs=1, joint_epochs=1
    )
    shared = Experiment(config, tmp_path)
    joint = run_experiment(shared, stage="joint")
    assert (
        joint["status"] == "partial" and joint["sequential_statistics"]["n_seeds"] == 0
    )
    result = run_experiment(Experiment(config, tmp_path, resume=True), stage="all")
    assert result["status"] == "complete" and not result["test_evaluated"]
    assert result["sequential_statistics"]["n_seeds"] == 2
    assert len(result["paired_results"]) == 3 and len(result["joint_pilot"]) == 1
    for row in result["paired_results"]:
        assert row["initialization_equal"] and row["sample_orders_equal"]
        if row["mode"] == "sequential":
            assert (
                row["frontend_equal"] and row["max_concept_probability_difference"] == 0
            )
    a, b = (load_checkpoint(tmp_path / f"initializations/seed{s}.pt") for s in (0, 1))
    assert state_hash(a["model"]) != state_hash(b["model"])
    assert (
        run_experiment(Experiment(config, tmp_path, resume=True))["status"]
        == "complete"
    )
    path = tmp_path / "joint/seed0/no_feedback/predictions.pt"
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="Evaluation artifact changed"):
        evaluate_cell(shared, "joint/seed0/no_feedback")


def test_default_reuse_full_budget_cuda_step_and_config_guards(tmp_path):
    shared = Experiment(Config(), tmp_path)
    assert len(shared.manifest["reuse"]) == 4
    assert "joint/seed0/no_feedback" not in shared.manifest["reuse"]
    for cell in shared.manifest["reuse"]:
        import_historical(shared, cell)
        verify_complete(shared, cell)
    assert len(shared.data["train"]["angles"]) == 25593
    assert len(shared.data["val"]["angles"]) == 5479
    model = RouteRun(shared, "joint/seed0/no_feedback")
    model.train_step(torch.arange(1024, device="cuda"))
    assert sum(p.numel() for p in model.model.frontend.parameters()) == 240
    assert sum(p.numel() for p in model.model.label_head.parameters()) == 24
    assert model.gradient_checks["gradient_l2"]["frontend"] > 0
    assert model.gradient_checks["gradient_l2"]["label_head"] > 0
    with pytest.raises(ValueError, match="identical config"):
        Experiment(replace(Config(), joint_epochs=1), tmp_path, resume=True)
    with (tmp_path / "initializations/seed1.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="Shared artifact changed"):
        Experiment(Config(), tmp_path, resume=True)


def test_seed_statistics_are_paired_and_single_seed_std_is_not_zero():
    def row(seed, a, b):
        return {
            "seed": seed,
            "feedback_accuracy": a,
            "no_feedback_accuracy": b,
            "delta_accuracy_pp": 100 * (a - b),
            "feedback_shot_accuracy": a,
            "no_feedback_shot_accuracy": b,
            "delta_shot_accuracy_pp": 100 * (a - b),
        }

    one = statistics_for([row(0, 0.8, 0.7)])
    assert one["delta_accuracy_pp"]["sample_std"] is None
    two = statistics_for([row(0, 0.8, 0.7), row(1, 0.6, 0.7)])
    assert two["delta_accuracy_pp"]["mean"] == pytest.approx(0)
    assert two["delta_accuracy_pp"]["sample_std"] == pytest.approx(2**0.5 * 10)
    assert two["n_seeds"] == 2


@pytest.mark.parametrize("seeds", ["0,0", "-1", "", "abc"])
def test_invalid_seed_lists_rejected(seeds):
    with pytest.raises(ValueError):
        replace(Config(), seeds=seeds).validate()
