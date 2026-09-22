"""Physical branch semantics, label-only gradients, CUDA recovery and provenance."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from experiments.grouped_dynamic_vqc.model import (
    GroupedDynamicVQC,
    controlled_branches,
    controls_for,
)
from experiments.grouped_dynamic_vqc.runtime import cuda_runtime
from experiments.grouped_dynamic_vqc.tests.test_training import assert_tree_equal

from ..diagnostics import (
    diagnose,
    fixed_control_probabilities,
    leave_one_out_controls,
    paired_metrics,
)
from ..protocol import Config, Experiment, module_hash, read_json
from ..results import verify_diagnostic
from ..runner import RouteRun, verify_complete
from ..train import run_experiment


def small_config() -> Config:
    return Config(
        epochs=2,
        head_epochs=2,
        batch_size=18,
        eval_batch_size=18,
        train_limit=36,
        val_limit=18,
        checkpoint_steps=1,
        shots=16,
    )


def test_exact_constant_control_matches_actual_circuit_and_incoherent_branches():
    cuda_runtime(42)
    model = GroupedDynamicVQC().cuda()
    states = torch.randn(3, 1024, device="cuda", dtype=torch.complex64)
    states /= states.norm(dim=1, keepdim=True)
    with torch.no_grad():
        kernel = model.label_head.one_amplitudes()
        actual = fixed_control_probabilities(states, kernel)
        direct = torch.stack(
            [
                model.from_state(
                    states, torch.full((3, 32), code, device="cuda"), direct=True
                )["label_prob"]
                for code in range(32)
            ],
            dim=1,
        )
        torch.testing.assert_close(actual, direct, atol=2e-6, rtol=2e-6)
        phases = torch.exp(1j * torch.randn(3, 32, 1, device="cuda"))
        changed = (states.reshape(3, 32, 32) * phases).reshape(3, 1024)
        torch.testing.assert_close(
            fixed_control_probabilities(changed, kernel), actual, atol=2e-6, rtol=2e-6
        )
        truth = torch.tensor([[0, 1], [1, 2], [2, 3]], device="cuda")
        p = states.reshape(-1, 32, 32).abs().square().sum(-1)
        for mode in ("measured", "shape", "scale", "both", "zero"):
            branches = controlled_branches(
                states, controls_for(3, states.device, mode, truth)
            )
            torch.testing.assert_close(branches.abs().square().sum(-1), p)


def test_random_control_retains_joint_distribution_including_invalid_codes():
    p = torch.zeros(3, 32)
    p[0, 31], p[1, 6], p[2, 0] = 1, 1, 1
    q = leave_one_out_controls(p)
    torch.testing.assert_close(q.mean(0), p.mean(0))
    assert q[0, 31] == 0 and q[0, 6] == 0.5
    h = torch.arange(96).reshape(3, 32).float() / 100
    actual = (h * q).sum(1)
    expected = torch.stack(
        [
            torch.stack([(h[i] * p[j]).sum() for j in range(3) if j != i]).mean()
            for i in range(3)
        ]
    )
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="two images"):
        leave_one_out_controls(p[:1])


def test_paired_transition_metrics_keep_threshold_and_empty_groups():
    baseline = torch.tensor([0.49, 0.8, 0.2, 0.7])
    changed = torch.tensor([0.5, 0.1, 0.4, 0.9])
    labels = torch.tensor([1.0, 1.0, 0.0, 1.0])
    result = paired_metrics(changed, baseline, labels)
    assert result["wrong_to_right_count"] == result["right_to_wrong_count"] == 1
    assert result["delta_accuracy"] == 0 and result["prediction_flip_fraction"] == 0.5
    assert result["mean_true_label_probability_change"] < 0
    assert paired_metrics(changed[:0], baseline[:0], labels[:0]) == {"n_samples": 0}


def test_standard_gradients_are_independent_of_concept_targets(tmp_path):
    shared = Experiment(small_config(), tmp_path / "one")
    first = RouteRun(shared, "standard")
    indices = torch.arange(18, device="cuda")
    loss1, nll1 = first.train_step(indices)
    other = Experiment(small_config(), tmp_path / "two")
    other.data["train"]["concepts"][:, 0] = (
        other.data["train"]["concepts"][:, 0] + 1
    ) % 3
    second = RouteRun(other, "standard")
    loss2, nll2 = second.train_step(indices)
    assert loss1 == loss2 and nll1 != nll2
    assert_tree_equal(first.model.state_dict(), second.model.state_dict())
    assert_tree_equal(first.optimizer.state_dict(), second.optimizer.state_dict())
    assert first.gradient_checks["input_device"].startswith("cuda")


@pytest.mark.parametrize(
    "route", ["standard", "head_true", "head_zero", "head_measured"]
)
def test_cuda_resume_matches_uninterrupted_at_mid_epoch_and_before_validation(
    tmp_path, monkeypatch, route
):
    config = small_config()
    full_shared = Experiment(config, tmp_path / "full")
    full = RouteRun(full_shared, route)
    full.run()
    resumed_shared = Experiment(config, tmp_path / "resumed")
    paused = RouteRun(resumed_shared, route)
    assert paused.run(max_steps=1)["status"] == "paused"
    assert paused.progress["offset"] == 18
    resumed_shared = Experiment(config, tmp_path / "resumed", resume=True)
    resumed = RouteRun(resumed_shared, route, resume=True)

    def crash(_count):
        raise RuntimeError("crash before validation")

    monkeypatch.setattr(resumed, "finish_epoch", crash)
    with pytest.raises(RuntimeError, match="before validation"):
        resumed.run()
    resumed = RouteRun(resumed_shared, route, resume=True)
    resumed.run()
    first = torch.load(
        full.output / "resume.pt", weights_only=False, map_location="cpu"
    )
    second = torch.load(
        resumed.output / "resume.pt", weights_only=False, map_location="cpu"
    )
    for key in ("model", "optimizer", "progress", "rng"):
        assert_tree_equal(first[key], second[key])
    assert verify_complete(resumed_shared, route)["global_step"] == 4
    if route != "standard":
        assert (
            module_hash(resumed.model.state_dict(), "frontend")
            == resumed_shared.pairing["source_frontend_sha256"]
        )
        assert resumed.gradient_checks["gradient_l2"]["frontend"] is None
        assert resumed.gradient_checks["gradient_l2"]["label_head"] > 0
    with (resumed.output / "endpoint.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_complete(resumed_shared, route)


def test_full_pipeline_exact_diagnostics_and_idempotent_resume(tmp_path):
    shared = Experiment(small_config(), tmp_path)
    result = run_experiment(shared)
    assert result["status"] == "complete" and not result["test_evaluated"]
    assert len(result["head_matrix"]) == 9
    assert not result["pairing"]["standard_matches_historical_budget"]
    model = shared.make_model(shared.load("joint_final.pt")["model"])
    report, raw = diagnose(model, shared.data["val"], 18, 16, 0)
    assert report["random_control"]["mean_code_frequency_error"] < 1e-6
    direct = model(shared.data["val"]["angles"])
    torch.testing.assert_close(
        raw["label_probabilities"]["measured"], direct["label_prob"].detach().cpu()
    )
    assert (
        sum(
            report["groups"][f"shape_{s}_scale_{c}"]["measured"]["n_samples"]
            for s in range(3)
            for c in range(6)
        )
        == 18
    )
    again = Experiment(small_config(), tmp_path, resume=True)
    assert run_experiment(again)["status"] == "complete"
    assert read_json(tmp_path / "heartbeat.json")["status"] == "complete"
    with (tmp_path / "diagnostics/joint.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Diagnostic artifact"):
        verify_diagnostic(again, "joint", tmp_path / "joint_final.pt")


def test_default_pairing_and_shared_artifact_resume_guard(tmp_path):
    shared = Experiment(Config(), tmp_path)
    assert shared.reuse_measured_head and shared.standard_paired
    assert shared.routes == ("standard", "head_true", "head_zero")
    assert len(shared.data["train"]["angles"]) == 25593
    assert len(shared.data["val"]["angles"]) == 5479
    with pytest.raises(ValueError, match="identical config"):
        Experiment(replace(Config(), head_epochs=1), tmp_path, resume=True)
    with (tmp_path / "source_frontend.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Shared artifact"):
        Experiment(Config(), tmp_path, resume=True)
