"""Verify physical branches, corrected sampling and source-preserving recovery."""

from dataclasses import replace

import pytest
import torch

import torchquantum as tq
import torchquantum.functional as tqf
from experiments.grouped_dynamic_vqc.model import (
    ControlMode,
    GroupedDynamicVQC,
    controls_for,
)
from experiments.grouped_dynamic_vqc.runtime import cuda_runtime, sha256
from experiments.grouped_feedback_ablation.protocol import load_checkpoint, read_json
from experiments.grouped_vqc_training_modes.protocol import state_hash

from ..evaluation import evaluate, forward_control, paired_changes
from ..protocol import MODES, Config, check_output
from ..results import statistics_for
from ..runner import Experiment, run_experiment


def test_semantic_groups_and_invalid_physical_branches():
    truth = torch.tensor([[2, 3]])  # true 10|011, original outcome 01|100 = 12
    expected: dict[ControlMode, int] = {
        "measured": 12,
        "shape": 20,
        "scale": 11,
        "both": 19,
    }
    for mode, value in expected.items():
        controls = controls_for(1, truth.device, mode, truth)
        assert controls.shape == (1, 32)
        assert controls[0, 12] == value
    assert controls_for(1, truth.device, "shape", truth)[0, 31] == 23
    assert controls_for(1, truth.device, "scale", truth)[0, 31] == 27


@pytest.mark.parametrize("mode", MODES)
@torch.no_grad()
def test_against_explicit_projection_normalization_and_conditional_x(mode):
    cuda_runtime(871)
    model = GroupedDynamicVQC().cuda().eval().requires_grad_(False)
    states = model.frontend(torch.randn(2, 10, 4, device="cuda"))
    before_states = states.clone()
    before_model = state_hash(model.state_dict())
    truth = torch.tensor([[2, 3], [1, 5]], device="cuda")
    fast = forward_control(model, states, mode, truth)
    projected = states.reshape(2, 32, 32)
    weights = projected.abs().square().sum(-1)
    normalized = projected / weights.clamp_min(1e-20).sqrt().unsqueeze(-1)
    qdev = tq.QuantumDevice(n_wires=5, bsz=64, device="cuda")
    qdev.set_states(normalized.reshape(64, 32))
    # Apply actual X gates to normalized B states, independently of the gather kernel.
    measured_bits = [
        (torch.arange(32, device="cuda") >> (4 - wire)) & 1 for wire in range(5)
    ]
    for wire in range(5):
        chosen = measured_bits[wire].expand(2, -1)
        if mode in {"shape", "both"} and wire < 2:
            chosen = ((truth[:, :1] >> (1 - wire)) & 1).expand(-1, 32)
        if mode in {"scale", "both"} and wire >= 2:
            chosen = ((truth[:, 1:] >> (4 - wire)) & 1).expand(-1, 32)
        before_gate = qdev.get_states_1d().clone()
        tqf.x(qdev, wires=wire)
        qdev.set_states(
            torch.where(chosen.reshape(-1, 1).bool(), qdev.get_states_1d(), before_gate)
        )
    conditional_y = model.label_head(qdev.get_states_1d()).reshape(2, 32)
    expected_mass = weights * conditional_y
    torch.testing.assert_close(
        fast["branch_label_mass"], expected_mass, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        fast["label_prob"], expected_mass.sum(1), atol=2e-6, rtol=2e-6
    )
    assert torch.equal(states, before_states)
    assert state_hash(model.state_dict()) == before_model
    assert fast["label_prob"].is_cuda


@torch.no_grad()
def test_each_condition_samples_its_own_distribution_including_invalid_m():
    cuda_runtime(872)
    model = GroupedDynamicVQC().cuda().eval()
    # A fixed, sensitive readout of one Shape bit and one Scale bit avoids
    # accidentally testing a nearly constant random classification circuit.
    for parameter in model.label_head.parameters():
        parameter.zero_()
    model.label_head.a[0, 5] = torch.pi / 2
    model.label_head.gamma[0, 0] = 0.8
    model.label_head.gamma[0, 4] = 0.6
    model.label_head.readout[:] = torch.pi / 2
    states = torch.zeros(32, 32, 32, dtype=torch.complex64, device="cuda")
    states[torch.arange(32), torch.arange(32), 0] = 1
    states = states.flatten(1)
    data = {
        "concepts": torch.tensor([[2, 3]], device="cuda").expand(32, -1),
        "labels": torch.zeros(32, device="cuda"),
        "source_index": torch.arange(32, device="cuda"),
    }
    predictions = {}
    for mode in MODES:
        metrics, raw = evaluate(
            model, data, states, mode, batch_size=32, shots=32768, seed=19
        )
        assert torch.equal(raw["concept_probabilities"], torch.eye(32))
        assert metrics["finite_shots"]["control_mode"] == mode
        assert (
            raw["shot_label_probabilities"] - raw["label_probabilities"]
        ).abs().max() < 0.02
        predictions[mode] = raw
    assert (
        predictions["both"]["label_probabilities"]
        - predictions["measured"]["label_probabilities"]
    ).abs().max() > 0.05
    assert predictions["both"]["label_probabilities"].std() < 1e-6
    assert (
        len(
            {
                predictions[m]["shot_label_probabilities"].numpy().tobytes()
                for m in MODES
            }
        )
        == 4
    )


def test_pair_counts_and_training_seed_statistics_keep_negative_effects():
    changes = paired_changes(
        torch.tensor([0.1, 0.9, 0.6, 0.2]),
        torch.tensor([0.8, 0.1, 0.1, 0.2]),
        torch.ones(4),
    )
    assert changes["wrong_to_right_count"] == 1
    assert changes["right_to_wrong_count"] == 2
    assert changes["delta_accuracy_pp"] == -25
    rows = [
        {"seed": seed, "mode": "both", "n_samples": 4, "delta_accuracy_pp": value}
        for seed, value in ((0, 10), (1, -10), (2, 99))
    ]
    stats = statistics_for(rows, [0, 1])["both"]
    assert stats["n_seeds"] == 2
    assert stats["delta_accuracy_pp"]["mean"] == 0
    assert stats["delta_accuracy_pp"]["sample_std"] == pytest.approx(10 * 2**0.5)
    assert statistics_for(rows, [0])["both"]["delta_accuracy_pp"]["sample_std"] is None


def test_two_seed_resume_is_identical_and_corruption_is_rejected(tmp_path):
    config = Config(seeds="0,1", val_limit=18, eval_batch_size=18, shots=32)
    full = Experiment(config, tmp_path / "full")
    reference = read_json(full.output / "reference_lock.json")
    result = run_experiment(full)
    assert result["status"] == "complete" and result["engineering_subset"]
    assert result["complete_seeds"] == [0, 1] and len(result["rows"]) == 8
    paused = Experiment(config, tmp_path / "paused")
    partial = run_experiment(paused, max_conditions=2)
    assert partial["status"] == "partial" and partial["complete_seeds"] == []
    existing = paused.directory(0, "shape") / "predictions.pt"
    timestamp = existing.stat().st_mtime_ns
    resumed = Experiment(config, paused.output, resume=True)
    complete = run_experiment(resumed)
    assert complete["rows"] == result["rows"]
    assert complete["statistics"] == result["statistics"]
    assert existing.stat().st_mtime_ns == timestamp
    for seed in (0, 1):
        for mode in MODES:
            a = load_checkpoint(full.directory(seed, mode) / "predictions.pt")
            b = load_checkpoint(resumed.directory(seed, mode) / "predictions.pt")
            assert all(torch.equal(a[key], b[key]) for key in a)
    for name, digest in reference["artifacts"].items():
        assert sha256(full.source.output / name) == digest
    with pytest.raises(ValueError, match="identical config"):
        Experiment(replace(config, shots=64), paused.output, resume=True)
    with existing.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        resumed.verify_condition(0, "shape")


def test_default_full_validation_baseline_reproduces_old_exact_and_256_shots(tmp_path):
    shared = Experiment(Config(seeds="0"), tmp_path)
    assert len(shared.data["labels"]) == 5479
    result = run_experiment(shared, max_conditions=1)
    assert result["status"] == "partial"
    baseline = shared.verify_condition(0, "measured")
    assert baseline is not None
    assert baseline["baseline_reproduction"]["finite_shots_identical"]
    assert max(baseline["baseline_reproduction"]["max_errors"].values()) == 0
    assert not result["test_evaluated"] and not result["engineering_subset"]


def test_isolation_and_invalid_config_are_rejected():
    config = Config()
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    with pytest.raises(ValueError, match="isolated"):
        check_output(config, Path(config.reference) / "interventions")
    for bad in (
        replace(config, seeds="0,0"),
        replace(config, shots=0),
        replace(config, val_limit=1),
    ):
        with pytest.raises(ValueError):
            bad.validate()
