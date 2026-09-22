"""Keep true-control training distinct from evaluation interventions."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from experiments.grouped_control_diagnostics.protocol import (
    Config as OldConfig,
)
from experiments.grouped_control_diagnostics.protocol import (
    Experiment as OldExperiment,
)
from experiments.grouped_control_diagnostics.runner import RouteRun as OldRun
from experiments.grouped_dynamic_vqc.tests.test_training import assert_tree_equal
from experiments.grouped_feedback_ablation.protocol import load_checkpoint, read_json
from experiments.grouped_sequential_intervention.protocol import MODES, verify_artifacts
from experiments.grouped_vqc_training_modes.protocol import state_hash

from ..artifacts import import_historical, verify_complete
from ..evaluation import directory, evaluate_model, verify_condition
from ..protocol import Config, Experiment, check_output
from ..results import compare_modes, summarize
from ..train import run_experiment
from ..trainer import IndependentRun


def small_config(**changes) -> Config:
    return replace(
        Config(
            seeds="0",
            head_epochs=2,
            batch_size=18,
            eval_batch_size=18,
            shots=32,
            train_limit=36,
            val_limit=18,
            checkpoint_steps=1,
            development=True,
        ),
        **changes,
    )


def test_cuda_training_matches_historical_true_record_step(tmp_path):
    shared = Experiment(small_config(), tmp_path / "new")
    current = IndependentRun(shared, 0)
    initial = shared.initial_for(0)
    proxy = SimpleNamespace(
        output=tmp_path / "old",
        routes=("head_true",),
        config=OldConfig(head_epochs=2, batch_size=18),
        start_epoch=100,
        source=initial,
        initial=initial,
        make_model=shared.make_model,
        data=shared.data,
        manifest_hash=shared.manifest_hash,
        runtime=shared.runtime,
    )
    original = OldRun(cast(OldExperiment, proxy), "head_true")
    indices = torch.arange(18, device="cuda")
    a = current.train_step(indices)
    b = original.train_step(indices)
    assert a[0] == pytest.approx(b[0], abs=2e-7)
    for key, value in current.model.state_dict().items():
        torch.testing.assert_close(
            value, original.model.state_dict()[key], atol=2e-6, rtol=2e-6
        )
    assert state_hash(current.model.frontend.state_dict()) == current.initial_frontend
    assert current.gradient_checks["gradient_l2"]["frontend"] is None
    assert current.gradient_checks["gradient_l2"]["label_head"] > 0
    assert sum(p.numel() for p in current.model.parameters() if p.requires_grad) == 24
    assert all(int(s["step"]) == 1 for s in current.optimizer.state.values())
    assert current.mode == "both"


def test_mid_epoch_adam_resume_is_exact(tmp_path):
    config = small_config()
    full_shared = Experiment(config, tmp_path / "full")
    full = IndependentRun(full_shared, 0)
    full.run()
    paused_shared = Experiment(config, tmp_path / "paused")
    paused = IndependentRun(paused_shared, 0)
    assert paused.run(max_steps=1)["status"] == "paused"
    assert paused.progress["offset"] == 18
    resumed_shared = Experiment(config, paused_shared.output, resume=True)
    resumed = IndependentRun(resumed_shared, 0)
    resumed.run()
    assert_tree_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_tree_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_tree_equal(full.progress, resumed.progress)
    verify_complete(resumed_shared, 0)
    with (resumed.output / "endpoint.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_complete(resumed_shared, 0)


def test_two_seed_pipeline_pairing_report_and_evaluation_recovery(tmp_path):
    config = small_config(seeds="0,1", head_epochs=1)
    shared = Experiment(config, tmp_path)
    before = read_json(tmp_path / "reference_lock.json")["artifacts"]
    partial = run_experiment(shared, max_seeds=1)
    assert partial["status"] == "partial" and partial["complete_seeds"] == [0]
    p = directory(shared, "independent", 0, "measured") / "predictions.pt"
    timestamp = p.stat().st_mtime_ns
    resumed = Experiment(config, tmp_path, resume=True)
    result = run_experiment(resumed)
    assert result["status"] == "complete" and result["complete_seeds"] == [0, 1]
    assert result["engineering_subset"] and not result["paired_training_budget"]
    assert len(result["rows"]) == 16 and len(result["paired_results"]) == 8
    assert p.stat().st_mtime_ns == timestamp
    # Recompute one interrupted/uncommitted condition with its fixed sampling seed.
    target = directory(resumed, "independent", 1, "scale")
    previous = load_checkpoint(target / "predictions.pt")
    (target / "evaluation_lock.json").unlink()
    evaluate_model(resumed, "independent", 1)
    actual = load_checkpoint(target / "predictions.pt")
    assert all(torch.equal(previous[k], actual[k]) for k in previous)
    for seed in (0, 1):
        a = load_checkpoint(
            directory(resumed, "independent", seed, "measured") / "predictions.pt"
        )
        for training in ("independent", "sequential"):
            for mode in MODES:
                b = load_checkpoint(
                    directory(resumed, training, seed, mode) / "predictions.pt"
                )
                assert torch.equal(
                    a["concept_probabilities"], b["concept_probabilities"]
                )
                assert torch.equal(a["source_index"], b["source_index"])
    verify_artifacts(Path("/"), before)
    summarize(resumed)
    with (target / "predictions.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_condition(resumed, "independent", 1, "scale")


def test_full_seed0_reuse_budget_and_finite_sampling(tmp_path):
    shared = Experiment(Config(seeds="0"), tmp_path)
    assert shared.true_reference["reuse_seed"] == 0
    assert shared.sequential_reference["reuse"] and shared.paired_training
    result = import_historical(shared, 0)
    assert result["origin"] == "historical" and result["global_step"] == 2500
    evaluate_model(shared, "sequential", 0)
    evaluate_model(shared, "independent", 0)
    summary = summarize(shared)
    assert summary["status"] == "complete" and not summary["engineering_subset"]
    assert summary["n_samples"] == 5479 and summary["train_samples"] == 25593
    baseline = verify_condition(shared, "independent", 0, "measured")
    corrected = verify_condition(shared, "independent", 0, "both")
    assert baseline is not None and corrected is not None
    assert baseline["label"]["accuracy"] == pytest.approx(0.7070633172988892)
    assert corrected["label"]["accuracy"] == pytest.approx(0.939039945602417)
    assert corrected["finite_shots"]["control_mode"] == "both"
    # Finite shots can change thresholded accuracy substantially near 0.5.
    # Validate the sampled probabilities against binomial variance instead.
    for mode in MODES:
        raw = load_checkpoint(
            directory(shared, "independent", 0, mode) / "predictions.pt"
        )
        p = raw["label_probabilities"]
        observed = (raw["shot_label_probabilities"] - p).square().mean()
        expected = (p * (1 - p) / shared.config.shots).mean()
        assert 0.8 < float(observed / expected) < 1.2
    assert (
        directory(shared, "sequential", 0, "measured") / "predictions.pt"
    ).read_bytes() == (
        Path(shared.config.sequential_reference) / "seed0/measured/predictions.pt"
    ).read_bytes()
    with pytest.raises(ValueError, match="identical config"):
        Experiment(replace(shared.config, shots=128), tmp_path, resume=True)


def test_full_batch_has_cuda_gradients_without_reusing_trained_classifier(tmp_path):
    shared = Experiment(Config(seeds="1", retrain_reference=True), tmp_path)
    runner = IndependentRun(shared, 1)
    trained = load_checkpoint(shared.model_path("sequential", 1))["model"]
    assert state_hash(runner.model.state_dict()) != state_hash(trained)
    assert not runner.optimizer.state
    runner.train_step(torch.arange(1024, device="cuda"))
    assert runner.gradient_checks["gradient_l2"]["label_head"] > 0
    assert runner.gradient_checks["gradient_l2"]["frontend"] is None


def test_pair_gap_and_guardrails():
    def row(accuracy, gain):
        return {
            "seed": 0,
            "mode": "both",
            "accuracy": accuracy,
            "delta_accuracy_pp": gain,
            "shot_accuracy": accuracy,
            "shot_delta_accuracy_pp": gain,
            "bce": 0.5,
        }

    gap = compare_modes(row(0.8, 20), row(0.9, 10))
    assert gap["accuracy_difference_pp"] == pytest.approx(-10)
    assert gap["intervention_gain_difference_pp"] == 10
    with pytest.raises(ValueError, match="isolated"):
        check_output(Config(), Path(Config().true_reference) / "new")
    for config in (
        replace(Config(), seeds="0,0"),
        replace(Config(), train_limit=18),
        replace(Config(), head_epochs=0),
        replace(Config(), learning_rate=float("nan")),
    ):
        with pytest.raises(ValueError):
            config.validate()
