"""Exercise real frozen CUDA models and the independently saved zero baseline."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.grouped_dynamic_vqc.runtime import sha256
from experiments.grouped_feedback_ablation.protocol import load_checkpoint, read_json

from .. import protocol
from ..protocol import VARIANTS, Config, Sources, check_output
from ..results import paired_row, statistics_for
from ..runner import Experiment, run_experiment


def test_full_validation_reuse_and_cuda_replay_agree(tmp_path):
    config = Config(seeds="0")
    reused = Experiment(config, tmp_path / "reuse")
    assert reused.reuse and reused.runtime["device"] == "NVIDIA GeForce RTX 4090"
    reference = reused.sources.hashes.copy()
    result = run_experiment(reused)
    assert result["status"] == "complete" and not result["engineering_subset"]
    assert result["n_samples"] == 5479 and not result["test_evaluated"]
    replay = Experiment(replace(config, recompute=True), tmp_path / "replay")
    assert not replay.reuse
    other = run_experiment(replay)
    assert result["rows"] == other["rows"]
    assert result["paired_results"] == other["paired_results"]
    for variant in VARIANTS:
        a = load_checkpoint(reused.directory(0, variant) / "predictions.pt")
        b = load_checkpoint(replay.directory(0, variant) / "predictions.pt")
        assert all(torch.equal(a[key], b[key]) for key in b)
        assert sha256(reused.directory(0, variant) / "predictions.pt") == sha256(
            reused.sources.evaluation_path(0, variant) / "predictions.pt"
        )
    for filename, digest in reference.items():
        assert sha256(Path(filename)) == digest
    # Normal Independent prediction must not accidentally import corrected controls.
    metrics = reused.verify_condition(0, "feedback")
    assert metrics is not None and metrics["control_mode"] == "measured"
    assert metrics["label"]["accuracy"] == pytest.approx(0.7070633173)
    assert result["paired_results"][0]["no_feedback_accuracy"] == pytest.approx(
        0.6439131498
    )


def test_resume_partial_pair_and_tamper_guards(tmp_path):
    config = Config(
        seeds="0,1", val_limit=18, development=True, shots=32, eval_batch_size=18
    )
    shared = Experiment(config, tmp_path / "paused")
    partial = run_experiment(shared, max_conditions=1)
    assert partial["complete_seeds"] == [] and partial["statistics"] == {}
    assert len(partial["rows"]) == 1 and not partial["paired_results"]
    existing = shared.directory(0, "feedback") / "predictions.pt"
    timestamp = existing.stat().st_mtime_ns
    resumed = Experiment(config, shared.output, resume=True)
    completed = run_experiment(resumed)
    assert completed["complete_seeds"] == [0, 1]
    assert len(completed["rows"]) == 4 and len(completed["paired_results"]) == 2
    assert completed["engineering_subset"]
    assert existing.stat().st_mtime_ns == timestamp
    full = run_experiment(Experiment(config, tmp_path / "full"))
    assert completed["rows"] == full["rows"]
    assert completed["statistics"] == full["statistics"]
    with pytest.raises(ValueError, match="identical config"):
        Experiment(replace(config, shots=64), shared.output, resume=True)
    with existing.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        resumed.verify_condition(0, "feedback")


def test_wrong_training_pair_is_rejected(monkeypatch):
    original = protocol.verify_zero

    def mismatched(*args, **kwargs):
        value = dict(original(*args, **kwargs))
        value["initial_model_sha256"] = "different"
        return value

    monkeypatch.setattr(protocol, "verify_zero", mismatched)
    with pytest.raises(ValueError, match="Training pairing mismatch"):
        Sources(Config(seeds="0"))


def test_metrics_preserve_harms_and_reject_mispaired_concepts():
    baseline = {
        "labels": torch.ones(4),
        "concepts": torch.zeros(4, 2, dtype=torch.long),
        "source_index": torch.arange(4),
        "concept_probabilities": torch.full((4, 32), 1 / 32),
        "label_probabilities": torch.tensor([0.1, 0.9, 0.6, 0.2]),
        "shot_label_probabilities": torch.tensor([0.1, 0.9, 0.6, 0.2]),
    }
    feedback = {**baseline, "label_probabilities": torch.tensor([0.8, 0.1, 0.1, 0.2])}
    row = paired_row(0, feedback, baseline)
    assert row["wrong_to_right_count"] == 1 and row["right_to_wrong_count"] == 2
    assert row["delta_accuracy_pp"] == -25
    stats = statistics_for([row, {**row, "seed": 1, "delta_accuracy_pp": 25}])
    assert stats["delta_accuracy_pp"]["mean"] == 0
    assert stats["delta_accuracy_pp"]["sample_std"] == pytest.approx(25 * 2**0.5)
    assert statistics_for([row])["delta_accuracy_pp"]["sample_std"] is None
    with pytest.raises(ValueError, match="concept distribution differ"):
        paired_row(0, {**feedback, "source_index": torch.arange(4).flip(0)}, baseline)


def test_isolation_and_cli_plan_creates_no_output(tmp_path):
    config = Config()
    for path in (
        Path(config.reference),
        Path(config.reference) / "new",
        Path(config.reference).parent,
        Path(read_json(Path(config.reference) / "config.json")["reference"]) / "new",
    ):
        with pytest.raises(ValueError, match="isolated"):
            check_output(config, path)
    for bad in (
        replace(config, seeds="0,0"),
        replace(config, val_limit=18),
        replace(config, shots=0),
        replace(config, val_limit=1, development=True),
    ):
        with pytest.raises(ValueError):
            bad.validate()
    check_output(config, tmp_path / "valid")
