"""Use the real frozen VQC cache to validate the classical controls."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from experiments.grouped_dynamic_vqc.runtime import atomic_json
from experiments.grouped_dynamic_vqc.tests.test_training import assert_tree_equal

from ..model import PARAMETERS, TinyMLP, evaluate, objective
from ..operations import status_table
from ..protocol import Config, Experiment, read_json
from ..runner import CellRun
from ..train import run_experiment


def small_config() -> Config:
    return Config(
        concept_epochs=2,
        label_epochs=3,
        train_limit=2050,
        val_limit=18,
        checkpoint_steps=1,
        learning_rates=(0.001, 0.01),
    )


@pytest.mark.parametrize("task", ["concept", "label"])
def test_parameter_counts_and_objective_supervision(task):
    model = TinyMLP(task)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[task]
    angles = torch.randn(12, 10, 4)
    indices = torch.arange(12)
    data = {
        "concepts": torch.tensor([[i % 3, i % 6] for i in range(12)]),
        "labels": (indices % 2).float(),
    }
    original = objective(task, model(angles), data, indices)
    unrelated = (
        {**data, "labels": 1 - data["labels"]}
        if task == "concept"
        else {**data, "concepts": torch.zeros_like(data["concepts"])}
    )
    changed = objective(task, model(angles), unrelated, indices)
    assert torch.equal(original, changed)
    original.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    if task == "concept":
        probabilities = model(angles).softmax(-1)
        assert probabilities.shape == (12, 32)
        # No post-hoc filtering of the 14 invalid codes.
        assert torch.all(probabilities[:, [6, 7, 24, 31]] > 0)


@pytest.mark.parametrize("task", ["concept", "label"])
def test_cuda_recovery_mid_epoch_and_before_validation(tmp_path, task, monkeypatch):
    config = small_config()
    full_exp = Experiment(config, tmp_path / "full")
    full = CellRun(full_exp, task, 0.001)
    full_result = full.run()
    resumed_path = tmp_path / "resumed"
    exp = Experiment(config, resumed_path)
    interrupted = CellRun(exp, task, 0.001)
    assert interrupted.run(max_steps=1)["status"] == "paused"
    assert interrupted.progress["offset"] == 1024
    exp = Experiment(config, resumed_path, resume=True)
    interrupted = CellRun(exp, task, 0.001, resume=True)

    # Simulate a crash after the last batch was checkpointed, before validation.
    def interrupt_validation():
        raise RuntimeError("simulated interruption before validation")

    monkeypatch.setattr(interrupted, "finish_epoch", interrupt_validation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted.run()
    exp = Experiment(config, resumed_path, resume=True)
    interrupted = CellRun(exp, task, 0.001, resume=True)
    assert interrupted.progress["offset"] == 2050
    resumed_result = interrupted.run()
    assert_tree_equal(full_result["validation"], resumed_result["validation"])
    first = torch.load(
        full.output / "resume.pt", map_location="cpu", weights_only=False
    )
    second = torch.load(
        interrupted.output / "resume.pt", map_location="cpu", weights_only=False
    )
    for key in ("model", "optimizer", "progress", "rng", "gradient_check"):
        assert_tree_equal(first[key], second[key])
    for role in exp.data.values():
        assert all(tensor.is_cuda for tensor in role.values())
    assert resumed_result["parameters"] == PARAMETERS[task]
    assert resumed_result["test_evaluated"] is False
    assert interrupted.gradient_check["input_device"] == "cuda:0"
    metrics = evaluate(interrupted.model, exp.data["val"], config.eval_batch_size)
    assert_tree_equal(metrics, resumed_result["validation"])


def test_grid_selection_full_input_pairing_and_completed_resume(tmp_path, monkeypatch):
    # Full cached rows: even a short engineering run must use the VQC permutations.
    config = replace(
        small_config(), concept_epochs=1, label_epochs=1, train_limit=0, val_limit=0
    )
    experiment = Experiment(config, tmp_path)
    assert experiment.full_reference_inputs
    assert experiment.order_hashes == experiment.reference["order_hashes"][:1]
    reference_cache = str(Path(config.reference).resolve() / "data.pt")
    assert (
        experiment.manifest["shared_artifacts"]["data.pt"]
        == experiment.reference["artifacts"][reference_cache]
    )
    summary = run_experiment(experiment)
    assert len(summary["all_cells"]) == 4
    assert summary["checks"]["full_reference_input_and_order"]
    assert (
        summary["comparisons"] == []
    )  # One epoch is not the 100/200 epoch comparison.
    for task in ("concept", "label"):
        cells = [cell for cell in summary["all_cells"] if cell["task"] == task]
        assert len({cell["initial_model_sha256"] for cell in cells}) == 1
        winner = min(
            cells, key=lambda cell: (cell["validation"]["loss"], cell["learning_rate"])
        )
        assert summary["selected_by_task"][task]["cell"] == winner["cell"]

    def forbid_training(*_args, **_kwargs):
        raise AssertionError("Completed cells must be verified and skipped")

    monkeypatch.setattr(CellRun, "train_step", forbid_training)
    assert run_experiment(experiment)["status"] == "complete"
    path = tmp_path / "concept/lr_0p001/history.json"
    history = read_json(path)
    history[0]["order_sha256"] = "modified"
    atomic_json(path, history)
    with pytest.raises(ValueError):
        run_experiment(experiment)


def test_input_cache_and_configuration_cannot_change_on_resume(tmp_path):
    config = small_config()
    experiment = Experiment(config, tmp_path)
    with pytest.raises(ValueError, match="identical config"):
        Experiment(replace(config, concept_epochs=1), tmp_path, resume=True)
    with (tmp_path / "data.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Shared artifact changed"):
        Experiment(config, tmp_path, resume=True)
    assert experiment.audit["test_preprocessed"] is False
    assert experiment.audit["routing_fit_rows"] == 25593


def test_invalid_configuration_and_task_specific_monitor(tmp_path):
    with pytest.raises(ValueError):
        replace(Config(), learning_rates=(0.001, 0.001)).validate()
    with pytest.raises(ValueError):
        replace(Config(), learning_rates=(float("nan"),)).validate()
    now = datetime.now(UTC)
    atomic_json(
        tmp_path / "heartbeat.json",
        {
            "updated_at": now.isoformat(),
            "pid": 999999999,
            "status": "complete",
            "epoch_completed": 1,
            "epochs_total": 1,
            "global_step": 1,
            "offset": 0,
            "device": "CUDA",
            "cell": "concept/lr_0p001",
            "completed_cells": 6,
            "total_cells": 6,
            "validation": {"loss": 1.0, "concept": {"joint_map_accuracy": 0.5}},
        },
    )
    assert status_table(tmp_path)[1] == "complete"
    assert status_table(tmp_path, since=now.timestamp() + 1)[1] == "starting"
