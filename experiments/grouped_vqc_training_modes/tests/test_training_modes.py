"""Exercise real dSprites/CUDA at mid-epoch and stage-boundary interruptions."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest
import torch

from experiments.grouped_dynamic_vqc.runtime import atomic_json
from experiments.grouped_dynamic_vqc.tests.test_training import assert_tree_equal

from ..operations import status_table
from ..protocol import ExperimentConfig, SharedExperiment, state_hash
from ..results import summarize
from ..runner import RouteRun
from ..train import run_experiment


def small_config() -> ExperimentConfig:
    return ExperimentConfig(
        epochs=4,
        concept_epochs=2,
        batch_size=18,
        eval_batch_size=18,
        train_limit=36,
        val_limit=18,
        checkpoint_steps=1,
        shots=8,
    )


@pytest.mark.parametrize("route", ["joint", "sequential"])
def test_exact_resume_across_mid_epoch_and_phase_boundary(tmp_path, route):
    config = small_config()
    full_shared = SharedExperiment(config, tmp_path / "full", resume=False)
    full = RouteRun(full_shared, route)
    full_result = full.run()
    resumed_path = tmp_path / "resumed"
    resumed_shared = SharedExperiment(config, resumed_path, resume=False)
    interrupted = RouteRun(resumed_shared, route)
    for step in (1, 4, 5):
        assert interrupted.run(max_steps=step)["status"] == "paused"
        if step == 1:
            assert interrupted.progress["offset"] == 18
        if step == 4:
            assert interrupted.progress["completed_epoch"] == 2
            assert (interrupted.output / "endpoints/epoch_0002.json").exists()
            if route == "sequential":
                assert interrupted.phase == "concept"
                assert (
                    state_hash(interrupted.model.label_head.state_dict())
                    == interrupted.initial_head_hash
                )
        if step == 5 and route == "sequential":
            assert interrupted.phase == "label"
            assert all(
                not p.requires_grad for p in interrupted.model.frontend.parameters()
            )
            assert all(
                int(value["step"]) == 1
                for value in interrupted.optimizer.state.values()
            )
        resumed_shared = SharedExperiment(config, resumed_path, resume=True)
        interrupted = RouteRun(resumed_shared, route, resume=True)
    resumed_result = interrupted.run()
    assert_tree_equal(full_result["validation"], resumed_result["validation"])
    full_checkpoint = torch.load(
        full.output / "resume.pt", weights_only=False, map_location="cpu"
    )
    resumed_checkpoint = torch.load(
        interrupted.output / "resume.pt", weights_only=False, map_location="cpu"
    )
    for key in (
        "model",
        "optimizer",
        "progress",
        "rng",
        "module_steps",
        "phase",
        "gradient_checks",
    ):
        assert_tree_equal(full_checkpoint[key], resumed_checkpoint[key])
    assert all(value.is_cuda for value in resumed_shared.data["train"].values())
    assert resumed_result["test_evaluated"] is False


def test_concept_only_update_is_independent_of_labels_and_freezes_head(tmp_path):
    shared = SharedExperiment(small_config(), tmp_path / "shared", resume=False)
    alternate = copy.copy(shared)
    alternate.output = tmp_path / "opposite_labels"
    alternate.data = {
        **shared.data,
        "train": {**shared.data["train"], "labels": 1 - shared.data["train"]["labels"]},
    }
    first = RouteRun(shared, "sequential")
    second = RouteRun(alternate, "sequential")
    initial = state_hash(first.model.frontend.state_dict())
    indices = torch.arange(18, device="cuda")
    first.train_step(indices)
    second.train_step(indices)
    assert state_hash(first.model.frontend.state_dict()) != initial
    assert_tree_equal(first.model.state_dict(), second.model.state_dict())
    assert state_hash(first.model.label_head.state_dict()) == first.initial_head_hash
    assert first.gradient_checks["concept"]["gradient_l2"]["label_head"] is None
    assert first.gradient_checks["concept"]["gradient_l2"]["frontend"] > 0


def test_paired_runner_checks_frozen_frontend_orders_and_budgets(tmp_path, monkeypatch):
    shared = SharedExperiment(small_config(), tmp_path, resume=False)
    summary = run_experiment(shared)
    assert summary["status"] == "complete"
    assert all(summary["pairing_checks"].values())
    assert len(summary["rows"]) == 4
    assert summary["rows"][1]["label_head_trained"] is False
    checkpoints = {
        route: torch.load(
            tmp_path / route / "resume.pt", map_location="cpu", weights_only=False
        )
        for route in ("joint", "sequential")
    }
    assert checkpoints["joint"]["module_steps"] == {"frontend": 8, "label_head": 8}
    assert checkpoints["sequential"]["module_steps"] == {"frontend": 4, "label_head": 4}
    checks = checkpoints["sequential"]["gradient_checks"]
    assert checks["label"]["gradient_l2"]["frontend"] is None
    assert checks["label"]["gradient_l2"]["label_head"] > 0
    assert all(
        len(endpoint["validation"]["control_record_interventions"]) == 4
        for endpoint in summary["endpoints"]
    )

    def forbid_training(*_args, **_kwargs):
        raise AssertionError("Completed routes must not retrain")

    monkeypatch.setattr(RouteRun, "train_step", forbid_training)
    assert run_experiment(shared)["status"] == "complete"
    path = tmp_path / "joint/history.json"
    history = json.loads(path.read_text())
    history[0]["order_sha256"] = "changed"
    atomic_json(path, history)
    with pytest.raises(ValueError):
        summarize(shared)


def test_resume_rejects_config_and_shared_initialization_changes(tmp_path):
    config = small_config()
    SharedExperiment(config, tmp_path, resume=False)
    with pytest.raises(ValueError, match="identical config"):
        SharedExperiment(replace(config, learning_rate=0.001), tmp_path, resume=True)
    with (tmp_path / "initialization.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Shared artifact changed"):
        SharedExperiment(config, tmp_path, resume=True)


def test_protocol_and_monitor_reject_invalid_or_stale_state(tmp_path):
    with pytest.raises(ValueError, match="concept_epochs"):
        replace(small_config(), concept_epochs=4).validate()
    now = datetime.now(UTC)
    atomic_json(
        tmp_path / "heartbeat.json",
        {
            "updated_at": now.isoformat(),
            "pid": 999999999,
            "status": "paused",
            "epoch_completed": 2,
            "epochs_total": 4,
            "global_step": 4,
            "offset": 0,
            "device": "CUDA",
            "route": "sequential",
            "phase": "concept",
        },
    )
    assert status_table(tmp_path)[1] == "paused"
    assert status_table(tmp_path, since=now.timestamp() + 1)[1] == "starting"
