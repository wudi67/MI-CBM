"""Check original-loop parity, paired samples, CUDA gradients and exact recovery."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.grouped_dynamic_vqc.runtime import sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import Route as PilotRoute
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .. import results
from ..model import FRONT_KEYS, verify_initializations
from ..operations import status_table
from ..protocol import CELLS, Config, cell_name, check_output
from ..runner import Experiment, run_experiment
from ..training import Job, Route

pytest_plugins = ["experiments.grouped_robot_continuation.tests.test_continuation"]


def snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file() and p.name != "heartbeat.json"
    }


def config_for(baseline) -> Config:
    return Config(
        reference=str(baseline.output),
        concept_epochs=3,
        checkpoint_steps=1,
        development=True,
    )


def test_cuda_comparison_preserves_original_training_and_references(
    baseline, tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "plots", lambda *_: None)
    before = snapshot(baseline.output)
    shared = Experiment(config_for(baseline), tmp_path / "comparison")
    verify_initializations(shared.initial, baseline.initial, shared.config)
    assert shared.config.sigma == 0.0025
    for method, index in CELLS:
        value = shared.initial[cell_name(method, index)]
        assert tree_hash(value["rng"]) == tree_hash(baseline.initial["rng"])
        for name, tensor in value["model"].items():
            if name not in FRONT_KEYS:
                assert torch.equal(tensor, baseline.initial["model"][name])
        angles = torch.cat([value["model"][k].flatten() for k in FRONT_KEYS])
        assert angles.numel() == 240
        if method == "uniform":
            assert 0 <= float(angles.min()) < float(angles.max()) <= torch.pi
        else:
            assert 0.002 < float(angles.std()) < 0.003
    run_experiment(shared)
    summary = read_json(shared.output / "summary.json")
    assert summary["completed_training_cells"] == 6
    assert summary["completed_conditions"] == 12
    assert summary["new_adam_updates"] == 36
    assert summary["same_sample_orders_verified"]
    assert not summary["test_read"] and not summary["test_evaluated"]
    assert len(summary["paired_vs_uniform"]) == 2
    for method, index in CELLS:
        ck = load(Job(shared, method, index).checkpoint_path)
        assert ck["gradient_checks"]["device"] == "cuda:0"
        assert ck["gradient_checks"]["gradient_l2"]["frontend"] > 0
        assert ck["gradient_checks"]["gradient_l2"]["label_head"] is None
        assert ck["gradient_checks"]["frontend_parameters"] == 240
        assert {int(v["step"]) for v in ck["optimizer"]["state"].values()} == {6}
    gold = Job(shared, "uniform", 0)
    gold.output = tmp_path / "unmodified_loop"
    PilotRoute(gold, "concept").run()
    actual = load(Job(shared, "uniform", 0).checkpoint_path)
    expected = load(gold.checkpoint_path)
    for key in ("model", "optimizer", "progress", "rng"):
        assert tree_hash(actual[key]) == tree_hash(expected[key])
    assert snapshot(baseline.output) == before
    after = snapshot(shared.output)
    run_experiment(Experiment(shared.config, shared.output, resume=True))
    assert snapshot(shared.output) == after
    assert status_table(shared.output)[1] == "complete"
    (shared.output / "eft_gaussian/init_2/validation/predictions.pt").write_bytes(
        b"tampered"
    )
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(shared.config, shared.output, resume=True)


def test_mid_batch_recovery_and_protocol_guards(baseline, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "plots", lambda *_: None)
    cfg = config_for(baseline)
    shared = Experiment(cfg, tmp_path / "paused")
    job = Job(shared, "eft_gaussian", 0)
    with pytest.raises(InterruptedError):
        Route(job).run(max_steps=1)
    paused = load(job.output / "training/concept/resume.pt")
    assert paused["progress"]["offset"] == 32
    shared = Experiment(cfg, shared.output, resume=True)
    run_experiment(shared)
    resumed = load(Job(shared, "eft_gaussian", 0).checkpoint_path)
    gold = Job(shared, "eft_gaussian", 0)
    gold.output = tmp_path / "continuous"
    Route(gold).run()
    expected = load(gold.checkpoint_path)
    for key in ("model", "optimizer", "progress", "rng"):
        assert tree_hash(resumed[key]) == tree_hash(expected[key])
    with pytest.raises(ValueError, match="unchanged"):
        Experiment(replace(cfg, init_kappa=0.2), shared.output, resume=True)
    for root in (
        baseline.output,
        baseline.output / "new",
        baseline.output.parent,
        Path(baseline.config.dataset) / "new",
    ):
        with pytest.raises(ValueError):
            check_output(cfg, root)
    for value in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replace(cfg, init_kappa=value).source()
    with pytest.raises(ValueError):
        replace(cfg, development=False).source()
    broken = Job(shared, "eft_gaussian", 1)
    broken.output = tmp_path / "broken"
    with pytest.raises(InterruptedError):
        Route(broken).run(max_steps=1)
    path = broken.output / "training/concept/resume.pt"
    value = load(path)
    value["progress"]["order"] = value["progress"]["order"].flip(0)
    torch.save(value, path)
    with pytest.raises(ValueError, match="order"):
        Route(broken)
    assert state_hash(shared.initial["uniform/init_0"]["model"]) == state_hash(
        baseline.initial["model"]
    )
