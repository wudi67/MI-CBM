"""Physical correction, correct averaging, and deterministic resumed evaluation."""

from dataclasses import replace
from pathlib import Path
from statistics import stdev

import pytest
import torch

import torchquantum as tq
import torchquantum.functional as tqf
from experiments.grouped_dynamic_vqc.evaluation import label_metrics
from experiments.grouped_dynamic_vqc.runtime import cuda_runtime, sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_pilot.model import RobotVQC, bits
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_robot_shots_final import results as shots_results
from experiments.grouped_robot_shots_final.protocol import Config as ShotsConfig
from experiments.grouped_robot_shots_final.runner import Experiment as ShotsExperiment
from experiments.grouped_robot_shots_final.runner import run_experiment as run_shots

from .. import results
from ..evaluation import infer, verify_concepts
from ..protocol import MODES, Config, check_output, selected_concepts
from ..results import curve_rows
from ..runner import Experiment, run_experiment

pytest_plugins = ["experiments.grouped_robot_shots_final.tests.test_pipeline"]


def test_all_subsets_scored_before_averaging_and_seed_statistics():
    assert [len([m for m in range(32) if m.bit_count() == k]) for k in range(6)] == [
        1,
        5,
        10,
        10,
        5,
        1,
    ]
    assert selected_concepts(16) == ["head_shape"]
    assert selected_concepts(1) == ["foot_shape"]
    evaluations = []
    for seed in (0, 1):
        for mode in MODES:
            for mask in range(32):
                p = 0.9 if mask in ((1,) if seed == 0 else (1, 2)) else 0.49
                evaluations.append(
                    {
                        "role": "test",
                        "seed": seed,
                        "training": mode,
                        "mask": mask,
                        "metrics": {
                            "label": label_metrics(
                                torch.tensor([p, 1 - p]), torch.tensor([1.0, 0.0])
                            )
                        },
                    }
                )
    rows, summaries = curve_rows(evaluations, [0, 1])
    one = [r for r in rows if r["training"] == "independent" and r["count"] == 1]
    assert [r["accuracy"] for r in one] == [0.2, 0.4]
    # Mean probabilities would classify both examples correctly (accuracy 1),
    # while the intended random-subset policy succeeds on only 1/5 or 2/5.
    summary = next(
        r for r in summaries if r["training"] == "independent" and r["count"] == 1
    )
    assert summary["accuracy"]["mean"] == pytest.approx(0.3)
    assert summary["accuracy"]["std"] == stdev([0.2, 0.4])
    assert summary["gain_pp"]["mean"] == pytest.approx(30)
    assert len(rows) == 36 and len(summaries) == 18
    assert (
        next(
            r
            for r in rows
            if r["seed"] == 0 and r["training"] == "joint" and r["count"] == 2
        )["step_gain_pp"]
        < 0
    )
    with pytest.raises(ValueError, match="exactly once"):
        curve_rows(evaluations[:-1], [0, 1])
    with pytest.raises(ValueError, match="exactly once"):
        curve_rows(evaluations + [evaluations[0]], [0, 1])


@torch.no_grad()
def test_cuda_partial_interventions_match_explicit_physical_x_gates():
    cuda_runtime(7)
    model = RobotVQC(front_layers=4, label_layers=5).cuda().eval().requires_grad_(False)
    data = {
        "angles": torch.rand(2, 10, 4),
        "concepts": bits(torch.device("cpu"))[[9, 22]],
        "labels": torch.tensor([0.0, 1.0]),
        "source_index": torch.arange(2),
        "robot_ids": torch.arange(2),
    }
    states = model.frontend(data["angles"].cuda())
    before = tree_hash(model.state_dict())
    baseline = infer(model, data, states, 0, 2, lambda *_a, **_k: None)
    for mask in (1, 10, 17, 31):
        actual = infer(model, data, states, mask, 2, lambda *_a, **_k: None)
        verify_concepts(actual, baseline)
        trajectories = []
        for row in range(2):
            for measured in range(32):
                qdev = tq.QuantumDevice(n_wires=5, bsz=1, device="cuda")
                qdev.set_states(states[row].reshape(32, 32)[measured : measured + 1])
                for wire in range(5):
                    chosen = (
                        int(data["concepts"][row, wire])
                        if mask & (1 << (4 - wire))
                        else (measured >> (4 - wire)) & 1
                    )
                    if chosen:
                        tqf.x(qdev, wires=wire)
                trajectories.append(qdev.get_states_1d())
        direct = model.label_head(torch.cat(trajectories)).reshape(2, 32).cpu()
        torch.testing.assert_close(
            actual["branch_label_mass"], direct, atol=2e-6, rtol=2e-6
        )
    assert tree_hash(model.state_dict()) == before
    assert not any(p.requires_grad for p in model.parameters())
    assert states.device.type == "cuda"


@pytest.fixture(name="source_run")
def completed_shots(models, tmp_path, monkeypatch):
    monkeypatch.setattr(shots_results, "plots", lambda *_: [])
    cfg = ShotsConfig(
        reference=str(models.output),
        seeds="0,1",
        shots="8",
        repeats=1,
        eval_batch_size=32,
        val_limit=32,
        development=True,
    )
    source = ShotsExperiment(cfg, tmp_path / "shots")
    run_shots(source)
    return source


def snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_cuda_frozen_pipeline_resume_endpoints_and_reference_integrity(
    source_run, tmp_path, monkeypatch
):
    old = snapshot(source_run.output)
    model_before = snapshot(Path(source_run.config.reference))
    cfg = Config(
        reference=str(source_run.output),
        seeds="0,1",
        eval_batch_size=32,
        limit=32,
        development=True,
    )

    def forbidden_optimizer(*_a, **_k):
        raise AssertionError("Curve evaluation created an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", forbidden_optimizer)
    shared = Experiment(cfg, tmp_path / "paused")
    # Includes one reused endpoint and two new partial corrections.
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_conditions=3)
    first = shared.directory(0, "independent", 1) / "joint.pt"
    timestamp = first.stat().st_mtime_ns
    assert not (shared.output / "result_lock.json").exists()
    resumed = Experiment(cfg, shared.output, resume=True)
    run_experiment(resumed)
    assert first.stat().st_mtime_ns == timestamp
    summary = read_json(shared.output / "summary.json")
    assert summary["completed_conditions"] == 192
    assert summary["reused_endpoints"] == 12
    assert summary["new_partial_conditions"] == 180
    assert not summary["test_read"] and not summary["test_evaluated"]
    assert (shared.output / "intervention_curve.pdf").stat().st_size > 1000
    assert not list(shared.output.rglob("endpoint.pt"))
    for seed in (0, 1):
        for mode in MODES:
            for mask in (0, 31):
                actual = load(resumed.directory(seed, mode, mask) / "joint.pt")
                expected = resumed.sources.endpoint(seed, mode, mask)
                assert tree_hash(actual) == tree_hash(expected)
    monkeypatch.setattr(results, "plots", lambda *_: None)
    full = Experiment(cfg, tmp_path / "full")
    run_experiment(full)
    for p in shared.output.glob("seed*/*/mask*/joint.pt"):
        assert tree_hash(load(p)) == tree_hash(
            load(full.output / p.relative_to(shared.output))
        )
    assert read_json(full.output / "summary.json") == read_json(
        shared.output / "summary.json"
    )
    locked = snapshot(shared.output)
    run_experiment(Experiment(cfg, shared.output, resume=True))
    assert {
        k: v for k, v in snapshot(shared.output).items() if k != "heartbeat.json"
    } == {k: v for k, v in locked.items() if k != "heartbeat.json"}
    assert snapshot(source_run.output) == old
    assert snapshot(Path(source_run.config.reference)) == model_before
    with pytest.raises(ValueError, match="unchanged"):
        Experiment(replace(cfg, eval_batch_size=16), shared.output, resume=True)
    with pytest.raises(ValueError, match="overlaps"):
        check_output(cfg, source_run.output / "child")
    with pytest.raises(ValueError, match="Formal curves"):
        replace(cfg, development=False).validate()
    with first.open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(cfg, shared.output, resume=True)
