"""Actual CUDA circuits: seed pairing, physical interventions and exact recovery."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.grouped_dynamic_vqc.runtime import atomic_checkpoint, sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import Route as PilotRoute
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .. import results
from ..model import make_model, module_state
from ..protocol import Config, check_output
from ..reference import verify_endpoint
from ..runner import Experiment, run_experiment
from ..training import Job, Route

pytest_plugins = ["experiments.grouped_robot_continuation.tests.test_continuation"]


def config_for(baseline, **kwargs) -> Config:
    return Config(
        reference=str(baseline.output),
        concept_epochs=2,
        head_epochs=2,
        head_order_offset=2,
        checkpoint_steps=1,
        seeds="0,1",
        fresh_all=True,
        development=True,
        **kwargs,
    )


def snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file() and p.name != "heartbeat.json"
    }


def assert_same(left: dict, right: dict) -> None:
    for name in ("model", "optimizer", "progress", "rng"):
        assert tree_hash(left[name]) == tree_hash(right[name]), name


def test_full_cuda_pairing_interventions_and_original_loop(baseline, tmp_path):
    before = snapshot(baseline.output)
    cfg = config_for(baseline)
    shared = Experiment(cfg, tmp_path / "full")
    run_experiment(shared)
    summary = read_json(shared.output / "summary.json")
    assert summary["completed_training_cells"] == 6
    assert summary["completed_conditions"] == 12
    assert summary["new_adam_updates"] == 24
    assert not summary["test_read"] and not summary["test_evaluated"]
    assert not (Path(baseline.config.dataset) / "robot_images_test_labels.csv").exists()
    assert (shared.output / "validation_results.pdf").stat().st_size > 1000
    for seed in cfg.seed_values:
        for cell in ("concept", "independent", "no_feedback"):
            job = Job(shared, seed, cell)
            checkpoint = load(job.checkpoint_path)
            gradients = checkpoint["gradient_checks"]
            assert gradients["device"] == "cuda:0"
            assert gradients["active_parameters"] == (240 if cell == "concept" else 112)
            active = "frontend" if cell == "concept" else "label_head"
            frozen = "label_head" if cell == "concept" else "frontend"
            assert gradients["gradient_l2"][active] > 0
            assert gradients["gradient_l2"][frozen] is None
            assert checkpoint["progress"]["history"][0]["order_epoch"] == (
                1 if cell == "concept" else 3
            )
        a, b = (
            read_json(
                shared.output / f"seed_{seed}/training/{cell}/initialization.json"
            )
            for cell in ("independent", "no_feedback")
        )
        assert a["head_sha256"] == b["head_sha256"]
        assert a["frontend_sha256"] == b["frontend_sha256"]
        assert a["initial_model_sha256"] == b["initial_model_sha256"]
        raw_root = shared.output / f"seed_{seed}/validation"
        normal = load(raw_root / "independent/measured/predictions.pt")
        corrected = load(raw_root / "independent/correct_all_five/predictions.pt")
        zero = load(raw_root / "no_feedback/zero/predictions.pt")
        assert torch.equal(
            normal["concept_probabilities"], corrected["concept_probabilities"]
        )
        assert torch.equal(
            normal["concept_probabilities"], zero["concept_probabilities"]
        )
        pair = next(
            r
            for r in summary["paired_rows"]
            if r["role"] == "validation" and r["seed"] == seed
        )
        assert pair["correction_gain_pp"] == 100 * (
            pair["corrected_label_accuracy"] - pair["normal_label_accuracy"]
        )
    # A complete seed changes frontend initialization AND training permutations.
    assert state_hash(shared.initial["0"]["model"]) != state_hash(
        shared.initial["1"]["model"]
    )
    assert not torch.equal(epoch_order(64, 0, 1), epoch_order(64, 1, 1))
    assert snapshot(baseline.output) == before
    # The adapter must reproduce the pre-existing mathematical training loop.
    for cell in ("concept", "independent", "no_feedback"):
        job = Job(shared, 1, cell)
        job.output = tmp_path / f"gold_{cell}"
        PilotRoute(job, cell).run()
        assert_same(
            load(job.output / f"training/{cell}/endpoint.pt"),
            load(shared.checkpoint_path(1, cell)),
        )
    value = summary["paired_summary"]["validation"]["feedback_gain_pp"]
    assert value["n"] == 2 and value["sample_std"] is not None


def test_mid_batch_resume_identical_and_output_tamper_rejected(
    baseline, tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "plots", lambda *_: None)
    cfg = config_for(baseline)
    paused = Experiment(cfg, tmp_path / "paused")
    with pytest.raises(InterruptedError):
        run_experiment(paused, max_steps=1)
    partial = load(paused.output / "seed_0/training/concept/resume.pt")
    assert partial["progress"]["offset"] == 32
    resumed = Experiment(cfg, paused.output, resume=True)
    with pytest.raises(InterruptedError):
        run_experiment(resumed, max_steps=4)
    partial = load(paused.output / "seed_0/training/independent/resume.pt")
    assert partial["progress"]["offset"] == 32
    run_experiment(Experiment(cfg, paused.output, resume=True))
    full = Experiment(cfg, tmp_path / "continuous")
    run_experiment(full)
    for seed in cfg.seed_values:
        for cell in ("concept", "independent", "no_feedback"):
            assert_same(
                load(paused.checkpoint_path(seed, cell)),
                load(full.checkpoint_path(seed, cell)),
            )
    for path in paused.output.glob("seed_*/*/*/*/predictions.pt"):
        assert tree_hash(load(path)) == tree_hash(
            load(full.output / path.relative_to(paused.output))
        )
    before = snapshot(paused.output)
    run_experiment(Experiment(cfg, paused.output, resume=True))
    assert snapshot(paused.output) == before
    with pytest.raises(ValueError, match="unchanged"):
        Experiment(replace(cfg, head_epochs=3), paused.output, resume=True)
    saved = paused.output / "seed_0/validation/independent/measured/predictions.pt"
    with saved.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(cfg, paused.output, resume=True)


def test_reuse_admission_and_recovery_order_guards(baseline, tmp_path):
    cfg = replace(config_for(baseline), seeds="0")
    shared = Experiment(cfg, tmp_path / "guards")
    job = Job(shared, 0, "concept")
    route = Route(job)
    with pytest.raises(InterruptedError):
        route.run(max_steps=1)
    saved = load(route.output / "resume.pt")
    damaged = deepcopy(saved)
    damaged["progress"]["order"] = damaged["progress"]["order"].flip(0)
    atomic_checkpoint(route.output / "resume.pt", damaged)
    with pytest.raises(ValueError, match="sample order"):
        Route(job)
    atomic_checkpoint(route.output / "resume.pt", saved)
    Route(job).run()
    initial = job.initial_for("concept")
    args: dict = dict(epochs=2, offset=0, count=64, batch=32, seed=0, cell="concept")
    admitted = verify_endpoint(job.checkpoint_path, initial, **args)
    assert admitted["progress"]["global_step"] == 4
    wrong_seed: dict = {**args, "seed": 1}
    wrong_epochs: dict = {**args, "epochs": 3}
    with pytest.raises(ValueError, match="sample order"):
        verify_endpoint(job.checkpoint_path, initial, **wrong_seed)
    with pytest.raises(ValueError, match="budget"):
        verify_endpoint(job.checkpoint_path, initial, **wrong_epochs)
    wrong = deepcopy(initial)
    wrong["model"]["frontend.q_params_rot"].add_(0.1)
    with pytest.raises(ValueError, match="initialization"):
        verify_endpoint(job.checkpoint_path, wrong, **args)
    with pytest.raises(ValueError, match="overlaps"):
        check_output(cfg, baseline.output / "child")
    with pytest.raises(ValueError, match="fresh-all"):
        replace(cfg, fresh_all=False).source()
    model = make_model(initial["model"])
    assert sum(p.numel() for p in model.parameters()) == 352
    assert all(p.is_cuda for p in model.parameters())
    assert len(module_state(initial["model"], "label_head")) == 5
