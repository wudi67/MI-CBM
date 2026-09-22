"""Exercise real CUDA models; development never opens held-out test images."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from experiments.grouped_dynamic_vqc.runtime import sha256
from experiments.grouped_feedback_ablation.protocol import load_checkpoint, read_json
from experiments.grouped_feedback_ablation.runner import RouteRun as JointRun
from experiments.grouped_shots_final.data import subset
from experiments.grouped_shots_final.evaluation import infer
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .. import results, runner
from ..protocol import CONDITIONS, TRAININGS, Config, check_output
from ..runner import Experiment, run_experiment
from ..training import Adapter, Standard


def small_config() -> Config:
    return Config(
        seeds="0,1",
        epochs=2,
        train_limit=36,
        val_limit=18,
        batch_size=18,
        eval_batch_size=18,
        shots="8,16",
        repeats=2,
        checkpoint_steps=1,
        development=True,
    )


def test_cuda_joint_mid_epoch_resume_and_evaluation_recovery(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Development must never read held-out images")

    monkeypatch.setattr(runner, "read_test", forbidden)
    monkeypatch.setattr(results, "plots", lambda *_: [])
    cfg = small_config()
    paused = Experiment(cfg, tmp_path / "paused")
    sources_before = paused.sources.hashes.copy()
    # Standard takes four updates; stop Joint after its first minibatch.
    with pytest.raises(InterruptedError):
        run_experiment(paused, max_steps=5)
    path = paused.sources.model_path(0, "joint").parent / "resume.pt"
    checkpoint = load_checkpoint(path)
    assert checkpoint["progress"]["global_step"] == 1
    assert checkpoint["progress"]["offset"] == 18
    assert checkpoint["optimizer"]["state"]
    assert not (paused.output / "test_access.json").exists()
    resumed = Experiment(cfg, paused.output, resume=True)
    with pytest.raises(InterruptedError):
        run_experiment(resumed, max_conditions=1)
    saved = resumed.directory(0, "independent", "measured") / "samples.pt"
    timestamp = saved.stat().st_mtime_ns
    resumed = Experiment(cfg, paused.output, resume=True)
    run_experiment(resumed)
    full = Experiment(cfg, tmp_path / "full")
    run_experiment(full)
    assert saved.stat().st_mtime_ns == timestamp
    assert not (full.output / "test").exists()
    assert read_json(full.output / "summary.json")["test_evaluated"] is False
    assert read_json(full.output / "test_access.json")["role"] == "test_proxy"
    for seed in cfg.seed_list():
        for training in TRAININGS:
            a = load_checkpoint(full.sources.model_path(seed, training))
            b = load_checkpoint(resumed.sources.model_path(seed, training))
            assert state_hash(a["model"]) == state_hash(b["model"])
            for key in a["optimizer"]["state"]:
                for field in a["optimizer"]["state"][key]:
                    assert torch.equal(
                        a["optimizer"]["state"][key][field],
                        b["optimizer"]["state"][key][field],
                    )
            assert a["progress"] == b["progress"]
            checks = a["gradient_checks"]
            assert checks["device"].startswith("cuda")
            assert all(value > 0 for value in checks["gradient_l2"].values())
    for role in ("validation", "test_proxy"):
        assert read_json(full.output / role / "summary.json") == read_json(
            resumed.output / role / "summary.json"
        )
        for seed in cfg.seed_list():
            for training, mode in CONDITIONS:
                relative = Path(role) / f"seed{seed}" / training / mode / "samples.pt"
                a, b = (
                    load_checkpoint(full.output / relative),
                    load_checkpoint(resumed.output / relative),
                )
                assert all(
                    torch.equal(a["values"][k], b["values"][k]) for k in a["values"]
                )
        # Distinct Joint/Standard learned frontends must never share cached states.
        base = full.output / role / "seed0"
        a = load_checkpoint(base / "joint/measured/joint.pt")["concept_probabilities"]
        b = load_checkpoint(base / "standard/measured/joint.pt")[
            "concept_probabilities"
        ]
        assert not torch.equal(a, b)
    for name, digest in sources_before.items():
        assert sha256(Path(name)) == digest
    with pytest.raises(ValueError, match="unchanged config"):
        Experiment(replace(cfg, epochs=3), resumed.output, resume=True)
    with saved.open("ab") as stream:
        stream.write(b"tamper")
    resumed.stage = "validation"
    with pytest.raises(ValueError, match="artifact changed"):
        resumed.verify_condition(0, "independent", "measured")


def test_standard_gradient_does_not_use_concept_targets(tmp_path):
    shared = Experiment(small_config(), tmp_path / "objective")
    data = {
        role: {
            k: v.cuda()
            for k, v in subset(shared.sources.baseline.data[role], 18).items()
        }
        for role in ("train", "val")
    }
    # The original BCE implementation may monitor concepts but must not train on them.
    adapter: Any = Adapter(shared, 0, "standard", data)
    a = Standard(adapter, "standard")
    adapter.output = shared.output / "second"
    b = Standard(adapter, "standard")
    indices = torch.arange(18, device="cuda")
    a.train_step(indices)
    data["train"]["concepts"] = (data["train"]["concepts"] + 1) % torch.tensor(
        [3, 6], device="cuda"
    )
    b.train_step(indices)
    assert state_hash(a.model.state_dict()) == state_hash(b.model.state_dict())


def test_formal_validation_import_preserves_old_observations(tmp_path):
    shared = Experiment(Config(), tmp_path / "import")
    shared.stage = "validation"
    shared.prepare_data()
    assert shared.import_condition(0, "independent", "measured")
    old = Path(shared.config.formal_reference) / "validation/seed0/independent/measured"
    new = shared.directory(0, "independent", "measured")
    for name in ("joint.pt", "samples.pt"):
        assert sha256(old / name) == sha256(new / name)
    metric = shared.verify_condition(0, "independent", "measured")
    assert metric is not None
    assert metric["repeats"] == read_json(old / "evaluation.json")["repeats"]
    assert not (shared.output / "test_access.json").exists()
    assert shared.sources.model_path(0, "standard").is_relative_to(
        Path(Config().standard_reference)
    )
    assert shared.sources.model_path(1, "standard").is_relative_to(shared.output)
    assert not shared.import_condition(0, "joint", "measured")
    # Full validation must reproduce the reused seed-0 training endpoints.
    for training, mode in (
        ("standard", "measured"),
        ("joint", "measured"),
        ("joint_no_feedback", "zero"),
    ):
        states = shared.states(0, training)
        model = shared.sources.make_model(
            load_checkpoint(shared.sources.model_path(0, training))["model"]
        )
        raw = infer(
            model, shared.data, states, mode, shared.config.eval_batch_size, shared.tick
        )
        shared.compare_validation(raw, 0, training, mode)


def test_default_batch_runs_all_three_objectives_on_cuda(tmp_path):
    config = replace(small_config(), seeds="1", train_limit=0, batch_size=1024)
    shared = Experiment(config, tmp_path / "batch1024")
    data = shared.sources.baseline.data
    indices = torch.arange(1024, device="cuda")
    for training in TRAININGS:
        adapter: Any = Adapter(shared, 1, training, data)
        if training == "standard":
            route = Standard(adapter, "standard")
        else:
            variant = "feedback" if training == "joint" else "no_feedback"
            route = JointRun(adapter, f"joint/seed1/{variant}")
        route.train_step(indices)
        assert all(p.is_cuda for p in route.model.parameters())
        assert all(value > 0 for value in route.gradient_checks["gradient_l2"].values())
        assert (
            torch.cuda.max_memory_allocated()
            < torch.cuda.get_device_properties(0).total_memory
        )
        del route


def test_fixed_budgets_and_correct_joint_comparator():
    for cfg in (
        Config(epochs=2),
        Config(batch_size=512),
        Config(seeds="0"),
        Config(shots="64"),
        Config(train_limit=36),
    ):
        with pytest.raises(ValueError):
            cfg.validate()
    for source in (Config().standard_reference, Config().formal_reference):
        with pytest.raises(ValueError, match="isolated"):
            check_output(Config(), Path(source) / "new")
    rows = []
    for seed in (0, 1):
        for training, mode, accuracy in (
            ("standard", "measured", 0.95),
            ("no_feedback", "zero", 0.1),
            ("joint_no_feedback", "zero", 0.8),
            ("joint", "measured", 0.6),
            ("joint", "shape", 0.7),
            ("joint", "scale", 0.5),
            ("joint", "both", 0.4),
        ):
            rows.append(
                {
                    "seed": seed,
                    "training": training,
                    "control_mode": mode,
                    "role": "validation",
                    "shots": 0,
                    "label_accuracy": accuracy,
                    "label_balanced_accuracy": accuracy,
                    "label_bce": 1 - accuracy,
                }
            )
    effects = results.paired_rows(rows)
    assert len(effects) == 8
    assert all(r["training"] == "joint" for r in effects)
    feedback = [r for r in effects if r["effect"] == "feedback"]
    assert all(r["accuracy_gain_pp"] == pytest.approx(-20) for r in feedback)
