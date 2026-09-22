"""Use real frozen models, but never use real held-out images for development."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.grouped_dynamic_vqc.data import pooled_features
from experiments.grouped_dynamic_vqc.runtime import atomic_json, cuda_runtime, sha256
from experiments.grouped_feedback_ablation.protocol import load_checkpoint, read_json

from .. import protocol, results, runner
from ..data import read_test
from ..evaluation import sample_repeats, sampled_metrics
from ..protocol import CONDITIONS, Config, Sources, check_output
from ..runner import Experiment, run_experiment


def test_joint_sampling_keeps_correlation_invalid_codes_and_reproducibility():
    cuda_runtime(0)
    p = torch.zeros(64, 32)
    p[:, 0], p[:, 31] = 0.25, 0.75
    mass = torch.zeros_like(p)
    mass[:, 31] = 0.75
    raw = {
        "concept_probabilities": p,
        "branch_label_mass": mass,
        "concepts": torch.zeros(64, 2, dtype=torch.long),
        "labels": torch.ones(64),
    }
    config = Config(shots="64,4096", repeats=3, eval_batch_size=32)

    def tick(*_args, **_kwargs):
        return None

    a = sample_repeats(raw, config, ("validation", 0, "independent", "measured"), tick)
    b = sample_repeats(raw, config, ("validation", 0, "independent", "measured"), tick)
    for key in a["values"]:
        assert torch.equal(a["values"][key], b["values"][key])
    v = a["values"]
    budgets = torch.tensor(config.shot_list())[None, :, None]
    assert torch.equal(v["label_ones"], v["invalid_record_count"])
    assert torch.all(v["label_ones"] + v["true_record_count"] == budgets)
    assert not torch.equal(v["label_ones"][0], v["label_ones"][1])
    assert torch.all(v["label_ones"][:, 1] >= v["label_ones"][:, 0])
    assert float(v["label_ones"][:, 1].float().mean() / 4096) == pytest.approx(
        0.75, abs=0.003
    )
    rows = sampled_metrics(raw, a)
    assert all(r["concept"]["joint_map_accuracy"] == 0 for r in rows)
    assert all(r["concept"]["invalid_joint_map_fraction"] == 1 for r in rows)


def test_cuda_two_stage_pause_resume_and_no_real_test_access(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Real held-out data must never be read during development")

    monkeypatch.setattr(runner, "read_test", forbidden)
    monkeypatch.setattr(results, "plots", lambda *_: [])
    config = Config(
        seeds="0,1",
        shots="8,16",
        repeats=2,
        val_limit=18,
        development=True,
        eval_batch_size=18,
    )
    shared = Experiment(config, tmp_path / "paused")
    source_hashes = shared.sources.hashes.copy()
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_conditions=1)
    saved = shared.output / "validation/seed0/independent/measured/samples.pt"
    timestamp = saved.stat().st_mtime_ns
    assert not (shared.output / "test_access.json").exists()
    assert not (shared.output / "validation/result_lock.json").exists()
    resumed = Experiment(config, shared.output, resume=True)
    with pytest.raises(InterruptedError):
        run_experiment(resumed, max_conditions=17)
    assert (shared.output / "validation/result_lock.json").exists()
    assert not (shared.output / "protocol_lock.json").exists()
    resumed = Experiment(config, shared.output, resume=True)
    run_experiment(resumed)
    assert saved.stat().st_mtime_ns == timestamp
    assert read_json(shared.output / "summary.json")["test_evaluated"] is False
    assert read_json(shared.output / "test_access.json")["role"] == "test_proxy"
    assert not (shared.output / "test").exists()
    full = Experiment(config, tmp_path / "full")
    run_experiment(full)
    for role in ("validation", "test_proxy"):
        assert read_json(full.output / role / "summary.json") == read_json(
            shared.output / role / "summary.json"
        )
        for seed in config.seed_list():
            for training, mode in CONDITIONS:
                relative = Path(role) / f"seed{seed}" / training / mode / "samples.pt"
                a, b = (
                    load_checkpoint(shared.output / relative),
                    load_checkpoint(full.output / relative),
                )
                assert all(
                    torch.equal(a["values"][key], b["values"][key])
                    for key in a["values"]
                )
    for filename, digest in source_hashes.items():
        assert sha256(Path(filename)) == digest
    with pytest.raises(ValueError, match="unchanged config"):
        Experiment(replace(config, repeats=3), shared.output, resume=True)
    with saved.open("ab") as stream:
        stream.write(b"tamper")
    resumed.stage = "validation"
    with pytest.raises(ValueError, match="artifact changed"):
        resumed.verify_condition(0, "independent", "measured")


def test_formal_reader_uses_frozen_permutation_and_requires_seal(tmp_path):
    images = np.zeros((3, 8, 8), dtype=np.uint8)
    images[:, 2:5, 3:5] = 1
    path = tmp_path / "synthetic.npz"
    np.savez(
        path,
        imgs=images,
        c_int=np.array([[0, 0], [1, 1], [2, 2]]),
        y_task=np.array([0, 0, 1]),
        split=np.array(["train", "val", "test"]),
        raster_group_id=np.arange(3),
        source_index=np.arange(100, 103),
    )
    audit = {
        "dataset": str(path),
        "dataset_sha256": sha256(path),
        "routing_fit_role": "full original train images only",
        "permutation": list(range(39, -1, -1)),
        "test_row_count_metadata": 1,
        "angle_scale": np.pi,
    }
    gate, access = tmp_path / "protocol.json", tmp_path / "access.json"
    atomic_json(gate, {"manifest_sha256": "fixed", "development": False})
    atomic_json(access, {"protocol_sha256": sha256(gate), "role": "test"})
    result = read_test(audit, gate, access, "fixed")
    expected = (pooled_features(images[2:])[:, ::-1].reshape(-1, 10, 4) * np.pi).astype(
        np.float32
    )
    assert np.array_equal(result["angles"].numpy(), expected)
    assert result["source_index"].tolist() == [102]
    with pytest.raises(ValueError, match="seal"):
        read_test(audit, gate, access, "wrong")
    atomic_json(gate, {"manifest_sha256": "fixed", "development": True})
    with pytest.raises(ValueError, match="seal"):
        read_test(audit, gate, access, "fixed")


def test_wrong_sequential_pair_and_unsafe_configs_are_rejected(monkeypatch, tmp_path):
    original = protocol.verify_complete

    def wrong(*args, **kwargs):
        return {**original(*args, **kwargs), "initial_model_sha256": "mismatch"}

    monkeypatch.setattr(protocol, "verify_complete", wrong)
    with pytest.raises(ValueError, match="Sequential pairing mismatch"):
        Sources(Config(seeds="0", development=True))
    for config in (
        Config(seeds="0"),
        Config(val_limit=18),
        Config(shots="128,64"),
        Config(repeats=0),
    ):
        with pytest.raises(ValueError):
            config.validate()
    with pytest.raises(ValueError, match="isolated"):
        check_output(Config(), Path(Config().reference) / "new")
    assert not (tmp_path / "test").exists()


def test_aggregation_preserves_seed_units_and_negative_effects():
    rows = [
        {"training": "sequential", "seed": 0, "gain": -10.0},
        {"training": "sequential", "seed": 1, "gain": 2.0},
    ]
    value = results.aggregate(rows, ("training",), ("gain",))[0]
    assert value["n_seeds"] == 2
    assert value["gain"]["mean"] == -4
    assert value["gain"]["std"] == pytest.approx(12 / 2**0.5)
