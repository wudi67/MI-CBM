"""Joint measurement counts, read-only CUDA inference and gated test access."""

import csv
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace

import pytest
import torch

from experiments.grouped_dynamic_vqc.model import sample_shots
from experiments.grouped_dynamic_vqc.runtime import atomic_json, cuda_runtime, sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_four_modes import results as training_results
from experiments.grouped_robot_four_modes.protocol import Config as TrainingConfig
from experiments.grouped_robot_four_modes.runner import Experiment as TrainingExperiment
from experiments.grouped_robot_four_modes.runner import run_experiment as run_training
from experiments.grouped_robot_pilot.data import pooled
from experiments.grouped_robot_pilot.model import bits
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_shots_final.evaluation import draw_seed
from robot_dataset_schema import SPLIT_FILES

from .. import results
from ..data import read_test
from ..evaluation import sample_repeats, sampled_metrics
from ..protocol import Config, check_output
from ..runner import Experiment, run_experiment

pytest_plugins = ["experiments.grouped_robot_four_modes.tests.test_four_modes"]


@pytest.fixture(name="models")
def completed_four_modes(reference, tmp_path, monkeypatch):
    monkeypatch.setattr(training_results, "plots", lambda *_: None)
    config = TrainingConfig(
        reference=str(reference.output),
        seeds="0,1",
        epochs=4,
        checkpoint_steps=1,
        development=True,
    )
    shared = TrainingExperiment(config, tmp_path / "four_modes")
    run_training(shared)
    return shared


def config_for(models):
    return Config(
        reference=str(models.output),
        seeds="0,1",
        shots="8,16",
        repeats=2,
        eval_batch_size=64,
        val_limit=32,
        development=True,
    )


def snapshot(root):
    return {
        str(p.relative_to(root)): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file() and p.name != "heartbeat.json"
    }


def test_joint_sampler_bit_order_correlation_prefixes_and_replay():
    cuda_runtime(0)
    config = Config(shots="8,16,32", repeats=3, eval_batch_size=32, development=True)
    table = bits(torch.device("cpu"))
    # Deterministic complete concept code and parity label for every legal code.
    p = torch.eye(32)
    labels = (table.sum(1) % 2).float()
    raw = {
        "concept_probabilities": p,
        "branch_label_mass": p * labels[:, None],
        "concepts": table,
        "labels": labels,
    }
    identity = ("validation", 0, "independent", "measured")
    samples = sample_repeats(raw, config, identity, lambda *_a, **_k: None)
    for row in sampled_metrics(raw, samples):
        assert row["label"]["accuracy"] == 1
        assert row["concept"]["mean_bit_accuracy"] == 1
        assert row["concept"]["joint_map_accuracy"] == 1
        assert row["concept"]["joint_single_shot_probability"] == 1
    # Perfectly correlated m=00000/y=0 or m=11111/y=1; independent sampling
    # of record and label would violate the exact count equalities below.
    p = torch.zeros(32, 32)
    p[:, 0] = 0.5
    p[:, 31] = 0.5
    one = torch.zeros_like(p)
    one[:, 31] = 0.5
    raw.update(concept_probabilities=p, branch_label_mass=one)
    first = sample_repeats(raw, config, identity, lambda *_a, **_k: None)
    second = sample_repeats(raw, config, identity, lambda *_a, **_k: None)
    assert tree_hash(first) == tree_hash(second)
    assert torch.equal(
        first["values"]["bit_ones"],
        first["values"]["label_ones"][..., None].expand(-1, -1, -1, 5),
    )
    assert torch.all(
        first["values"]["bit_ones"][:, 1:] >= first["values"]["bit_ones"][:, :-1]
    )
    generator = torch.Generator(device="cuda").manual_seed(
        draw_seed(config.sampling_seed, *identity, 0)
    )
    measured, y = sample_shots(
        {"concept_probs": p.cuda(), "branch_label_mass": one.cuda()}, 32, generator
    )
    for j, shots in enumerate(config.shot_list()):
        assert torch.equal(
            first["values"]["label_ones"][0, j], y[:, :shots].sum(1).cpu().int()
        )
        count = torch.nn.functional.one_hot(measured[:, :shots], 32).sum(1)
        assert torch.equal(
            first["values"]["joint_prediction"][0, j], count.argmax(1).cpu().int()
        )
    assert len(set(first["seeds"])) == 3 and first["device"] == "cuda:0"
    bad = deepcopy(first)
    bad["values"]["bit_ones"][0, 0, 0, 0] = 9
    with pytest.raises(ValueError, match="counts"):
        sampled_metrics(raw, bad)


def test_read_only_cuda_pipeline_resume_and_complete_statistics(
    models, tmp_path, monkeypatch
):
    before = snapshot(models.output)
    cfg = config_for(models)

    # Evaluation must not create or step any optimizer.
    def no_optimizer(*_args, **_kwargs):
        raise AssertionError("Evaluation tried to create an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", no_optimizer)
    shared = Experiment(cfg, tmp_path / "paused")
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_conditions=1)
    assert not (shared.output / "protocol_lock.json").exists()
    first = shared.output / "validation/seed0/standard/measured/samples.pt"
    timestamp = first.stat().st_mtime_ns
    # Finish validation; the pause must happen before the final data gate opens.
    with pytest.raises(InterruptedError):
        run_experiment(Experiment(cfg, shared.output, resume=True), max_conditions=17)
    assert (shared.output / "validation/result_lock.json").exists()
    assert not (shared.output / "test_access.json").exists()
    with pytest.raises(InterruptedError):
        run_experiment(Experiment(cfg, shared.output, resume=True), max_conditions=1)
    assert read_json(shared.output / "test_access.json")["role"] == "test_proxy"
    run_experiment(Experiment(cfg, shared.output, resume=True))
    assert first.stat().st_mtime_ns == timestamp
    summary = read_json(shared.output / "summary.json")
    assert summary["completed_conditions"] == 36
    assert not summary["test_read"] and not summary["test_evaluated"]
    assert not (Path(models.pilot_config.dataset) / SPLIT_FILES["test"]).exists()
    assert (shared.output / "test_proxy/classification_shots.pdf").stat().st_size > 1000
    for role in ("validation", "test_proxy"):
        stage = read_json(shared.output / role / "summary.json")
        averages = list(
            csv.DictReader((shared.output / role / "condition_results.csv").open())
        )
        repeated = list(
            csv.DictReader((shared.output / role / "sampling_repeats.csv").open())
        )
        for aggregate in stage["conditions"]:
            values = [
                float(r["label_accuracy"])
                for r in averages
                if (r["training"], r["control_mode"], int(r["shots"]))
                == (
                    aggregate["training"],
                    aggregate["control_mode"],
                    aggregate["shots"],
                )
            ]
            assert aggregate["n_seeds"] == 2
            assert aggregate["label_accuracy"]["mean"] == mean(values)
            assert aggregate["label_accuracy"]["std"] == stdev(values)
            if aggregate["training"] == "standard":
                assert aggregate["concept_mean_bit_accuracy"] is None
        for row in averages:
            if int(row["shots"]):
                values = [
                    float(r["label_accuracy"])
                    for r in repeated
                    if all(
                        r[k] == row[k]
                        for k in ("seed", "training", "control_mode", "shots")
                    )
                ]
                assert float(row["label_accuracy"]) == mean(values)
        for seed in (0, 1):
            base = shared.output / role / f"seed{seed}"
            a = load(base / "joint/measured/joint.pt")
            b = load(base / "joint/correct_all_five/joint.pt")
            assert torch.equal(a["concept_probabilities"], b["concept_probabilities"])
    assert snapshot(models.output) == before
    assert not list(shared.output.rglob("endpoint.pt"))
    monkeypatch.setattr(results, "plots", lambda *_: [])
    full = Experiment(cfg, tmp_path / "continuous")
    run_experiment(full)
    for path in shared.output.glob("*/seed*/*/*/*.pt"):
        assert tree_hash(load(path)) == tree_hash(
            load(full.output / path.relative_to(shared.output))
        )
    done = snapshot(shared.output)
    run_experiment(Experiment(cfg, shared.output, resume=True))
    assert done == snapshot(shared.output)
    with pytest.raises(ValueError, match="unchanged"):
        Experiment(replace(cfg, repeats=3), shared.output, resume=True)
    with pytest.raises(ValueError, match="overlaps"):
        check_output(cfg, models.output / "child")
    with first.open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(cfg, shared.output, resume=True).verify_report_lock(
            shared.output / "validation"
        )


def seal_for_test(output: Path, manifest_hash: str):
    validation = output / "validation"
    atomic_json(validation / "summary.json", {"status": "complete"})
    atomic_json(
        validation / "result_lock.json",
        {
            "manifest_sha256": manifest_hash,
            "artifacts": {"summary.json": sha256(validation / "summary.json")},
        },
    )
    atomic_json(
        output / "protocol_lock.json",
        {
            "manifest_sha256": manifest_hash,
            "development": False,
            "validation_result_lock_sha256": sha256(validation / "result_lock.json"),
        },
    )
    atomic_json(
        output / "test_access.json",
        {
            "protocol_sha256": sha256(output / "protocol_lock.json"),
            "role": "test",
            "real_test_values_requested": True,
        },
    )


def test_test_reader_gate_preserves_observed_rows_and_frozen_routing(models, tmp_path):
    pilot = models.reference.reference.reference
    source = SimpleNamespace(
        dataset=Path(models.pilot_config.dataset),
        audit=pilot.audit,
        preprocessing_path=pilot.output / "preprocessing.json",
    )
    output = tmp_path / "reader"

    def tick(*_args, **_kwargs):
        return None

    # Missing seal must fail before attempting to read the absent test CSV.
    with pytest.raises(FileNotFoundError, match="protocol_lock"):
        read_test(source, output, "test_manifest", tick, expected_rows=128)
    seal_for_test(output, "test_manifest")
    rows = list(csv.DictReader((source.dataset / SPLIT_FILES["val"]).open()))
    for row in rows:
        old = source.dataset / "robot_images" / Path(row["image_path"]).name
        target = source.dataset / "robot_images" / ("test_" + old.name)
        target.write_bytes(old.read_bytes())
        row["image_path"] = target.name
        row["robot_id"] = str(int(row["robot_id"]) + 32)
        row["label"] = str(
            1 - int(row["label"])
        )  # Preserve observed values, not an oracle rule.
    table = source.dataset / SPLIT_FILES["test"]

    def save_rows():
        with table.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    save_rows()
    data, inputs = read_test(source, output, "test_manifest", tick, expected_rows=128)
    assert len(data["labels"]) == 128 and not inputs["routing_refitted"]
    assert inputs["test_development_identity_overlap"] == 0
    assert data["labels"].tolist() == [int(r["label"]) for r in rows]
    feature, _ = pooled(
        [str(source.dataset / "robot_images" / rows[0]["image_path"])], tick
    )
    expected = torch.from_numpy(
        feature[:, source.audit["permutation"]].reshape(1, 10, 4)
        * source.audit["angle_scale"]
    )
    torch.testing.assert_close(data["angles"][:1], expected)
    assert data["source_index"].tolist() == list(range(128))
    # A complete test identity moved to train must be rejected, even if its
    # four renders and annotations are internally valid.
    for row in rows[:4]:
        row["robot_id"] = "0"
    save_rows()
    with pytest.raises(ValueError, match="overlap"):
        read_test(source, output, "test_manifest", tick, expected_rows=128)
    gate = read_json(output / "protocol_lock.json")
    gate["development"] = True
    atomic_json(output / "protocol_lock.json", gate)
    with pytest.raises(ValueError, match="seal"):
        read_test(source, output, "test_manifest", tick, expected_rows=128)
