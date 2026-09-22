"""Verify branch interventions, Independent inputs, provenance and CUDA recovery."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.grouped_dynamic_vqc.runtime import cuda_runtime, rng_state, sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_label_continuation.protocol import Config as LabelConfig
from experiments.grouped_robot_label_continuation.runner import (
    Experiment as LabelExperiment,
)
from experiments.grouped_robot_label_continuation.runner import (
    run_experiment as run_labels,
)
from experiments.grouped_robot_pilot.model import bits
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash

from ..model import ConceptMLP, marginalize, predict, transitions
from ..protocol import Config, check_output
from ..runner import Experiment, run_experiment
from ..training import Route, verify_complete

pytest_plugins = [
    "experiments.grouped_robot_label_continuation.tests.test_label_continuation"
]


def test_all_codes_and_masks_preserve_original_born_weights():
    table = torch.linspace(0.01, 0.99, 32, dtype=torch.float64)
    concepts = bits(torch.device("cpu"))
    probability = torch.arange(1, 1025, dtype=torch.float64).reshape(32, 32)
    probability /= probability.sum(1, keepdim=True)
    for mask in range(32):
        mass = marginalize(table, probability, concepts, mask)
        expected = torch.empty_like(mass)
        for truth_code in range(32):
            for measured in range(32):
                code = measured
                # Independent per-bit oracle: no calls to the production control helper.
                for bit_index in range(5):
                    bit_mask = 2 ** (4 - bit_index)
                    if mask & bit_mask:
                        code += (
                            int(concepts[truth_code, bit_index])
                            - ((measured >> (4 - bit_index)) & 1)
                        ) * bit_mask
                expected[truth_code, measured] = (
                    probability[truth_code, measured] * table[code]
                )
        torch.testing.assert_close(mass, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        marginalize(table, probability, concepts, 31).sum(1), table
    )


def test_weighted_probabilities_not_map_or_weighted_logits():
    table = torch.full((32,), 0.9)
    table[0] = 0.01
    probability = torch.zeros(1, 32)
    probability[0, 0], probability[0, 31] = 0.6, 0.4
    answer = marginalize(table, probability, torch.zeros(1, 5).long(), 0).sum()
    assert float(answer) == pytest.approx(0.366)
    assert float(answer) != pytest.approx(float(table[0]))
    mixed_logits = (probability * table.logit()).sum().sigmoid()
    assert abs(float(answer - mixed_logits)) > 0.1


def test_transitions_account_for_accuracy_change():
    result = transitions(
        torch.tensor([0.1, 0.1, 0.7, 0.8]),
        torch.tensor([0.7, 0.6, 0.9, 0.1]),
        torch.tensor([0, 1, 1, 0]),
    )
    assert result == {
        "both_correct": 1,
        "correct_to_wrong": 1,
        "wrong_to_correct": 2,
        "both_wrong": 0,
        "net_correct": 1,
        "accuracy_change_pp": 25.0,
    }


def stub_shared(output: Path, *, disturb: bool = False):
    cuda_runtime(0)
    model = ConceptMLP().cuda()
    initial = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "rng": rng_state(),
    }
    concepts = bits(torch.device("cpu")).repeat(2, 1)
    labels = ((concepts[:, 0] + concepts[:, 1]) > 0).float()
    probabilities = torch.full((64, 32), 1 / 32)
    cpu_data = {
        role: {
            "concepts": concepts.clone(),
            "labels": labels.clone(),
            "concept_probabilities": probabilities.clone(),
        }
        for role in ("train", "validation")
    }
    if disturb:
        cpu_data["train"]["concept_probabilities"] = torch.eye(32).repeat(2, 1)
        cpu_data["validation"]["labels"] = 1 - labels
        cpu_data["validation"]["concepts"] = 1 - concepts
    return SimpleNamespace(
        config=Config(
            epochs=3,
            batch_size=16,
            diagnostic_every=1,
            checkpoint_steps=1,
            development=True,
        ),
        output=output,
        data={
            role: {k: v.cuda() for k, v in data.items()}
            for role, data in cpu_data.items()
        },
        cpu_data=cpu_data,
        control={"stop": False},
        initial=initial,
        initial_hash=state_hash(initial["model"]),
        manifest_hash="manifest",
        data_hash="data",
        order_seed=0,
        order_offset=100,
        heartbeat=lambda *_args, **_kwargs: None,
    )


def test_cuda_training_is_independent_of_predictions_and_validation(tmp_path):
    first = stub_shared(tmp_path / "first")
    altered = stub_shared(tmp_path / "altered", disturb=True)
    Route(first).run()
    Route(altered).run()
    a, b = (load(s.output / "training/endpoint.pt") for s in (first, altered))
    assert state_hash(a["model"]) == state_hash(b["model"])
    assert tree_hash(a["optimizer"]) == tree_hash(b["optimizer"])
    assert a["gradient_check"]["input_device"] == "cuda:0"
    assert a["gradient_check"]["parameters"] == 113
    assert a["gradient_check"]["gradient_norm"] > 0
    verify_complete(first)


def test_cuda_mid_epoch_resume_exactly_matches_continuous(tmp_path):
    continuous = stub_shared(tmp_path / "continuous")
    paused = stub_shared(tmp_path / "paused")
    Route(continuous).run()
    with pytest.raises(InterruptedError):
        Route(paused).run(max_steps=3)
    saved = load(paused.output / "training/resume.pt")
    assert saved["progress"]["epoch"] == 0
    assert saved["progress"]["offset"] == 48
    Route(paused).run()
    a, b = (load(s.output / "training/endpoint.pt") for s in (continuous, paused))
    for key in ("model", "optimizer", "progress", "rng"):
        assert tree_hash(a[key]) == tree_hash(b[key])


def snapshot(root: Path) -> dict:
    return {
        str(p): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_real_vqc_source_audit_cache_correction_and_immutable_resume(
    source, tmp_path, monkeypatch
):
    from .. import results

    monkeypatch.setattr(results, "plots", lambda *_: None)
    reference = LabelExperiment(
        LabelConfig(
            reference=str(source.output),
            head_epochs=5,
            diagnostic_every=2,
            development=True,
        ),
        tmp_path / "label_source",
    )
    run_labels(reference)
    protected = [reference.output, source.output, source.reference.output]
    before = [snapshot(p) for p in protected]
    config = Config(
        reference=str(reference.output),
        epochs=2,
        batch_size=32,
        diagnostic_every=1,
        development=True,
    )
    output = tmp_path / "mlp"
    shared = Experiment(config, output)
    with pytest.raises(InterruptedError):
        run_experiment(shared, max_steps=1)
    resumed = Experiment(config, output, resume=True)
    run_experiment(resumed)
    assert [snapshot(p) for p in protected] == before
    assert read_json(output / "heartbeat.json")["status"] == "complete"
    assert set(load(output / "concept_cache.pt")) == {"train", "validation"}
    assert read_json(output / "frontend_cuda_check.json")["device"] == "cuda:0"
    assert not read_json(output / "summary.json")["test_evaluated"]
    raw = load(output / "validation/predictions.pt")
    torch.testing.assert_close(
        raw["branch_label_mass"]["correct_all_five"].sum(1),
        raw["direct_true_probability"],
        atol=2e-5,
        rtol=0,
    )
    # Complete resume must reverify without changing any locked result artifact.
    locked_before = read_json(output / "result_lock.json")
    run_experiment(Experiment(config, output, resume=True))
    assert read_json(output / "result_lock.json") == locked_before
    with pytest.raises(ValueError, match="historical"):
        check_output(config, reference.output / "overwrite")
    with pytest.raises(ValueError, match="source/data"):
        check_output(config, Path(__file__).resolve().parent)
    (output / "validation/evaluation.json").write_text("{}")
    with pytest.raises(ValueError, match="Locked artifact changed"):
        Experiment(config, output, resume=True)


def test_prediction_uses_32_binary_inputs_on_cuda():
    cuda_runtime(0)
    model = ConceptMLP().cuda()
    data = {
        "concepts": bits(torch.device("cuda")),
        "concept_probabilities": torch.eye(32, device="cuda"),
    }
    raw = predict(model, data)
    torch.testing.assert_close(
        raw["branch_label_mass"]["measured"].sum(1), raw["direct_true_probability"]
    )
    with pytest.raises(ValueError, match="five explicit"):
        model(torch.zeros(2, 40, device="cuda"))
