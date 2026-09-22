"""Real CUDA circuits with synthetic images; no real test split is required."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import torchquantum as tq
import torchquantum.functional as tqf
from experiments.dynamic_vqc.robot.data import center_gray
from experiments.grouped_dynamic_vqc.model import controlled_branches
from experiments.grouped_dynamic_vqc.runtime import cuda_runtime
from experiments.grouped_vqc_training_modes.protocol import state_hash
from robot_dataset_schema import ROBOT_CONCEPT_NAMES

from .. import results
from ..data import pooled
from ..evaluation import concept_metrics
from ..model import RobotVQC, bits, codes, concept_loss, controls, forward_control
from ..protocol import CELLS, CONDITIONS, Config, check_output, read_json
from ..runner import Experiment, run_experiment
from ..training import Route, load


@pytest.fixture(name="dataset")
def robot_dataset(tmp_path):
    root = tmp_path / "robot"
    (root / "robot_images").mkdir(parents=True)
    for role, offset in (("train", 0), ("val", 32)):
        rows = []
        for code in range(32):
            values = [(code >> bit) & 1 for bit in range(4, -1, -1)]
            for render in range(4):
                pixels = np.full((32, 32), 255, dtype=np.uint8)
                for i, value in enumerate(values):
                    pixels[4 + 4 * i : 7 + 4 * i, 5 : 13 + 3 * value] = (
                        30 + 20 * i + render
                    )
                name = f"{role}_{code}_{render}.png"
                Image.fromarray(pixels).save(root / "robot_images" / name)
                # Keep observed label differences within the same concept tuple.
                label = (values[0] + values[1] + (render == 3)) % 2
                rows.append(
                    {
                        "image_path": name,
                        "robot_id": offset + code,
                        "render_id": render,
                        **dict(zip(ROBOT_CONCEPT_NAMES, values, strict=True)),
                        "label": label,
                        "class": "glorp" if label else "drent",
                    }
                )
        with (root / f"robot_images_{role}_labels.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    # There is deliberately no test CSV and no test image.
    return root


def config_for(dataset: Path) -> Config:
    return Config(
        dataset=str(dataset),
        development=True,
        concept_epochs=2,
        head_epochs=2,
        batch_size=32,
        eval_batch_size=64,
        train_limit=64,
        val_limit=32,
        checkpoint_steps=1,
    )


def test_robot_bit_order_and_all_32_codes_are_legal():
    table = bits(torch.device("cpu"))
    assert codes(table).tolist() == list(range(32))
    assert codes(torch.tensor([[1, 0, 0, 0, 1]])).item() == 17
    for mask in range(32):
        actual = controls(table, mask=mask)
        for row in range(32):
            for measured in range(32):
                expected = [
                    (
                        table[row, bit]
                        if mask & (1 << (4 - bit))
                        else table[measured, bit]
                    ).item()
                    for bit in range(5)
                ]
                assert table[actual[row, measured]].tolist() == expected
    identity = torch.eye(32)
    metric = concept_metrics(identity, table)
    assert metric["all_concepts_accuracy"] == metric["joint_map_accuracy"] == 1
    assert metric["joint_single_shot_probability"] == 1
    assert float(concept_loss(identity, table)) == 0
    assert torch.count_nonzero(controls(table, zero=True)) == 0


def test_correction_matches_explicit_x_gates_and_preserves_branch_weights():
    cuda_runtime(0)
    model = RobotVQC().cuda()
    state = torch.randn(2, 1024, dtype=torch.complex64, device="cuda")
    state = state / state.norm(dim=1, keepdim=True)
    targets = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], device="cuda")
    original = state.reshape(2, 32, 32)
    for zero, mask in ((False, 0), (False, 16), (False, 1), (False, 31), (True, 0)):
        record = controls(targets, zero=zero, mask=mask)
        changed = controlled_branches(state, record)
        references = []
        for row in range(2):
            trajectories = []
            for measured in range(32):
                qdev = tq.QuantumDevice(n_wires=5, device="cuda")
                qdev.set_states(original[row, measured][None])
                for bit in range(5):
                    if int(record[row, measured]) & (1 << (4 - bit)):
                        tqf.paulix(qdev, wires=bit)
                trajectories.append(qdev.get_states_1d()[0])
            references.append(torch.stack(trajectories))
        torch.testing.assert_close(changed, torch.stack(references))
        fast = forward_control(model, state, targets, zero=zero, mask=mask)
        direct = model.from_state(state, record, direct=True)
        torch.testing.assert_close(
            fast["label_prob"], direct["label_prob"], atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(
            fast["concept_probs"], original.abs().square().sum(-1)
        )


def test_preprocessing_preserves_grayscale_and_training_only_routing(dataset, tmp_path):
    cfg = config_for(dataset)
    shared = Experiment(cfg, tmp_path / "prepared")
    assert shared.data["train"]["angles"].is_cuda
    assert shared.data["train"]["angles"].shape == (64, 10, 4)
    assert shared.audit["routing_fit_rows"] == 128
    assert not shared.audit["test_read"]
    images = torch.zeros(1, 32, 32)
    images[0, 2:5, 1:7] = 0.4
    centered = center_gray(images)
    assert torch.allclose(centered.sum(), images.sum())
    assert (centered == 0.4).sum() == 18
    features, _ = pooled(
        [str(p) for p in sorted((dataset / "robot_images").glob("train*"))],
        lambda *_a, **_k: None,
    )
    variance = features.astype(np.float64).var(0)
    ranked = sorted(range(40), key=lambda i: (-variance[i], i))
    assert shared.audit["permutation"] == [
        i for wire in range(10) for i in ranked[wire::10]
    ]
    first = shared.data["train"]["source_index"][0].item()
    path = dataset / "robot_images" / f"train_{first // 4}_{first % 4}.png"
    feature, _ = pooled([str(path)], lambda *_a, **_k: None)
    expected = torch.from_numpy(
        (feature[:, shared.audit["permutation"]].reshape(1, 10, 4) * np.pi).astype(
            np.float32
        )
    )
    torch.testing.assert_close(shared.data["train"]["angles"][:1].cpu(), expected)
    # Validation pixels cannot change the fitted feature assignment.
    for p in (dataset / "robot_images").glob("val*"):
        with Image.open(p) as image:
            pixels = np.array(image)
        pixels[3:25, 20:23] = 110
        Image.fromarray(pixels).save(p)
    changed = Experiment(cfg, tmp_path / "changed_val")
    assert changed.audit["permutation"] == shared.audit["permutation"]
    assert changed.audit["variances"] == shared.audit["variances"]
    with pytest.raises(ValueError, match="artifact changed"):
        Experiment(cfg, shared.output, resume=True)


def test_cuda_pipeline_mid_epoch_resume_and_no_test_access(
    dataset, tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "plots", lambda *_: [])
    cfg = config_for(dataset)
    paused = Experiment(cfg, tmp_path / "paused")
    with pytest.raises(InterruptedError):
        run_experiment(paused, max_steps=5)
    partial = load(paused.output / "training/independent/resume.pt")
    assert partial["progress"]["global_step"] == 1
    assert partial["progress"]["offset"] == 32
    resumed = Experiment(cfg, paused.output, resume=True)
    with pytest.raises(InterruptedError):
        run_experiment(resumed, max_conditions=1)
    saved = resumed.output / "validation/independent/measured/predictions.pt"
    timestamp = saved.stat().st_mtime_ns
    resumed = Experiment(cfg, resumed.output, resume=True)
    run_experiment(resumed)
    full = Experiment(cfg, tmp_path / "full")
    run_experiment(full)
    assert saved.stat().st_mtime_ns == timestamp
    assert read_json(full.output / "summary.json") == read_json(
        resumed.output / "summary.json"
    )
    for cell in CELLS:
        a = load(full.output / "training" / cell / "endpoint.pt")
        b = load(resumed.output / "training" / cell / "endpoint.pt")
        assert state_hash(a["model"]) == state_hash(b["model"])
        assert a["progress"] == b["progress"]
        for index in a["optimizer"]["state"]:
            for field in a["optimizer"]["state"][index]:
                assert torch.equal(
                    a["optimizer"]["state"][index][field],
                    b["optimizer"]["state"][index][field],
                )
        frozen = "label_head" if cell == "concept" else "frontend"
        active = "frontend" if cell == "concept" else "label_head"
        assert a["gradient_checks"]["gradient_l2"][frozen] is None
        assert a["gradient_checks"]["gradient_l2"][active] > 0
    for cell, name, _ in CONDITIONS:
        relative = Path("validation") / cell / name / "predictions.pt"
        a, b = load(full.output / relative), load(resumed.output / relative)
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert not (full.output / "test").exists()
    assert read_json(full.output / "summary.json")["test_evaluated"] is False
    assert (
        read_json(full.output / "training/independent/result.json")["training_control"]
        == "true"
    )
    with pytest.raises(ValueError, match="unchanged config"):
        Experiment(replace(cfg, learning_rate=0.02), resumed.output, resume=True)
    with saved.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        resumed.verify_condition("independent", "measured", 0)


def test_default_batch_cuda_and_concept_loss_ignores_task_labels(dataset, tmp_path):
    cfg = config_for(dataset)
    shared = Experiment(cfg, tmp_path / "batch1024")
    # Build a genuine 1024-row CUDA batch from the synthetic images.
    shared.data["train"] = {
        k: v.repeat((16,) + (1,) * (v.ndim - 1))
        for k, v in shared.data["train"].items()
    }
    a = Route(shared, "concept")
    original = shared.output
    shared.output = tmp_path / "second_objective"
    b = Route(shared, "concept")
    indices = torch.arange(1024, device="cuda")
    a.step(indices)
    shared.data["train"]["labels"] = 1 - shared.data["train"]["labels"]
    b.step(indices)
    assert state_hash(a.model.state_dict()) == state_hash(b.model.state_dict())
    assert all(p.is_cuda for p in a.model.parameters())
    assert (
        torch.cuda.max_memory_allocated()
        < torch.cuda.get_device_properties(0).total_memory
    )
    shared.output = original
    for mask, zero in ((31, False), (0, True)):
        model = shared.make_model(a.model.state_dict())
        model.frontend.requires_grad_(False)
        states = model.frontend(shared.data["train"]["angles"]).detach()
        output = forward_control(
            model, states, shared.data["train"]["concepts"], mask=mask, zero=zero
        )
        torch.nn.functional.binary_cross_entropy(
            output["label_prob"].clamp(1e-7, 1 - 1e-7), shared.data["train"]["labels"]
        ).backward()
        assert any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in model.label_head.parameters()
        )
        assert all(p.grad is None for p in model.frontend.parameters())


def test_unsafe_outputs_and_wrong_pilot_budgets_are_rejected(tmp_path):
    for cfg in (
        Config(seed=1),
        Config(concept_epochs=2),
        Config(train_limit=32),
        Config(batch_size=0),
    ):
        with pytest.raises(ValueError):
            cfg.validate()
    for path in (
        Path(Config().dataset) / "new",
        Path("experiments/new"),
        Path("outputs/grouped_four_modes/new"),
    ):
        with pytest.raises(ValueError):
            check_output(Config(), path)
    check_output(Config(), tmp_path / "safe")
