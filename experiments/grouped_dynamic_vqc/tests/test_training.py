"""Actual dSprites/CUDA recovery and provenance checks, including mid-epoch resume."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from ..runtime import sha256
from ..train import TrainConfig, TrainingRun


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_tree_equal(first, second)
    else:
        assert left == right


def test_cuda_dsprites_resume_matches_uninterrupted_adam_and_rng(tmp_path):
    config = TrainConfig(
        epochs=2,
        batch_size=18,
        eval_batch_size=18,
        train_limit=36,
        val_limit=18,
        checkpoint_steps=1,
        shots=8,
    )
    full_path = tmp_path / "full"
    resume_path = tmp_path / "resumed"
    full = TrainingRun(config, full_path)
    full.run()
    paused = TrainingRun(config, resume_path)
    assert paused.run(max_steps=1)["status"] == "paused"
    assert paused.progress["offset"] == 18
    continued = TrainingRun(config, resume_path, resume=True)
    assert continued.run(max_steps=1) == {"status": "paused", "global_step": 1}
    result = continued.run()
    assert result["status"] == "complete" and result["global_step"] == 4
    first = torch.load(full_path / "resume.pt", weights_only=False, map_location="cpu")
    second = torch.load(
        resume_path / "resume.pt", weights_only=False, map_location="cpu"
    )
    for key in ("model", "optimizer", "rng", "progress"):
        assert_tree_equal(first[key], second[key])
    assert result["test_evaluated"] is False
    assert len(result["validation"]["control_record_interventions"]) == 4
    assert result["checkpoint_sha256"] == sha256(resume_path / "resume.pt")
    audit = json.loads((resume_path / "preprocessing.json").read_text())
    assert audit["routing_fit_rows"] == 25593
    assert audit["roles"]["train"]["used_rows"] == 36
    assert audit["test_preprocessed"] is False
    assert all(tensor.is_cuda for tensor in continued.data["train"].values())
    with pytest.raises(ValueError, match="identical config"):
        TrainingRun(replace(config, learning_rate=0.001), resume_path, resume=True)
    with (resume_path / "data.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Cached inputs changed"):
        TrainingRun(config, resume_path, resume=True)
