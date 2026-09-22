"""The same four interventions and joint sampling for both training modes."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.evaluation import concept_metrics, label_metrics
from experiments.grouped_dynamic_vqc.runtime import (
    atomic_checkpoint,
    atomic_json,
    report,
    sha256,
)
from experiments.grouped_feedback_ablation.protocol import load_checkpoint, read_json
from experiments.grouped_sequential_intervention.evaluation import (
    evaluate,
    forward_control,
)
from experiments.grouped_sequential_intervention.protocol import MODES, verify_artifacts
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .protocol import Experiment


@torch.no_grad()
def evaluate_exact(
    model, data: dict, batch_size: int, mode: str, *, cached_states: torch.Tensor
) -> tuple[dict, dict]:
    model.eval()
    ps, ys = [], []
    for start in range(0, len(cached_states), batch_size):
        states = cached_states[start : start + batch_size]
        out = forward_control(
            model, states, mode, data["concepts"][start : start + batch_size]
        )
        ps.append(states.reshape(-1, 32, 32).abs().square().sum(-1).cpu())
        ys.append(out["label_prob"].cpu())
    p, y = torch.cat(ps), torch.cat(ys)
    result = {
        "control_mode": mode,
        "test_evaluated": False,
        "concept": concept_metrics(p, data["concepts"].cpu()),
        "label": label_metrics(y, data["labels"].cpu()),
    }
    return result, {"concept_probabilities": p, "label_probabilities": y}


def directory(shared: Experiment, training: str, seed: int, mode: str) -> Path:
    return shared.output / training / f"seed{seed}" / "evaluation" / mode


def expected(shared: Experiment, training: str, seed: int, mode: str) -> dict:
    return {
        "manifest_sha256": shared.manifest_hash,
        "checkpoint_sha256": sha256(shared.model_path(training, seed)),
        "training_mode": training,
        "seed": seed,
        "control_mode": mode,
    }


def verify_condition(
    shared: Experiment, training: str, seed: int, mode: str
) -> dict | None:
    path = directory(shared, training, seed, mode)
    if not (path / "evaluation_lock.json").exists():
        return None
    lock = read_json(path / "evaluation_lock.json")
    if any(lock[k] != v for k, v in expected(shared, training, seed, mode).items()):
        raise ValueError("Evaluation provenance mismatch")
    verify_artifacts(path, lock["artifacts"])
    return read_json(path / "evaluation.json")


def save_condition(
    shared: Experiment,
    training: str,
    seed: int,
    mode: str,
    metrics: dict,
    raw: dict | None,
) -> None:
    path = directory(shared, training, seed, mode)
    if raw is not None:
        atomic_checkpoint(path / "predictions.pt", raw)
    metrics.update(expected(shared, training, seed, mode))
    atomic_json(path / "evaluation.json", metrics)
    atomic_json(
        path / "evaluation_lock.json",
        {
            **expected(shared, training, seed, mode),
            "artifacts": {
                name: sha256(path / name)
                for name in ("evaluation.json", "predictions.pt")
            },
        },
    )


@torch.no_grad()
def evaluate_model(shared: Experiment, training: str, seed: int) -> None:
    missing = [
        mode for mode in MODES if verify_condition(shared, training, seed, mode) is None
    ]
    if not missing:
        return
    if training == "sequential" and shared.sequential_reference["reuse"]:
        for mode in missing:
            src = Path(shared.sequential_reference["root"]) / f"seed{seed}" / mode
            verify_artifacts(src, read_json(src / "evaluation_lock.json")["artifacts"])
            path = directory(shared, training, seed, mode)
            path.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / "predictions.pt", path / "predictions.pt")
            metrics = read_json(src / "evaluation.json")
            metrics["origin"] = "verified Sequential evaluation reuse"
            metrics["source_evaluation_sha256"] = sha256(src / "evaluation.json")
            save_condition(shared, training, seed, mode, metrics, None)
        report(f"[green]Sequential seed {seed}: 已复用四种纠正评价[/green]")
        return
    weights = load_checkpoint(shared.model_path(training, seed))["model"]
    model = shared.make_model(weights).eval().requires_grad_(False)
    before = state_hash(model.state_dict())
    shared.prepare_states(model)
    for mode in missing:

        def progress(rows: int, active_mode: str = mode) -> None:
            shared.heartbeat(
                "evaluating", cell=f"{training}/seed{seed}/{active_mode}", offset=rows
            )
            if shared.stop_requested:
                raise InterruptedError(
                    "Resume recomputes the current unsaved condition"
                )

        metrics, raw = evaluate(
            model,
            shared.data["val"],
            shared.state_cache["val"],
            mode,
            batch_size=shared.config.eval_batch_size,
            shots=shared.config.shots,
            seed=seed,
            progress=progress,
        )
        if state_hash(model.state_dict()) != before:
            raise ValueError("Evaluation modified frozen model weights")
        metrics.update(origin="evaluated", model_sha256=before)
        save_condition(shared, training, seed, mode, metrics, raw)
        report(
            f"{training} seed {seed}, {mode}: "
            f"Label {metrics['label']['accuracy']:.2%}; "
            f"{shared.config.shots} shots "
            f"{metrics['finite_shots']['label']['accuracy']:.2%}"
        )
