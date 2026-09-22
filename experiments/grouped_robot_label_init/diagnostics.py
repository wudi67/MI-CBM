"""Initial probability bias and gradients on one fixed TRAIN batch, without updates."""

import math

import torch
import torch.nn.functional as F

from experiments.grouped_dynamic_vqc.runtime import (
    array_hash,
    atomic_json,
    restore_rng,
    rng_state,
    sha256,
)
from experiments.grouped_robot_pilot.model import forward_control
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .model import make_model
from .protocol import CELLS, cell_name


def ensure_diagnostics(shared) -> None:
    lock_path = shared.output / "initial_diagnostics_lock.json"
    identity = {
        "manifest_sha256": shared.manifest_hash,
        "initialization_lock_sha256": shared.initial_hash,
    }
    if lock_path.exists():
        lock = read_json(lock_path)
        if any(lock.get(k) != v for k, v in identity.items()):
            raise ValueError("Initial diagnostic identity changed")
        verify_files(shared.output, lock["artifacts"])
        return
    cfg = shared.reference.pilot
    data = shared.data["train"]
    indices = epoch_order(len(data["labels"]), cfg.seed, cfg.concept_epochs + 1)[
        : cfg.batch_size
    ]
    cuda_indices = indices.cuda()
    states = shared.state_cache["train"][cuda_indices]
    saved_rng, rows = rng_state(), []
    try:
        for method, index in CELLS:
            name = cell_name(method, index)
            shared.tick("initial_probability_and_gradient_check", source_cell=name)
            model = make_model(shared.initial[name]["model"])
            before = state_hash(model.state_dict())
            prediction = forward_control(
                model, states, data["concepts"][cuda_indices], mask=31
            )
            p = prediction["label_prob"]
            loss = F.binary_cross_entropy(
                p.clamp(1e-7, 1 - 1e-7), data["labels"][cuda_indices]
            )
            if not torch.isfinite(loss):
                raise ValueError("Initial diagnostic loss is nonfinite")
            loss.backward()
            norms = {}
            for key, parameter in model.label_head.named_parameters():
                grad = parameter.grad
                if grad is None or not grad.is_cuda or not torch.isfinite(grad).all():
                    raise ValueError("Initial gradients must be finite and on CUDA")
                norms[key] = float(grad.norm())
            if any(v.grad is not None for v in model.frontend.parameters()):
                raise ValueError(
                    "Initial diagnostic differentiated the frozen frontend"
                )
            if before != state_hash(model.state_dict()):
                raise ValueError("Initial diagnostic changed weights")
            gradient = math.sqrt(sum(v * v for v in norms.values()))
            rows.append(
                {
                    "cell_name": name,
                    "method": method,
                    "initialization_index": index,
                    "initial_model_sha256": before,
                    "initial_bce": float(loss.detach()),
                    "p_mean": float(p.detach().mean()),
                    "p_min": float(p.detach().min()),
                    "p_max": float(p.detach().max()),
                    "clamped_fraction": float(
                        ((p.detach() < 1e-7) | (p.detach() > 1 - 1e-7)).float().mean()
                    ),
                    "gradient_l2_before_clip": gradient,
                    "parameter_gradient_l2": norms,
                    "would_clip": gradient > cfg.grad_clip,
                    "optimizer_updates": 0,
                    "device": str(p.device),
                }
            )
            del model, prediction, p, loss
    finally:
        restore_rng(saved_rng)
    atomic_json(
        shared.output / "initial_diagnostics.json",
        {
            **identity,
            "role": "train",
            "n_samples": len(indices),
            "correction_mask": 31,
            "indices_sha256": array_hash(indices.numpy()),
            "rows": rows,
            "interpretation": (
                "Initial-output/gradient diagnostic, not trained performance "
                "or a barren-plateau test"
            ),
        },
    )
    atomic_json(
        lock_path,
        {
            **identity,
            "artifacts": {
                "initial_diagnostics.json": sha256(
                    shared.output / "initial_diagnostics.json"
                )
            },
        },
    )
