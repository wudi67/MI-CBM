"""Reuse Fusion/TorchQuantum circuits, with fixed uniform/Gaussian initialization."""

from copy import deepcopy

import torch

from experiments.grouped_dynamic_vqc.runtime import cuda_runtime, rng_state
from experiments.grouped_robot_pilot.model import RobotVQC

from .protocol import SIGMA


def make_model(weights: dict) -> RobotVQC:
    model = RobotVQC(front_layers=4, label_layers=5).cuda()
    model.load_state_dict(weights)
    return model


def module_state(weights: dict, prefix: str) -> dict:
    return {
        k.removeprefix(prefix + "."): v
        for k, v in weights.items()
        if k.startswith(prefix + ".")
    }


def initializations(seeds: tuple[int, ...], original: dict) -> dict:
    initial = {}
    for seed in seeds:
        cuda_runtime(seed)
        model = RobotVQC(front_layers=4, label_layers=5)
        weights = deepcopy(model.state_dict())
        # Match the original Fusion initialization exactly; independent CPU streams
        # make the parameter draws independent of backend construction order.
        for prefix in ("frontend", "label_head"):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            for key, value in weights.items():
                if not key.startswith(prefix + "."):
                    continue
                if prefix == "frontend":
                    weights[key] = torch.pi * torch.rand(
                        value.shape, dtype=value.dtype, generator=generator
                    )
                else:
                    weights[key] = SIGMA * torch.randn(
                        value.shape, dtype=value.dtype, generator=generator
                    )
        weights["label_head.readout"][1].add_(torch.pi / 2)
        initial[str(seed)] = {
            "model": weights,
            # Preserve the historical seed-zero stream for exact provenance reuse.
            "rng": deepcopy(original["rng"]) if seed == 0 else rng_state(),
        }
    return initial


def label_initial(initial: dict, trained_frontend: dict) -> dict:
    value = deepcopy(initial)
    for key, tensor in trained_frontend.items():
        value["model"]["frontend." + key] = tensor.clone()
    return value
