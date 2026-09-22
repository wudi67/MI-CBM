"""Only initialize parameters; reuse the original TorchQuantum circuit verbatim."""

import math
from copy import deepcopy

import torch

from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_label_depth.model import make_model
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .protocol import CELLS, DEPTH, INDICES, Config, cell_name

__all__ = ["make_model", "initializations", "verify_initializations"]


def initializations(reference, config: Config) -> dict:
    values = {}
    for index in INDICES:
        uniform = deepcopy(reference.initial[f"L5/init_{index}"])
        gaussian = deepcopy(uniform)
        generator = torch.Generator(device="cpu").manual_seed(index)
        for name, value in gaussian["model"].items():
            if name.startswith("label_head."):
                gaussian["model"][name] = (
                    torch.randn(value.shape, generator=generator, dtype=value.dtype)
                    * config.sigma
                )
        shifted = deepcopy(gaussian)
        shifted["model"]["label_head.readout"][1].add_(math.pi / 2)
        for method, initial in (
            ("uniform", uniform),
            ("eft_gaussian", gaussian),
            ("eft_readout", shifted),
        ):
            values[cell_name(method, index)] = {
                "model": initial["model"],
                "rng": deepcopy(uniform["rng"]),
                "head_layers": DEPTH,
                "initialization_index": index,
                "initialization_method": method,
                "sigma": None if method == "uniform" else config.sigma,
                "readout_ry_center": math.pi / 2 if method == "eft_readout" else 0.0,
                "source_uniform_model_sha256": state_hash(uniform["model"]),
            }
    return values


def verify_initializations(values: dict, reference, config: Config) -> None:
    expected = initializations(reference, config)
    if set(values) != {cell_name(*cell) for cell in CELLS}:
        raise ValueError("Initialization cells changed")
    for name, original in expected.items():
        actual = values[name]
        if tree_hash(actual["rng"]) != tree_hash(original["rng"]):
            raise ValueError("Initialization RNG state changed")
        if state_hash(actual["model"]) != state_hash(original["model"]):
            raise ValueError("Initialization no longer follows the fixed method/seed")
        for key in (
            "head_layers",
            "initialization_index",
            "initialization_method",
            "sigma",
            "readout_ry_center",
            "source_uniform_model_sha256",
        ):
            if actual[key] != original[key]:
                raise ValueError("Initialization metadata changed")
