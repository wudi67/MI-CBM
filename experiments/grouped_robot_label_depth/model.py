"""Reuse the existing circuit verbatim; pair common layers at initialization."""

from copy import deepcopy

import torch

from experiments.grouped_dynamic_vqc.model import GroupedDynamicVQC, QuantumLabelHead
from experiments.grouped_dynamic_vqc.runtime import cuda_runtime, rng_state
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .protocol import INITIALIZATIONS, cell_name


def make_model(weights: dict) -> GroupedDynamicVQC:
    depth = weights["label_head.a"].shape[0]
    model = GroupedDynamicVQC(front_layers=4, label_layers=depth).cuda()
    model.load_state_dict(weights)
    model.frontend.requires_grad_(False)
    model.label_head.requires_grad_(True)
    if sum(p.numel() for p in model.label_head.parameters()) != 22 * depth + 2:
        raise ValueError("Unexpected label parameter count")
    return model


def initializations(reference) -> dict:
    historical = reference.reference.initial_for("independent")
    if (
        state_hash(historical["model"])
        != reference.training["long_label/independent"]["initial_model_sha256"]
    ):
        raise ValueError(
            "Historical A initialization does not match its training record"
        )
    front = {
        k: v.clone()
        for k, v in historical["model"].items()
        if k.startswith("frontend.")
    }
    result = {}
    for index in INITIALIZATIONS:
        if index == 0:
            small = {
                k.removeprefix("label_head."): v.clone()
                for k, v in historical["model"].items()
                if k.startswith("label_head.")
            }
            initial_rng = deepcopy(historical["rng"])
        else:
            cuda_runtime(index)
            head = QuantumLabelHead(1).cuda()
            small = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            initial_rng = rng_state()
        # Added layers are independently randomized, not initialized from trained A.
        cuda_runtime(100000 + index)
        large_head = QuantumLabelHead(5).cuda()
        large = {
            k: v.detach().cpu().clone() for k, v in large_head.state_dict().items()
        }
        for name in ("a", "b", "gamma", "kappa"):
            large[name][0].copy_(small[name][0])
        large["readout"].copy_(small["readout"])
        for depth, head_state in ((1, small), (5, large)):
            result[cell_name(depth, index)] = {
                "model": {
                    **front,
                    **{f"label_head.{k}": v for k, v in head_state.items()},
                },
                "rng": deepcopy(initial_rng),
                "initialization_index": index,
                "head_layers": depth,
                "extra_layers_seed": 100000 + index if depth == 5 else None,
            }
    return result


def verify_initializations(values: dict, frontend_hash: str) -> None:
    for index in INITIALIZATIONS:
        a, b = (values[cell_name(depth, index)]["model"] for depth in (1, 5))
        for name in ("a", "b", "gamma", "kappa"):
            if not torch.equal(a[f"label_head.{name}"][0], b[f"label_head.{name}"][0]):
                raise ValueError(
                    "The common first layer lost its paired initialization"
                )
        if not torch.equal(a["label_head.readout"], b["label_head.readout"]):
            raise ValueError("Readout initialization is not paired")
        for weights in (a, b):
            front = {
                k.removeprefix("frontend."): v
                for k, v in weights.items()
                if k.startswith("frontend.")
            }
            if state_hash(front) != frontend_hash:
                raise ValueError("Initialization changed the frozen frontend")
