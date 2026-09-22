"""Change only the 240 existing U3/CU3 angles; keep data uploading unchanged."""

from copy import deepcopy

import torch

from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_pilot.model import concept_loss
from experiments.grouped_robot_pilot.runner import Experiment as PilotExperiment
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .protocol import CELLS, INDICES, Config, cell_name

FRONT_KEYS = ("frontend.q_params_rot", "frontend.q_params_enta")
make_model = PilotExperiment.make_model


def initializations(original: dict, config: Config) -> dict:
    values = {}
    if {k for k in original["model"] if k.startswith("frontend.")} != set(FRONT_KEYS):
        raise ValueError("Unexpected frontend parameters; audit initialization first")
    for index in INDICES:
        uniform, gaussian = deepcopy(original), deepcopy(original)
        for method, value in (("uniform", uniform), ("eft_gaussian", gaussian)):
            generator = torch.Generator(device="cpu").manual_seed(index)
            for name in FRONT_KEYS:
                tensor = original["model"][name]
                if tuple(tensor.shape) != (4, 10, 3):
                    raise ValueError("Expected four layers of ten three-angle gates")
                if method == "uniform":
                    draw = torch.rand(
                        tensor.shape, generator=generator, dtype=tensor.dtype
                    )
                    value["model"][name] = torch.pi * draw
                else:
                    draw = torch.randn(
                        tensor.shape, generator=generator, dtype=tensor.dtype
                    )
                    value["model"][name] = config.sigma * draw
            value.update(method=method, initialization_index=index)
            values[cell_name(method, index)] = value
    if state_hash(values["uniform/init_0"]["model"]) != state_hash(original["model"]):
        raise ValueError(
            "Uniform seed zero must reproduce the original Fusion initialization"
        )
    return values


def verify_initializations(values: dict, original: dict, config: Config) -> None:
    if tree_hash(values) != tree_hash(initializations(original, config)):
        raise ValueError("Initialization weights, RNG or method changed")


def initial_diagnostics(values: dict, data: dict, batch_size: int) -> dict:
    rows = []
    x, target = data["angles"][:batch_size], data["concepts"][:batch_size]
    for method, index in CELLS:
        model = make_model(values[cell_name(method, index)]["model"])
        model.label_head.requires_grad_(False)
        probability = model.frontend(x).reshape(-1, 32, 32).abs().square().sum(-1)
        loss = concept_loss(probability, target)
        loss.backward()
        norms = {}
        for name, tensor in model.frontend.named_parameters():
            if (
                tensor.grad is None
                or not tensor.grad.is_cuda
                or not torch.isfinite(tensor.grad).all()
            ):
                raise RuntimeError("Concept diagnostics require finite CUDA gradients")
            norms[name] = [float(v.norm()) for v in tensor.grad]
        rows.append(
            {
                "method": method,
                "initialization_index": index,
                "n_samples": len(x),
                "joint_nll": float(loss.detach()),
                "gradient_l2_before_clip": sum(
                    v * v for group in norms.values() for v in group
                )
                ** 0.5,
                "layer_gradient_l2": norms,
                "device": str(x.device),
                "optimizer_updates": 0,
            }
        )
        del model
    return {"role": "first_train_batch", "test_read": False, "rows": rows}
