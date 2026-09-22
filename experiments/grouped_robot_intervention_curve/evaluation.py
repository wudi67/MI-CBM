"""Replace selected classical controls; retain original Born branches and states."""

import torch

from experiments.grouped_robot_pilot.evaluation import evaluate
from experiments.grouped_robot_shots_final.evaluation import exact_metrics


@torch.no_grad()
def infer(model, data: dict, states: torch.Tensor, mask: int, batch: int, tick) -> dict:
    if not 0 <= mask < 32:
        raise ValueError("Expected a five-bit correction mask")
    _, raw = evaluate(
        model,
        {**data, "concepts": data["concepts"].cuda()},
        batch,
        states=states,
        mask=mask,
        tick=tick,
    )
    exact_metrics(raw)
    return raw


def verify_concepts(raw: dict, baseline: dict) -> None:
    for key in ("labels", "concepts", "source_index", "robot_ids"):
        if not torch.equal(raw[key], baseline[key]):
            raise ValueError("Intervention changed sample identities/targets")
    if not torch.allclose(
        raw["concept_probabilities"],
        baseline["concept_probabilities"],
        atol=2e-6,
        rtol=0,
    ):
        raise ValueError("Intervention changed original Born concept probabilities")
