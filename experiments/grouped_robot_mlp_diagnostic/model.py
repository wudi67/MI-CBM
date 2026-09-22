"""Explicit binary concept inputs and exact Born-weighted label probabilities."""

import torch
from torch import nn

from experiments.grouped_dynamic_vqc.evaluation import label_metrics
from experiments.grouped_robot_pilot.model import bits, codes, controls

CONDITIONS = (("measured", 0), ("correct_foot_shape", 1), ("correct_all_five", 31))


class ConceptMLP(nn.Module):
    """Five binary concepts -> 16 Tanh units -> one label logit (113 parameters)."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        if concepts.ndim != 2 or concepts.shape[1] != 5:
            raise ValueError(
                "MLP requires five explicit concepts, never image features"
            )
        return self.network(concepts.float()).squeeze(-1)


def marginalize(
    table: torch.Tensor, probabilities: torch.Tensor, concepts: torch.Tensor, mask: int
) -> torch.Tensor:
    """Keep physical branch weights; replace only the selected classical bits."""
    if table.shape != (32,) or probabilities.shape != (len(concepts), 32):
        raise ValueError("Expected a 32-code MLP table and [N,32] Born probabilities")
    records = controls(concepts, mask=mask)
    return probabilities * table[records]


@torch.no_grad()
def predict(model: ConceptMLP, data: dict) -> dict:
    model.eval()
    device = data["concepts"].device
    table = model(bits(device)).sigmoid()
    masses = {
        name: marginalize(table, data["concept_probabilities"], data["concepts"], mask)
        for name, mask in CONDITIONS
    }
    direct = table[codes(data["concepts"])]
    torch.testing.assert_close(
        masses["correct_all_five"].sum(1), direct, atol=2e-5, rtol=0
    )
    return {
        "code_label_probabilities": table.cpu(),
        "branch_label_mass": {k: v.cpu() for k, v in masses.items()},
        "direct_true_probability": direct.cpu(),
    }


def scores(raw: dict, labels: torch.Tensor) -> dict:
    return {
        **{
            name: label_metrics(mass.sum(1), labels)
            for name, mass in raw["branch_label_mass"].items()
        },
        "direct_true": label_metrics(raw["direct_true_probability"], labels),
    }


def transitions(
    before: torch.Tensor, after: torch.Tensor, labels: torch.Tensor
) -> dict:
    """Sample counts explain accuracy changes independently of confidence/BCE."""
    previous = (before >= 0.5) == labels.bool()
    current = (after >= 0.5) == labels.bool()
    gained, lost = int((~previous & current).sum()), int((previous & ~current).sum())
    return {
        "both_correct": int((previous & current).sum()),
        "correct_to_wrong": lost,
        "wrong_to_correct": gained,
        "both_wrong": int((~previous & ~current).sum()),
        "net_correct": gained - lost,
        "accuracy_change_pp": 100 * (gained - lost) / len(labels),
    }
