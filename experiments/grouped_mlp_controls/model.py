"""251-parameter concept MLP and 253-parameter label MLP."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from experiments.grouped_dynamic_vqc.evaluation import concept_metrics, label_metrics

TASKS = ("concept", "label")
PARAMETERS = {"concept": 251, "label": 253}


class TinyMLP(nn.Module):
    """Flatten the unchanged 10x4 angle tensor; use one tanh hidden layer."""

    def __init__(self, task: str) -> None:
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"Unknown MLP task: {task}")
        self.task = task
        hidden, outputs = (3, 32) if task == "concept" else (6, 1)
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(40, hidden),
            nn.Tanh(),
            nn.Linear(hidden, outputs),
        )
        if sum(p.numel() for p in self.parameters()) != PARAMETERS[task]:
            raise RuntimeError("Unexpected MLP parameter count")

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.ndim != 3 or angles.shape[1:] != (10, 4):
            raise ValueError("Expected the unchanged [batch,10,4] VQC angle tensor")
        output = self.network(angles)
        return output if self.task == "concept" else output.squeeze(-1)


def objective(
    task: str, logits: torch.Tensor, data: dict, indices: torch.Tensor
) -> torch.Tensor:
    if task == "concept":
        concepts = data["concepts"][indices]
        return F.cross_entropy(logits, 8 * concepts[:, 0] + concepts[:, 1])
    if task == "label":
        return F.binary_cross_entropy_with_logits(logits, data["labels"][indices])
    raise ValueError(f"Unknown MLP task: {task}")


@torch.no_grad()
def evaluate(model: TinyMLP, data: dict, batch_size: int) -> dict:
    model.eval()
    outputs = [
        model(data["angles"][start : start + batch_size]).cpu()
        for start in range(0, len(data["angles"]), batch_size)
    ]
    logits = torch.cat(outputs)
    if model.task == "concept":
        concepts = data["concepts"].cpu()
        codes = 8 * concepts[:, 0] + concepts[:, 1]
        metrics = concept_metrics(logits.softmax(-1), concepts)
        # This is a classical predictive probability, not a hardware measurement.
        metrics["true_concept_probability"] = metrics.pop(
            "joint_single_shot_probability"
        )
        nll = float(F.cross_entropy(logits, codes))
        metrics["joint_nll"] = nll
        return {"n_samples": len(logits), "loss": nll, "concept": metrics}
    labels = data["labels"].cpu()
    bce = float(F.binary_cross_entropy_with_logits(logits, labels))
    metrics = {**label_metrics(logits.sigmoid(), labels), "bce": bce}
    return {"n_samples": len(logits), "loss": bce, "label": metrics}
