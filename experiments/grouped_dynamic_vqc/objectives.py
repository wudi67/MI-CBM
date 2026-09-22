"""Joint grouped-concept NLL and binary label loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def loss_function(
    output: dict[str, torch.Tensor],
    concepts: torch.Tensor,
    labels: torch.Tensor,
    concept_weight: float = 1.0,
    label_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codes = concepts[:, 0] * 8 + concepts[:, 1]
    correct = output["concept_probs"].gather(1, codes[:, None]).squeeze(1)
    concept_loss = -correct.clamp_min(1e-7).log().mean()
    label_loss = F.binary_cross_entropy(
        output["label_prob"].clamp(1e-7, 1 - 1e-7), labels.float()
    )
    return (
        concept_weight * concept_loss + label_weight * label_loss,
        concept_loss,
        label_loss,
    )
