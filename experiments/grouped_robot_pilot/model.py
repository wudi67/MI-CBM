"""Reuse the dSprites circuit verbatim; adapt only five-binary-concept semantics."""

from __future__ import annotations

import torch

from experiments.grouped_dynamic_vqc.model import GroupedDynamicVQC as RobotVQC


def bits(device: torch.device) -> torch.Tensor:
    return (
        (
            torch.arange(32, device=device)[:, None]
            >> torch.arange(4, -1, -1, device=device)
        )
        & 1
    ).long()


def codes(concepts: torch.Tensor) -> torch.Tensor:
    if concepts.ndim != 2 or concepts.shape[1] != 5:
        raise ValueError("Expected five binary concepts in canonical Robot order")
    powers = torch.tensor([16, 8, 4, 2, 1], device=concepts.device)
    return (concepts.long() * powers).sum(1)


def controls(
    concepts: torch.Tensor, *, zero: bool = False, mask: int = 0
) -> torch.Tensor:
    """Replace selected CLASSICAL bits; preserve the physical measured branch m."""
    if not 0 <= mask <= 31 or (zero and mask):
        raise ValueError("Invalid correction mask / zero-feedback combination")
    measured = torch.arange(32, device=concepts.device).expand(len(concepts), -1)
    if zero:
        return torch.zeros_like(measured)
    if not mask:
        return measured
    return (measured & (31 ^ mask)) | (codes(concepts)[:, None] & mask)


def forward_control(
    model: RobotVQC,
    states: torch.Tensor,
    concepts: torch.Tensor,
    *,
    zero: bool = False,
    mask: int = 0,
) -> dict:
    return model.from_state(states, controls(concepts, zero=zero, mask=mask))


def concept_loss(probabilities: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
    # Exactly the previous joint-record NLL, now all 32 Robot records are legal.
    return (
        -probabilities.gather(1, codes(concepts)[:, None]).clamp_min(1e-7).log().mean()
    )
