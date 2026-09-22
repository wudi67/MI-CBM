"""Layer-specific affine angles using the original four-value FusionModel encoder."""

from __future__ import annotations

import torch
from torch import nn

import torchquantum as tq

from .circuits import ConceptCircuit, Direction, UploadMode

ENCODING_MODES = ("fixed", "trainable_scale", "trainable_affine")
PARAMETERS = dict(zip(ENCODING_MODES, (225, 369, 513), strict=True))


class TrainableUploadCircuit(ConceptCircuit):
    """Learn w*x+b per layer using differentiable FusionModel uploading."""

    uses_uploaded_cache = False

    def __init__(
        self,
        n_concepts: int = 9,
        n_layers: int = 4,
        direction: Direction = "fixed",
        upload_mode: UploadMode = "per_layer",
        encoding_mode: str = "trainable_affine",
    ) -> None:
        if encoding_mode not in ENCODING_MODES[1:] or upload_mode != "per_layer":
            raise ValueError("Trainable encoding requires scale/affine and per_layer")
        super().__init__(n_concepts, n_layers, direction, upload_mode)
        self.encoding_mode = encoding_mode
        shape = (n_layers, n_concepts, 4)
        # Constant initialization consumes no RNG: theta/beta stay seed-paired.
        self.upload_weight = nn.Parameter(torch.ones(shape))
        if encoding_mode == "trainable_affine":
            self.upload_bias = nn.Parameter(torch.zeros(shape))
        else:
            self.register_buffer("upload_bias", torch.zeros(shape))

    def state(self, angles: torch.Tensor) -> tq.QuantumDevice:
        if angles.ndim != 3 or angles.shape[1:] != (self.n_wires, 4):
            raise ValueError(f"Expected [batch, {self.n_wires}, 4] upload angles")
        # The fixed input is already in radians. Do not multiply by pi again.
        layers = self.upload_weight[:, None] * angles[None] + self.upload_bias[:, None]
        first = super().upload_state(layers[0]).get_states_1d()
        # All later uploads act on the evolving state, with autograd intact.
        return super().from_uploaded_state(first, angles=layers)

    def upload_state(self, angles: torch.Tensor) -> tq.QuantumDevice:
        return super().upload_state(
            self.upload_weight[0] * angles + self.upload_bias[0]
        )

    def from_uploaded_state(
        self, states: torch.Tensor, *, angles: torch.Tensor | None = None
    ) -> tq.QuantumDevice:
        raise ValueError(
            "Trainable uploading cannot use cached states; call state(angles)"
        )
