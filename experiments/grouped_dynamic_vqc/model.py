"""FusionModel frontend and exact measurement-conditioned quantum label head."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

import torchquantum as tq
import torchquantum.functional as tqf
from FusionModel import QNetArgs, TQLayer, single_enta_to_design

ControlMode = Literal["measured", "shape", "scale", "both", "zero"]


class StateReadout(nn.Module):
    """Expose the state without changing FusionModel's gate execution."""

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        return qdev.get_states_1d()


class FusionFrontend(TQLayer):
    """Call inherited forward/encoders/U3/CU3; replace only terminal readout."""

    def __init__(self, n_layers: int = 4) -> None:
        if n_layers < 1:
            raise ValueError("Frontend depth must be positive")
        single = [[wire + 1] + [1, 1] * n_layers for wire in range(10)]
        enta = [[wire + 1] + [(wire + 1) % 10 + 1] * n_layers for wire in range(10)]
        design = single_enta_to_design(single, enta, [10, n_layers])
        args = QNetArgs(
            task="QML_GroupedDynamicVQC",
            n_qubits=10,
            n_layers=n_layers,
            backend="tq",
            device="cpu",
            quiet=True,
            disable_progress=True,
        )
        super().__init__(args, design)
        self.measure = StateReadout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (10, 4):
            raise ValueError("Expected [batch, 10, 4] angles, already multiplied by pi")
        return super().forward(x.contiguous())


class QuantumLabelHead(nn.Module):
    """Five retained quantum wires plus a fresh readout wire in |0>."""

    def __init__(self, n_layers: int = 1) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("Label depth must be positive")
        self.n_layers = n_layers
        self.a = nn.Parameter(torch.empty(n_layers, 6))
        self.b = nn.Parameter(torch.empty(n_layers, 6))
        self.gamma = nn.Parameter(torch.empty(n_layers, 5))
        self.kappa = nn.Parameter(torch.empty(n_layers, 5))
        self.readout = nn.Parameter(torch.empty(2))
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -torch.pi, torch.pi)

    def evolve(self, retained: torch.Tensor) -> torch.Tensor:
        """Evolve arbitrary, possibly unnormalized, retained states [N,32]."""
        if retained.ndim != 2 or retained.shape[1] != 32:
            raise ValueError("Expected retained amplitudes [batch, 32]")
        with_readout = torch.stack((retained, torch.zeros_like(retained)), dim=-1)
        qdev = tq.QuantumDevice(n_wires=6, bsz=len(retained), device=retained.device)
        qdev.set_states(with_readout.reshape(-1, 64))
        for layer in range(self.n_layers):
            for wire in range(6):
                tqf.ry(qdev, wires=wire, params=self.a[layer, wire])
            for wire in range(5):
                tqf.rzz(qdev, wires=[wire, 5], params=self.gamma[layer, wire])
            for wire in range(6):
                tqf.ry(qdev, wires=wire, params=self.b[layer, wire])
            for wire in range(5):
                tqf.rxx(qdev, wires=[wire, 5], params=self.kappa[layer, wire])
        tqf.rz(qdev, wires=5, params=self.readout[0])
        tqf.ry(qdev, wires=5, params=self.readout[1])
        return qdev.get_states_1d().reshape(-1, 32, 2)

    def forward(self, retained: torch.Tensor) -> torch.Tensor:
        """Direct branch evolution, also used as an independent speedup check."""
        return self.evolve(retained)[:, :, 1].abs().square().sum(-1)

    def one_amplitudes(self) -> torch.Tensor:
        """Exact linear map K[b,j]=<j,1|U|b,0>, recomputed with gradients.

        Evolving 32 basis inputs once is sufficient for every branch in a batch.
        A complex state v maps to v @ K, preserving interference within B.
        This is a simulator optimization of the SAME circuit, not a lookup of
        classical concept probabilities or a trainable classical head.
        """
        basis = torch.eye(32, dtype=torch.complex64, device=self.a.device)
        return self.evolve(basis)[:, :, 1]


def controls_for(
    batch_size: int,
    device: torch.device,
    mode: ControlMode = "measured",
    concepts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace only the classical control record; keep the physical branch m."""
    measured = torch.arange(32, device=device).expand(batch_size, -1)
    if mode == "measured":
        return measured
    if mode == "zero":
        return torch.zeros_like(measured)
    if concepts is None or concepts.shape != (batch_size, 2):
        raise ValueError("Concept correction requires [batch, 2] integer targets")
    shape, scale = concepts[:, :1], concepts[:, 1:2]
    if mode == "shape":
        return shape * 8 + (measured & 7)
    if mode == "scale":
        return (measured & 24) + scale
    if mode == "both":
        return (shape * 8 + scale).expand(-1, 32)
    raise ValueError(f"Unknown control mode: {mode}")


def controlled_branches(states: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    """Unnormalized |psi_m> with X^(control_m) applied to B, q0 most significant.

    Splitting the amplitudes into 32 measured branches and summing their output
    probabilities implements projective measurement, including its backaction.
    Never sum amplitudes across different measured outcomes.
    """
    if states.ndim != 2 or states.shape[1] != 1024:
        raise ValueError("Expected ten-qubit amplitudes [batch, 1024]")
    if controls.shape != (len(states), 32):
        raise ValueError("Expected one five-bit control code per measured branch")
    indices = torch.arange(32, device=states.device) ^ controls.unsqueeze(-1)
    return states.reshape(-1, 32, 32).gather(2, indices)


class GroupedDynamicVQC(nn.Module):
    """10 uploaded wires, 5 measured concepts, 5 retained wires, 1 readout."""

    def __init__(self, front_layers: int = 4, label_layers: int = 1) -> None:
        super().__init__()
        self.frontend = FusionFrontend(front_layers)
        self.label_head = QuantumLabelHead(label_layers)

    def from_state(
        self,
        states: torch.Tensor,
        controls: torch.Tensor | None = None,
        *,
        direct: bool = False,
    ) -> dict[str, torch.Tensor]:
        if controls is None:
            controls = controls_for(len(states), states.device)
        branches = controlled_branches(states, controls)
        probabilities = branches.abs().square().sum(-1)
        if direct:
            label_mass = self.label_head(branches.reshape(-1, 32)).reshape(-1, 32)
        else:
            amplitudes = branches @ self.label_head.one_amplitudes()
            label_mass = amplitudes.abs().square().sum(-1)
        return {
            "concept_probs": probabilities,
            "label_prob": label_mass.sum(-1),
            "branch_label_mass": label_mass,
        }

    def forward(self, angles: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.from_state(self.frontend(angles))


def sample_shots(
    output: dict[str, torch.Tensor], shots: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the joint physical distribution of (measured code m, readout y)."""
    if shots < 1:
        raise ValueError("shots must be positive")
    p_one = output["branch_label_mass"]
    p_zero = output["concept_probs"] - p_one
    joint = torch.stack((p_zero, p_one), dim=-1).flatten(1).clamp_min(0)
    outcomes = torch.multinomial(joint, shots, replacement=True, generator=generator)
    return outcomes // 2, outcomes % 2
