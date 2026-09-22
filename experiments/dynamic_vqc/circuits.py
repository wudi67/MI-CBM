"""FusionModel circuits with explicit upload modes and editable terminal Ry gates."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

import torchquantum as tq
import torchquantum.functional as tqf
from FusionModel import QNetArgs, TQLayer, single_enta_to_design

Direction = Literal["fixed", "alternating"]
UploadMode = Literal["once", "per_layer"]


def build_design(
    n_concepts: int,
    n_layers: int,
    direction: Direction,
    upload_mode: UploadMode = "once",
) -> list:
    """Keep edge order and parameter slots fixed when reversing even layers."""
    if n_concepts < 2 or n_layers < 1:
        raise ValueError("Need at least two concept wires and one layer")
    if direction not in {"fixed", "alternating"}:
        raise ValueError("direction must be fixed or alternating")
    if upload_mode not in {"once", "per_layer"}:
        raise ValueError("upload_mode must be once or per_layer")
    repeat = [1, 1] if upload_mode == "per_layer" else [0, 1]
    single = [[j + 1, 1, 1] + repeat * (n_layers - 1) for j in range(n_concepts)]
    enta = [[j + 1] + [(j + 1) % n_concepts + 1] * n_layers for j in range(n_concepts)]
    design = single_enta_to_design(single, enta, [n_concepts, n_layers])
    ordered = []
    # Only reorder commuting data/U3 gates WITHIN a layer. Moving later data
    # gates to the start would destroy the intended data re-uploading circuit.
    for layer_index in range(n_layers):
        layer_ops = [op for op in design if op[2] == layer_index]
        ordered.extend(op for op in layer_ops if op[0] == "data")
        ordered.extend(op for op in layer_ops if op[0] != "data")
    body = []
    for gate, wires, layer in ordered:
        if gate == "C(U3)" and direction == "alternating" and layer % 2:
            wires = list(reversed(wires))
        body.append((gate, wires, layer))
    return body


def bit_table(n_bits: int, device: torch.device | str) -> torch.Tensor:
    """TorchQuantum logical order: q0 is the most significant bit."""
    values = torch.arange(2**n_bits, device=device)
    shifts = torch.arange(n_bits - 1, -1, -1, device=device)
    return (values[:, None] >> shifts) & 1


def bit_indices(bits: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(bits.shape[-1] - 1, -1, -1, device=bits.device)
    return (bits.long() << shifts).sum(dim=-1)


def basis_device(bits: torch.Tensor) -> tq.QuantumDevice:
    """Prepare computational basis inputs; no amplitude or marginal encoding."""
    n_bits = bits.shape[1]
    qdev = tq.QuantumDevice(n_wires=n_bits, bsz=len(bits), device=bits.device)
    states = F.one_hot(bit_indices(bits), num_classes=2**n_bits).to(torch.complex64)
    qdev.set_states(states)
    return qdev


def joint_probabilities(qdev: tq.QuantumDevice) -> torch.Tensor:
    return qdev.get_states_1d().abs().square()


class ConceptCircuit(TQLayer):
    """Reuse FusionModel parameters, design builder and four-value encoders.

    FusionModel's forward returns Z expectations immediately. This isolated
    subclass exposes the state before measurement and adds the final beta.
    CU3 parameters are indexed by the ORIGINAL edge slot, so paired direction
    experiments start with identical per-edge angles, including the closing edge.
    """

    n_wires: int
    basis_bits: torch.Tensor
    uses_uploaded_cache = True

    def __init__(
        self,
        n_concepts: int = 9,
        n_layers: int = 2,
        direction: Direction = "alternating",
        upload_mode: UploadMode = "once",
    ) -> None:
        design = build_design(n_concepts, n_layers, direction, upload_mode)
        args = QNetArgs(
            task="QML_DynamicCBM_4Upload",
            n_qubits=n_concepts,
            n_layers=n_layers,
            backend="tq",
            device="cpu",
            quiet=True,
            disable_progress=True,
        )
        super().__init__(args, design)
        self.n_wires = n_concepts
        self.direction = direction
        self.upload_mode = upload_mode
        self.beta = nn.Parameter(torch.zeros(n_concepts))
        self.register_buffer("basis_bits", bit_table(n_concepts, "cpu").float())

    def state(self, angles: torch.Tensor) -> tq.QuantumDevice:
        return self.from_uploaded_state(
            self.upload_state(angles).get_states_1d(), angles=angles
        )

    def upload_state(self, angles: torch.Tensor) -> tq.QuantumDevice:
        """angles is [batch, concept wires, 4], already scaled in radians."""
        if angles.ndim != 3 or angles.shape[1:] != (self.n_wires, 4):
            raise ValueError(f"Expected [batch, {self.n_wires}, 4] upload angles")
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=len(angles), device=angles.device
        )
        for wire in range(self.n_wires):
            self.uploading[wire](qdev, angles[:, wire])
        return qdev

    def from_uploaded_state(
        self, states: torch.Tensor, *, angles: torch.Tensor | None = None
    ) -> tq.QuantumDevice:
        """Cache only E(x)|0>; later uploads act on the evolving trainable state."""
        if states.ndim != 2 or states.shape[1] != 2**self.n_wires:
            raise ValueError("Invalid uploaded state shape")
        valid_shapes = (
            (len(states), self.n_wires, 4),
            (self.args.n_layers, len(states), self.n_wires, 4),
        )
        if self.upload_mode == "per_layer" and (
            angles is None or tuple(angles.shape) not in valid_shapes
        ):
            raise ValueError("Per-layer uploading requires the original scaled angles")
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=len(states), device=states.device
        )
        qdev.set_states(states.clone())
        edge_slots = [0] * self.args.n_layers
        for gate, wires, layer in self.design:
            if gate == "data":
                if layer > 0:
                    assert angles is not None
                    layer_angles = angles[layer] if angles.ndim == 4 else angles
                    self.uploading[wires[0]](qdev, layer_angles[:, wires[0]])
                continue
            elif gate == "U3":
                tqf.u3(
                    qdev,
                    wires=wires,
                    params=self.q_params_rot[layer, wires[0]].unsqueeze(0),
                )
            elif gate == "C(U3)":
                slot = edge_slots[layer]
                tqf.cu3(
                    qdev,
                    wires=wires,
                    params=self.q_params_enta[layer, slot].unsqueeze(0),
                )
                edge_slots[layer] += 1
            else:
                raise ValueError(f"Unexpected FusionModel operation: {gate}")
        for wire in range(self.n_wires):
            tqf.ry(qdev, wires=wire, params=self.beta[wire])
        return qdev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Born probabilities for direct bitwise BCE, with no concept head."""
        return joint_probabilities(self.state(x)) @ self.basis_bits

    def xz_statistics(self, angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qdev = self.state(angles)
        z = 1 - 2 * (joint_probabilities(qdev) @ self.basis_bits)
        for wire in range(self.n_wires):
            tqf.hadamard(qdev, wires=wire)
        x = 1 - 2 * (joint_probabilities(qdev) @ self.basis_bits)
        return x, z


class QuantumLabelHead(nn.Module):
    """Ry(all), RZZ(star), Ry(all), RXX(star), Rz(r), Ry(r)."""

    def __init__(self, n_concepts: int = 9, n_layers: int = 1) -> None:
        super().__init__()
        if n_concepts < 1 or n_layers < 1:
            raise ValueError("Positive concept count and label depth required")
        self.n_concepts = n_concepts
        self.n_layers = n_layers
        self.a = nn.Parameter(torch.empty(n_layers, n_concepts + 1))
        self.b = nn.Parameter(torch.empty(n_layers, n_concepts + 1))
        self.gamma = nn.Parameter(torch.empty(n_layers, n_concepts))
        self.kappa = nn.Parameter(torch.empty(n_layers, n_concepts))
        self.readout = nn.Parameter(torch.empty(2))
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -torch.pi, torch.pi)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        if concepts.ndim != 2 or concepts.shape[1] != self.n_concepts:
            raise ValueError("Label input must be [batch, n_concepts] basis bits")
        # Callers validate binary values at the interface boundary; avoiding a
        # host synchronization here keeps every training step on CUDA.
        bits = torch.cat((concepts, concepts.new_zeros(len(concepts), 1)), dim=1)
        qdev = basis_device(bits)
        return self.from_device(qdev)

    def from_device(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        """Continue an existing concept-plus-label register without resetting it."""
        if qdev.n_wires != self.n_concepts + 1:
            raise ValueError("Label circuit requires concept wires plus one label wire")
        readout_wire = self.n_concepts
        for layer in range(self.n_layers):
            for wire in range(self.n_concepts + 1):
                tqf.ry(qdev, wires=wire, params=self.a[layer, wire])
            for wire in range(self.n_concepts):
                tqf.rzz(
                    qdev, wires=[wire, readout_wire], params=self.gamma[layer, wire]
                )
            for wire in range(self.n_concepts + 1):
                tqf.ry(qdev, wires=wire, params=self.b[layer, wire])
            for wire in range(self.n_concepts):
                tqf.rxx(
                    qdev, wires=[wire, readout_wire], params=self.kappa[layer, wire]
                )
        tqf.rz(qdev, wires=readout_wire, params=self.readout[0])
        tqf.ry(qdev, wires=readout_wire, params=self.readout[1])
        # r is the final logical wire (least significant bit).
        return (
            joint_probabilities(qdev).reshape(-1, 2**self.n_concepts, 2)[:, :, 1].sum(1)
        )


class ClassicalLabelHead(nn.Module):
    """9 -> 4 -> 1 ReLU reference, 45 parameters for dSprites."""

    def __init__(self, n_concepts: int = 9, hidden: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_concepts, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        return self.network(concepts.float()).squeeze(-1).sigmoid()


def corrected_bits(
    measured: torch.Tensor,
    truth: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Classical feed-forward equivalent of X**(m XOR truth), per shot."""
    if truth is None:
        if mask is not None:
            raise ValueError("A correction mask requires confirmed concept targets")
        return measured.clone()
    selected = torch.ones_like(truth, dtype=torch.bool) if mask is None else mask.bool()
    flips = torch.bitwise_xor(measured.long(), truth.long()) * selected
    return torch.bitwise_xor(measured.long(), flips)


class DynamicConceptModel(nn.Module):
    """Mid-circuit Z measurement / basis continuation in an ideal simulator."""

    beta_version: torch.Tensor

    def __init__(
        self,
        n_concepts: int = 9,
        n_layers: int = 2,
        direction: Direction = "alternating",
        persistent_feedback: bool = False,
        upload_mode: UploadMode = "once",
        encoding_mode: str = "fixed",
    ) -> None:
        super().__init__()
        if encoding_mode == "fixed":
            self.frontend = ConceptCircuit(n_concepts, n_layers, direction, upload_mode)
        else:
            from .upload_circuit import TrainableUploadCircuit

            self.frontend = TrainableUploadCircuit(
                n_concepts, n_layers, direction, upload_mode, encoding_mode
            )
        self.label_head = QuantumLabelHead(n_concepts)
        self.persistent_feedback = persistent_feedback
        self.register_buffer("beta_version", torch.tensor(0, dtype=torch.long))

    def get_extra_state(self) -> dict:
        return {"persistent_feedback": self.persistent_feedback}

    def set_extra_state(self, state: dict) -> None:
        self.persistent_feedback = bool(state["persistent_feedback"])

    def set_persistent_feedback(self, enabled: bool) -> None:
        """Disable future edits without removing beta, inference or immediate X."""
        self.persistent_feedback = bool(enabled)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        distribution = joint_probabilities(self.frontend.state(angles))
        return distribution @ self.label_head(self.frontend.basis_bits)

    @torch.no_grad()
    def sample(
        self,
        angles: torch.Tensor,
        shots: int,
        truth: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if shots < 1:
            raise ValueError("shots must be positive")
        distribution = joint_probabilities(self.frontend.state(angles))
        indices = torch.multinomial(distribution, shots, True, generator=generator)
        measured = self.frontend.basis_bits[indices].long()
        targets = None if truth is None else truth[:, None, :]
        selected = None if mask is None else mask[:, None, :]
        corrected = corrected_bits(measured, targets, selected)
        # Cache every basis response for this call, then gather all sampled
        # bitstrings. This is exact for the frozen deterministic label circuit.
        table = self.label_head(self.frontend.basis_bits)
        probabilities = table[bit_indices(corrected)]
        labels = (
            torch.rand(probabilities.shape, device=angles.device, generator=generator)
            < probabilities
        )
        return {"measured": measured, "corrected": corrected, "labels": labels}
