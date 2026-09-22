"""Check quantum semantics independently of the optimized contraction."""

from __future__ import annotations

import pytest
import torch

import torchquantum as tq
import torchquantum.functional as tqf
from FusionModel import TQLayer

from ..model import (
    FusionFrontend,
    GroupedDynamicVQC,
    controlled_branches,
    controls_for,
    sample_shots,
)
from ..objectives import loss_function


@pytest.fixture(autouse=True)
def fixed_rng():
    torch.set_num_threads(1)
    torch.manual_seed(24)


def random_states(batch: int, device: str = "cpu") -> torch.Tensor:
    states = torch.randn(batch, 1024, dtype=torch.complex64, device=device)
    return states / states.norm(dim=1, keepdim=True)


def test_frontend_reuses_fusion_forward_and_four_complete_layers():
    frontend = FusionFrontend()
    assert sum(p.numel() for p in frontend.parameters()) == 240
    for layer in range(4):
        gates = [op for op in frontend.design if op[2] == layer]
        assert len(gates) == 30
        assert [w for gate, w, _ in gates if gate == "C(U3)"] == [
            [wire, (wire + 1) % 10] for wire in range(10)
        ]
    assert [entry["func"] for entry in frontend.data_uploading(3)] == [
        "ry",
        "rz",
        "rx",
        "ry",
    ]
    original = TQLayer(frontend.args, frontend.design)
    original.load_state_dict(frontend.state_dict())
    angles = torch.rand(2, 10, 4) * torch.pi
    states = frontend(angles)
    bits = ((torch.arange(1024)[:, None] >> torch.arange(9, -1, -1)) & 1).float()
    z_expectations = states.abs().square() @ (1 - 2 * bits)
    torch.testing.assert_close(z_expectations, original(angles), atol=2e-6, rtol=2e-6)
    assert sum(p.numel() for p in GroupedDynamicVQC().parameters()) == 264


@pytest.mark.parametrize("mode", ["measured", "shape", "scale", "both", "zero"])
def test_exact_mixture_matches_normalized_torchquantum_trajectories(mode):
    model = GroupedDynamicVQC()
    states = random_states(2)
    concepts = torch.tensor([[2, 5], [1, 2]])
    controls = controls_for(2, states.device, mode, concepts)
    output = model.from_state(states, controls)
    expected_masses = []
    for measured in range(32):
        retained = states.reshape(2, 32, 32)[:, measured, :]
        branch_probability = retained.abs().square().sum(1)
        normalized = retained / branch_probability.sqrt()[:, None]
        # Independent reference: apply actual X gates to each trajectory.
        reference = []
        for row in range(2):
            qdev = tq.QuantumDevice(n_wires=5)
            qdev.set_states(normalized[row : row + 1])
            for bit in range(5):
                if int(controls[row, measured]) & (1 << (4 - bit)):
                    tqf.paulix(qdev, wires=bit)
            reference.append(model.label_head(qdev.get_states_1d()))
        expected_masses.append(torch.cat(reference) * branch_probability)
    expected = torch.stack(expected_masses, dim=1)
    torch.testing.assert_close(
        output["branch_label_mass"], expected, atol=2e-7, rtol=3e-6
    )
    torch.testing.assert_close(output["concept_probs"].sum(1), torch.ones(2))


def test_measurement_removes_cross_branch_phase_but_preserves_within_branch_phase():
    model = GroupedDynamicVQC()
    states = random_states(2).reshape(2, 32, 32)
    reference = model.from_state(states.reshape(2, -1))["label_prob"]
    branch_phases = torch.exp(1j * torch.rand(2, 32, 1) * 6)
    changed = model.from_state((states * branch_phases).reshape(2, -1))["label_prob"]
    torch.testing.assert_close(reference, changed)
    internal_phases = torch.exp(1j * torch.rand(2, 32, 32) * 6)
    changed = model.from_state((states * internal_phases).reshape(2, -1))["label_prob"]
    assert float((reference - changed).abs().max()) > 1e-4


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fast_head_matches_direct_values_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = GroupedDynamicVQC().to(device)
    states = random_states(2, device).requires_grad_()
    parameters = [states, *model.label_head.parameters()]
    fast = model.from_state(states)["label_prob"]
    fast_grad = torch.autograd.grad(fast.sum(), parameters)
    direct = model.from_state(states, direct=True)["label_prob"]
    direct_grad = torch.autograd.grad(direct.sum(), parameters)
    torch.testing.assert_close(fast, direct, atol=2e-6, rtol=2e-6)
    for actual, expected in zip(fast_grad, direct_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)


def test_zero_probability_branches_are_finite_and_control_endianness_is_exact():
    states = torch.zeros(1, 1024, dtype=torch.complex64)
    states[0, 17 * 32 + 3] = 1
    branches = controlled_branches(states, controls_for(1, states.device))
    assert branches[0, 17, 3 ^ 17] == 1
    output = GroupedDynamicVQC().from_state(states)
    assert torch.isfinite(output["label_prob"]).all()
    assert output["concept_probs"][0, 17] == 1


def test_cuda_frontend_and_head_receive_real_gradients_and_match_cpu():
    if not torch.cuda.is_available():
        pytest.fail("This device must support CUDA for the requested experiment")
    cpu_model = GroupedDynamicVQC()
    cuda_model = GroupedDynamicVQC().cuda()
    cuda_model.load_state_dict(cpu_model.state_dict())
    angles = torch.rand(3, 10, 4) * torch.pi
    cpu_output = cpu_model(angles)
    cuda_output = cuda_model(angles.cuda())
    for key, value in cpu_output.items():
        torch.testing.assert_close(value, cuda_output[key].cpu(), atol=2e-6, rtol=3e-5)
    concepts = torch.tensor([[0, 0], [1, 3], [2, 5]], device="cuda")
    labels = torch.tensor([0, 1, 0], device="cuda").float()
    loss_function(cuda_output, concepts, labels)[0].backward()
    for module in (cuda_model.frontend, cuda_model.label_head):
        for parameter in module.parameters():
            assert parameter.is_cuda and parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().max() > 0


def test_joint_shot_sampling_respects_measured_label_correlation():
    probabilities = torch.zeros(2, 32)
    probabilities[:, 3] = 0.25
    probabilities[:, 17] = 0.75
    label_mass = torch.zeros_like(probabilities)
    label_mass[:, 17] = 0.75
    output = {"concept_probs": probabilities, "branch_label_mass": label_mass}
    measured, label = sample_shots(output, 10000, torch.Generator().manual_seed(0))
    assert torch.equal(label, (measured == 17).long())
    assert abs(float(label.float().mean()) - 0.75) < 0.02
