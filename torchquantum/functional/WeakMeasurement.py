import functools
import torch
import numpy as np

from typing import Callable, Union, Optional, List, Dict, TYPE_CHECKING
from ..macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from ..util.utils import pauli_eigs, diag
from torchpack.utils.logging import logger
from torchquantum.util import normalize_statevector

from .gate_wrapper import gate_wrapper, apply_unitary_einsum, apply_unitary_bmm

if TYPE_CHECKING:
    from torchquantum.device import QuantumDevice
else:
    QuantumDevice = None


def weak_measurement_matrix(params):
    """Compute matrix for weak measurement gate.

    Args:
        params (torch.Tensor): The measurement strength parameter g.

    Returns:
        torch.Tensor: The computed matrix.
    """
    g = params.type(C_DTYPE)
    
    # WeakMeasurement 矩阵: [[1, exp(-g²/2)], [exp(-g²/2), 1]]
    exp_term = torch.exp(-g * g / 2)
    
    return torch.stack(
        [torch.cat([torch.ones_like(g), exp_term], dim=-1), 
         torch.cat([exp_term, torch.ones_like(g)], dim=-1)], dim=-2
    ).squeeze(0)


_weak_measurement_mat_dict = {
    "weak_measurement": weak_measurement_matrix,
}


def _format_wires(wires, n_wires):
    if wires is None:
        return list(range(n_wires))
    if isinstance(wires, int):
        wires = [wires]

    formatted_wires = [int(wire) for wire in wires]
    for wire in formatted_wires:
        if wire < 0 or wire >= n_wires:
            raise ValueError(f"wire index {wire} is out of range for {n_wires} wires")
    return formatted_wires


def hamming_distance_matrix(n_wires, wires=None, device=None, dtype=torch.float32):
    """Build the Hamming-distance matrix over computational-basis states.

    Args:
        n_wires (int): Number of qubits in the density matrix.
        wires (Union[int, List[int]], optional): Qubits included in the weak
            measurement. If None, all qubits are included.
        device: Target torch device.
        dtype: Real-valued dtype of the returned matrix.

    Returns:
        torch.Tensor: A [2**n_wires, 2**n_wires] matrix whose entry (a, b) is
        the Hamming distance between basis states a and b on the selected wires.
    """
    wire_list = _format_wires(wires, n_wires)
    dim = 2 ** n_wires
    basis = torch.arange(dim, device=device, dtype=torch.long)
    xor = basis.unsqueeze(1) ^ basis.unsqueeze(0)
    distances = torch.zeros(dim, dim, device=device, dtype=dtype)

    for wire in wire_list:
        distances = distances + ((xor >> wire) & 1).to(dtype)

    return distances


def weak_measurement_mask(n_wires, params, wires=None, hamming_dist=None, device=None, dtype=None):
    """Return the analytical weak-measurement Hadamard mask.

    The mask implements rho'[a, b] = exp(-g**2 * d_H(a, b) / 2) * rho[a, b].
    A scalar ``params`` returns a [D, D] mask. A batch-shaped ``params`` with
    length B returns a [B, D, D] mask.
    """
    if params is None:
        raise ValueError("params must be provided for analytical weak measurement")

    if dtype is None:
        if isinstance(params, torch.Tensor) and params.is_floating_point():
            real_dtype = params.dtype
        else:
            real_dtype = torch.float32
    else:
        real_dtype = dtype

    if hamming_dist is None:
        hamming_dist = hamming_distance_matrix(
            n_wires=n_wires, wires=wires, device=device, dtype=real_dtype
        )
    else:
        if device is not None:
            hamming_dist = hamming_dist.to(device=device)
        hamming_dist = hamming_dist.to(dtype=real_dtype)

    if isinstance(params, torch.Tensor):
        g = params.to(device=hamming_dist.device)
        if g.is_complex():
            g = g.real
        g = g.to(dtype=real_dtype)
    else:
        g = torch.tensor(params, device=hamming_dist.device, dtype=real_dtype)

    if g.numel() == 1:
        return torch.exp(-0.5 * g.reshape(()) ** 2 * hamming_dist)

    return torch.exp(-0.5 * g.reshape(-1, 1, 1) ** 2 * hamming_dist.unsqueeze(0))


def analytical_weak_measurement(
    q_device,
    wires=None,
    params=None,
    hamming_dist=None,
    mask=None,
):
    """Apply analytical weak measurement to a density-matrix quantum device.

    This is an alternative implementation for the closed-form dephasing-style
    weak measurement. It does not modify the existing ``weak_measurement``
    function and does not perform trace normalization because the diagonal
    entries are unchanged by the mask.
    """
    if not hasattr(q_device, "densities"):
        raise ValueError("analytical_weak_measurement requires q_device.densities")

    density_shape = q_device.densities.shape
    batch_size = density_shape[0]
    n_wires = (q_device.densities.dim() - 1) // 2
    dim = 2 ** n_wires
    rho_2d = q_device.densities.reshape(batch_size, dim, dim)
    real_dtype = rho_2d.real.dtype

    if mask is None:
        mask = weak_measurement_mask(
            n_wires=n_wires,
            params=params,
            wires=wires,
            hamming_dist=hamming_dist,
            device=rho_2d.device,
            dtype=real_dtype,
        )
    else:
        mask = mask.to(device=rho_2d.device)
        if mask.is_complex():
            mask = mask.real
        mask = mask.to(dtype=real_dtype)

    if mask.dim() == 2:
        updated_rho_2d = rho_2d * mask.to(dtype=rho_2d.dtype).unsqueeze(0)
    elif mask.dim() == 3:
        if mask.shape[0] not in (1, batch_size):
            raise ValueError(
                f"batch mask has batch size {mask.shape[0]}, expected 1 or {batch_size}"
            )
        updated_rho_2d = rho_2d * mask.to(dtype=rho_2d.dtype)
    else:
        raise ValueError("mask must have shape [D, D] or [B, D, D]")

    q_device.densities = updated_rho_2d.reshape(density_shape)


def weak_measurement(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the weak measurement operation.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = "weak_measurement"
    mat = weak_measurement_matrix
    
    # 使用标准的 gate_wrapper，就像其他单量子比特门一样
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )
    
    # 非酉操作后的归一化：除以 trace
    if hasattr(q_device, 'densities'):
        normalize_density_after_measurement(q_device)


def normalize_density_after_measurement(q_device):
    """归一化密度矩阵，除以其迹 - 批量处理避免就地操作"""
    batch_size = q_device.densities.shape[0]
    n_qubits = (q_device.densities.dim() - 1) // 2
    total_dim = 2 ** n_qubits
    
    # 批量重塑为 2D 矩阵：[batch_size, total_dim, total_dim]
    rho_2d = q_device.densities.reshape(batch_size, total_dim, total_dim)
    
    # 批量计算迹：[batch_size]
    traces = torch.diagonal(rho_2d, dim1=-2, dim2=-1).sum(dim=-1).real
    
    # 避免除零
    traces = torch.where(torch.abs(traces) > 1e-10, traces, torch.ones_like(traces))
    
    # 批量归一化 - 创建新张量
    normalized_rho_2d = rho_2d / traces.unsqueeze(-1).unsqueeze(-1)
    
    # 重塑回原始形状并替换（不是就地操作）
    q_device.densities = normalized_rho_2d.reshape(q_device.densities.shape)
