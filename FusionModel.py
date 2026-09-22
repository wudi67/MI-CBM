import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from math import pi
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, track


console = Console()
QuantumCircuit = None
SparsePauliOp = None
GenericBackendV2 = None
NoiseModel = None
Estimator = None
ParameterVector = None
transpile = None
FakeTorino = None
FakeKyiv = None
_QISKIT_READY = False


REQUIRED_QNET_ARGS = ("task", "n_qubits", "n_layers", "backend", "device")


DEFAULT_QNET_ARGS = {
    "option": "mix_reg",
    "regular": True,
    "fold": 1,
    "kernel": 7,
    "noise": False,
    "shots": 10000,
    "name": "yorktown",
    "batch_size": 1,
    "qlr": 1e-2,
    "enable_group_avg": False,
    "qubit_group_sizes": None,
    "group_avg_num_groups": 2,
    "estimator_job_chunk_size": None,
    "disable_progress": False,
    "quiet": False,
}


class QNetArgs(SimpleNamespace):
    """Arguments object compatible with QNet without requiring Arguments.py.

    Circuit-defining and placement args are intentionally required so caller
    mistakes fail loudly instead of silently falling back to a different model.
    """

    def __init__(self, **kwargs):
        missing = [
            name for name in REQUIRED_QNET_ARGS
            if name not in kwargs or kwargs[name] is None
        ]
        if missing:
            raise ValueError(
                "Missing required QNet arguments: " + ", ".join(missing)
            )
        values = dict(DEFAULT_QNET_ARGS)
        values.update(kwargs)
        values["n_qubits"] = int(values["n_qubits"])
        values["n_layers"] = int(values["n_layers"])
        values["device"] = torch.device(values["device"])
        if values["n_qubits"] <= 0:
            raise ValueError("n_qubits must be positive")
        if values["n_layers"] <= 0:
            raise ValueError("n_layers must be positive")
        super().__init__(**values)


def normalize_arguments(arguments=None, **overrides):
    values = dict(DEFAULT_QNET_ARGS)
    if arguments is None:
        pass
    elif isinstance(arguments, dict):
        values.update(arguments)
    else:
        try:
            values.update(vars(arguments))
        except TypeError as exc:
            raise TypeError(
                "QNet arguments must be a dict, argparse.Namespace, SimpleNamespace, "
                "or another object supported by vars()."
            ) from exc
    values.update(overrides)
    return QNetArgs(**values)


def _progress_disabled(args=None):
    return bool(getattr(args, "disable_progress", False))


def _track(sequence, description, args=None, total=None):
    return track(
        sequence,
        description=description,
        total=total,
        console=console,
        transient=True,
        disable=_progress_disabled(args),
    )


def _make_progress(args=None):
    return Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=24),
        TextColumn("{task.completed:>3.0f}/{task.total:>3.0f}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=_progress_disabled(args),
    )


def _log(message, args=None, style="cyan"):
    if not bool(getattr(args, "quiet", False)):
        console.print(message, style=style)


def _require_qiskit():
    global QuantumCircuit, SparsePauliOp, GenericBackendV2, NoiseModel
    global Estimator, ParameterVector, transpile, FakeTorino, FakeKyiv
    global _QISKIT_READY

    if _QISKIT_READY:
        return
    try:
        from qiskit import QuantumCircuit as _QuantumCircuit
        from qiskit import transpile as _transpile
        from qiskit.quantum_info import SparsePauliOp as _SparsePauliOp
        from qiskit.providers.fake_provider import GenericBackendV2 as _GenericBackendV2
        from qiskit_aer.noise import NoiseModel as _NoiseModel
        from qiskit_aer.primitives import Estimator as _Estimator
        from qiskit.circuit import ParameterVector as _ParameterVector
    except ImportError as exc:
        raise ImportError(
            "The Qiskit backend requires qiskit, qiskit-aer, and their provider packages."
        ) from exc

    try:
        from qiskit_ibm_runtime.fake_provider import FakeTorino as _FakeTorino
        from qiskit_ibm_runtime.fake_provider import FakeKyiv as _FakeKyiv
    except ImportError:
        try:
            from qiskit.providers.fake_provider import FakeTorino as _FakeTorino
            from qiskit.providers.fake_provider import FakeKyiv as _FakeKyiv
        except ImportError:
            _FakeTorino = None
            _FakeKyiv = None

    QuantumCircuit = _QuantumCircuit
    SparsePauliOp = _SparsePauliOp
    GenericBackendV2 = _GenericBackendV2
    NoiseModel = _NoiseModel
    Estimator = _Estimator
    ParameterVector = _ParameterVector
    transpile = _transpile
    FakeTorino = _FakeTorino
    FakeKyiv = _FakeKyiv
    _QISKIT_READY = True


def _require_pennylane():
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError("The PennyLane backend requires pennylane.") from exc
    return qml


def gen_arch(change_code, base_code):        # start from 1, not 0
    # arch_code = base_code[1:] * base_code[0]
    n_qubits = base_code[0]    
    arch_code = ([i for i in range(2, n_qubits+1, 1)] + [1]) * base_code[1]
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]

        for i in range(len(change_code)):
            q = change_code[i][0]  # the qubit changed
            for id, t in enumerate(change_code[i][1:]):
                arch_code[q - 1 + id * n_qubits] = t
    return arch_code

def prune_single(change_code):
    single_dict = {}
    single_dict['current_qubit'] = []
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]
        length = len(change_code[0])
        change_code = np.array(change_code)
        change_qbit = change_code[:,0] - 1
        change_code = change_code.reshape(-1, length)    
        single_dict['current_qubit'] = change_qbit
        j = 0
        for i in change_qbit:            
            single_dict['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(-1, 2).transpose(1,0)
            j += 1
    return single_dict

def translator(single_code, enta_code, trainable, arch_code, fold=1):
    single_code = qubit_fold(single_code, 0, fold)
    enta_code = qubit_fold(enta_code, 1, fold)
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    updated_design = {}
    updated_design = prune_single(single_code)
    net = gen_arch(enta_code, arch_code) 

    if trainable == 'full' or enta_code == None:
        updated_design['change_qubit'] = None
    else:
        if type(enta_code[0]) != type([]): enta_code = [enta_code]
        updated_design['change_qubit'] = enta_code[-1][0]

    # number of layers
    updated_design['n_layers'] = n_layers

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(n_qubits):
            if net[j + layer * n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * n_qubits]-1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * n_qubits])-1, j])

    updated_design['total_gates'] = updated_design['n_layers'] * n_qubits * 2
    return updated_design

def single_enta_to_design(single, enta, arch_code, fold=1):
    """
    Generate a design list usable by QNET from single and enta codes

    Args:
        single: Single-qubit gate encoding, format: [[qubit, gate_config_layer0, gate_config_layer1, ...], ...]
                Each two bits of gate_config represent a layer: 00=Identity, 01=U3, 10=data, 11=data+U3
        enta: Two-qubit gate encoding, format: [[qubit, target_layer0, target_layer1, ...], ...]
              Each value represents the target qubit position in that layer
        arch_code_fold: [n_qubits, n_layers]

    Returns:
        design: List containing quantum circuit design info, each element is (gate_type, [wire_indices], layer)
    """
    design = []
    n_qubits, n_layers = arch_code
    
    # d = 1
    # for row in single:
    #     row[-2*d:] = [0] * (2*d)
    # for row in enta:
    #     row[-1*d:] = [row[0]] * d

    single = qubit_fold(single, 0, fold)
    enta = qubit_fold(enta, 1, fold)

    

    # Process each layer
    for layer in range(n_layers):
        # First process single-qubit gates
        for qubit_config in single:
            qubit = qubit_config[0] - 1  # Convert to 0-based index
            # The config for each layer is at position: 1 + layer*2 and 1 + layer*2 + 1
            config_start_idx = 1 + layer * 2
            if config_start_idx + 1 < len(qubit_config):
                gate_config = f"{qubit_config[config_start_idx]}{qubit_config[config_start_idx + 1]}"

                if gate_config == '01':  # U3
                    design.append(('U3', [qubit], layer))
                elif gate_config == '10':  # data
                    design.append(('data', [qubit], layer))
                elif gate_config == '11':  # data+U3
                    design.append(('data', [qubit], layer))
                    design.append(('U3', [qubit], layer))
                # 00 (Identity) skip

        # Then process two-qubit gates
        for qubit_config in enta:
            control_qubit = qubit_config[0] - 1  # Convert to 0-based index
            # The target qubit position in the list: 1 + layer
            target_idx = 1 + layer
            if target_idx < len(qubit_config):
                target_qubit = qubit_config[target_idx] - 1  # Convert to 0-based index

                # If control and target qubits are different, add C(U3) gate
                if control_qubit != target_qubit:
                    design.append(('C(U3)', [control_qubit, target_qubit], layer))
                # If same, skip (equivalent to Identity)

    return design

def cir_to_matrix(x, y, arch_code, fold=1):
    # x = qubit_fold(x, 0, fold)
    # y = qubit_fold(y, 1, fold)

    qubits = int(arch_code[0] / fold)
    layers = arch_code[1]
    entangle = gen_arch(y, [qubits, layers])
    entangle = np.array([entangle]).reshape(layers, qubits).transpose(1,0)
    single = np.ones((qubits, 2*layers))
    # [[1,1,1,1]
    #  [2,2,2,2]
    #  [3,3,3,3]
    #  [0,0,0,0]]

    if x != None:
        if type(x[0]) != type([]):
            x = [x]    
        x = np.array(x)
        index = x[:, 0] - 1
        index = [int(index[i]) for i in range(len(index))]
        single[index] = x[:, 1:]
    arch = np.insert(single, [(2 * i) for i in range(1, layers+1)], entangle, axis=1)
    return arch.transpose(1, 0)

def shift_ith_element_right(original_list, i):
    """
    对列表中每个item的第i个元素进行循环右移一位
    
    Args:
        original_list: 原始列表，如 [[3, 0, 5], [4, 3, 6], [5, 1, 7], [1, 2, 8]]
        i: 要循环右移的元素索引，如 i=1 表示第二个元素
   
    """   
    ith_elements = [item[i] for item in original_list]    
    # 循环右移一位：最后一个元素移到开头
    shifted_ith = [ith_elements[-1]] + ith_elements[:-1]    
    result = [item[:i] + [shifted_ith[idx]] + item[i+1:] for idx, item in enumerate(original_list)]
    return result

def qubit_fold(jobs, phase, fold=1):
    if fold > 1:
        job_list = []
        for job in jobs:            
            if phase == 0:
                q = job[0]
                job_list += [[fold*(q-1)+1+i] + job[1:] for i in range(0, fold)]
            else:
                job = [i-1 for i in job]
                q = job[0]
                indices = [i for i, x in enumerate(job) if x < q]
                enta = [[fold*j+i+1 for j in job] for i in range(0,fold)]
                for i in indices:
                    enta = shift_ith_element_right(enta, i)
                job_list += enta
    else:
        job_list = jobs
    return job_list

class TQLayer(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits        
        self.uploading = [tq.GeneralEncoder(self.data_uploading(i)) for i in range(self.n_wires)]

        self.q_params_rot = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3))  # each U3 gate needs 3 parameters
        self.q_params_enta = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3))  # each CU3 gate needs 3 parameters
        
        self.measure = tq.MeasureAll(tq.PauliZ)

    def data_uploading(self, qubit):
        input = [
            {"input_idx": [0], "func": "ry", "wires": [qubit]},
            {"input_idx": [1], "func": "rz", "wires": [qubit]},
            {"input_idx": [2], "func": "rx", "wires": [qubit]},
            {"input_idx": [3], "func": "ry", "wires": [qubit]},
        ]
        return input

    def forward(self, x):
        bsz = x.shape[0]
        kernel_size = self.args.kernel
        task_name = self.args.task        
        if not task_name.startswith('QML'):
            x = F.avg_pool2d(x, kernel_size)  # 'down_sample_kernel_size' = 6
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4, device=x.device)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            else:
                x = x.view(bsz, 4, 4).transpose(1, 2)
        else:
            x = x.view(bsz, self.n_wires, -1)

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        
        for i in range(len(self.design)):
            if self.design[i][0] == 'U3':                
                layer = self.design[i][2]
                qubit = self.design[i][1][0]
                params = self.q_params_rot[layer][qubit].unsqueeze(0)  # 重塑为 [1, 3]
                tqf.u3(qdev, wires=self.design[i][1], params=params)
            elif self.design[i][0] == 'C(U3)':               
                layer = self.design[i][2]
                control_qubit = self.design[i][1][0]
                params = self.q_params_enta[layer][control_qubit].unsqueeze(0)  # 重塑为 [1, 3]
                tqf.cu3(qdev, wires=self.design[i][1], params=params)
            else:   # data uploading: if self.design[i][0] == 'data'
                j = int(self.design[i][1][0])
                self.uploading[j](qdev, x[:,j])
        out = self.measure(qdev)
        # if task_name.startswith('QML'):            
        #     out = out[:, :2]    # only take the first two measurements for binary classification            
        return out

class EstimatorQiskitLayer(nn.Module):
    SEED = 170

    def __init__(self, arguments, design, shots=10000):
        super().__init__()
        _require_qiskit()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        self.n_layers = self.args.n_layers
        self.shots = getattr(self.args, 'shots', shots)
        configured_chunk_size = getattr(self.args, 'estimator_job_chunk_size', None)
        if configured_chunk_size is None:
            self.estimator_job_chunk_size = max(1, min(8, getattr(self.args, 'batch_size', 8)))
        else:
            self.estimator_job_chunk_size = max(1, int(configured_chunk_size))

        # Trainable parameters with identical structure to other layers
        self.q_params_rot = nn.Parameter(pi * torch.rand(self.n_layers, self.n_wires, 3), requires_grad=True)
        self.q_params_enta = nn.Parameter(pi * torch.rand(self.n_layers, self.n_wires, 3), requires_grad=True)

        # Reuse original circuit construction logic to ensure consistent structure
        self.qc_template, self.data_params, self.u3_param_map, self.cu3_param_map = self._build_parametric_circuit()
        self.observables = self._prebuild_observables()

        # Initialize backend and noise model from the same chip config.
        self._init_backend_and_noisemodel(arguments.name)
        self._init_estimator()
        self._cache_transpiled_template()

    def _init_backend_and_noisemodel(self, name):
        if name == 'heron_r1':
            if FakeTorino is None:
                raise ImportError(
                    "FakeTorino backend is not available. Install qiskit-ibm-runtime or use a Qiskit "
                    "release that still bundles FakeTorino in qiskit.providers.fake_provider."
                )
            self.backend = FakeTorino()
            self.noise_model = NoiseModel.from_backend(self.backend) if self.args.noise else None
            return

        if name == 'eagle_r3':
            if FakeKyiv is None:
                raise ImportError(
                    "FakeKyiv backend is not available. Install qiskit-ibm-runtime."
                )
            self.backend = FakeKyiv()
            self.noise_model = NoiseModel.from_backend(self.backend) if self.args.noise else None
            return

        try:
            from MyNoiseModel import create_noise_model, get_chip_config
        except ImportError as exc:
            raise ImportError(
                "The generic Qiskit noise backend requires MyNoiseModel.py."
            ) from exc

        cfg = get_chip_config(name)
        self.backend = GenericBackendV2(
            num_qubits=self.n_wires,
            basis_gates=list(cfg['basis_gates']),
            coupling_map=cfg['coupling_map'],
        )
        self.noise_model = create_noise_model(name) if self.args.noise else None

    def _init_estimator(self):
        """Initialize Estimator compatible with GenericBackendV2."""
        self.transpile_options = {
            'seed_transpiler': self.SEED,
        }
        backend_options = {
            'method': 'density_matrix',  # Use density matrix method for noise simulation
        }
        if self.noise_model is not None:
            backend_options['noise_model'] = self.noise_model

        self.estimator = Estimator(
            backend_options=backend_options,
            run_options={
                'shots': self.shots,
                'seed': self.SEED,
            },
            transpile_options=self.transpile_options,
            skip_transpilation=True,
        )

    def _cache_transpiled_template(self):
        """Transpile the parameterized circuit once and cache mapped observables."""
        self.transpiled_qc_template = transpile(
            self.qc_template,
            backend=self.backend,
            **self.transpile_options,
        )
        physical_qubit_indices = self._extract_physical_qubit_indices(self.transpiled_qc_template)
        self.transpiled_observables = self.create_pauli_observables(
            physical_qubit_indices,
            self.transpiled_qc_template.num_qubits,
        )
        self.transpiled_param_sources = self._build_transpiled_param_sources()

    def _build_transpiled_param_sources(self):
        param_sources = {}
        for qubit, param_vec in self.data_params.items():
            for idx, param in enumerate(param_vec):
                param_sources[str(param)] = ('data', qubit, idx)
        for (layer, qubit), param_vec in self.u3_param_map.items():
            for idx, param in enumerate(param_vec):
                param_sources[str(param)] = ('u3', layer, qubit, idx)
        for (layer, control_qubit), param_vec in self.cu3_param_map.items():
            for idx, param in enumerate(param_vec):
                param_sources[str(param)] = ('cu3', layer, control_qubit, idx)
        return [param_sources[str(param)] for param in self.transpiled_qc_template.parameters]

    def _extract_physical_qubit_indices(self, circuit):
        """Return logical-to-physical qubit mapping from a transpiled circuit layout."""
        layout = getattr(circuit, 'layout', None)
        initial_layout = getattr(layout, 'initial_layout', None)
        if initial_layout is None:
            return list(range(len(self.qc_template.qubits)))

        virtual_bits = initial_layout.get_virtual_bits()
        physical_qubit_indices = []
        for logical_idx, logical_qubit in enumerate(self.qc_template.qubits):
            physical_qubit_indices.append(virtual_bits.get(logical_qubit, logical_idx))
        return physical_qubit_indices

    def _build_parametric_circuit(self):
        """Construct parametric quantum circuit with consistent structure"""
        qc = QuantumCircuit(self.n_wires)
        data_param_map = {}
        u3_param_map = {}
        cu3_param_map = {}

        for i in _track(range(len(self.design)), "Building circuit", self.args):
            elem = self.design[i]
            if elem[0] == 'U3':
                layer = elem[2]
                qubit = elem[1][0]
                param_key = (layer, qubit)
                if param_key not in u3_param_map:
                    u3_params = ParameterVector(f'u3_l{layer}q{qubit}', length=3)
                    u3_param_map[param_key] = u3_params
                theta, phi, lam = u3_param_map[param_key]
                qc.u(theta, phi, lam, qubit)
            elif elem[0] == 'C(U3)':
                layer = elem[2]
                control_qubit = elem[1][0]
                target_qubit = elem[1][1]
                param_key = (layer, control_qubit)
                if param_key not in cu3_param_map:
                    cu3_params = ParameterVector(f'cu3_l{layer}cq{control_qubit}', length=3)
                    cu3_param_map[param_key] = cu3_params
                theta, phi, lam = cu3_param_map[param_key]
                qc.cu(theta, phi, lam, 0, control_qubit, target_qubit)
            else:
                j = int(elem[1][0])
                if j not in data_param_map:
                    data_param_map[j] = ParameterVector(f'data_q{j}', length=4)
                qc.ry(data_param_map[j][0], j)
                qc.rz(data_param_map[j][1], j)
                qc.rx(data_param_map[j][2], j)
                qc.ry(data_param_map[j][3], j)
        return qc, data_param_map, u3_param_map, cu3_param_map

    def _prebuild_observables(self):
        """Pre-build Pauli observables for expectation value calculation"""
        observables = []
        for q in range(self.n_wires):
            pauli_str = 'I' * q + 'Z' + 'I' * (self.n_wires - q - 1)
            observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
            observables.append(observable)
        return observables

    def _preprocess_x(self, x):
        """Preprocess input data following the original pipeline"""
        bsz = x.shape[0]
        kernel_size = self.args.kernel
        task_name = self.args.task
        if not task_name.startswith('QML'):
            x = F.avg_pool2d(x, kernel_size)
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4, device=x.device)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            else:
                x = x.view(bsz, 4, 4).transpose(1, 2)
        else:
            x = x.view(bsz, self.n_wires, -1)
        return x

    def create_pauli_observables(self, physical_qubit_indices, total_qubits=None):
        """
        Create Pauli-Z observables based on physical qubit mapping
        physical_qubit_indices = [0, 1, 3, 2] means:
            - Logical qubit 0 maps to physical qubit 0 -> 'ZIII'
            - Logical qubit 1 maps to physical qubit 1 -> 'IZII'
            - Logical qubit 2 maps to physical qubit 3 -> 'IIIZ'
            - Logical qubit 3 maps to physical qubit 2 -> 'IIZI'
        """
        observables = []
        if total_qubits is None:
            total_qubits = len(physical_qubit_indices)

        # Create one observable per logical qubit on the transpiled physical layout.
        for physical_qubit_idx in physical_qubit_indices:
            pauli_chars = ['I'] * total_qubits
            pauli_chars[total_qubits - 1 - physical_qubit_idx] = 'Z'
            pauli_str = ''.join(pauli_chars)
            observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
            observables.append(observable)
        # if self.args.task.startswith('QML'):            
        #     observables = observables[:2]  # Only take the first two observables for binary classification
        return observables

    def forward(self, x):
        """Forward pass: parameter binding and expectation value calculation via Estimator"""
        device = x.device
        x_pre = self._preprocess_x(x)
        bsz = x_pre.shape[0]

        # Parameter binding: assign parameters for each sample
        x_np = x_pre.detach().cpu().numpy()
        u3_np = self.q_params_rot.detach().cpu().numpy()
        cu3_np = self.q_params_enta.detach().cpu().numpy()

        sample_parameter_values = []
        num_observables = len(self.transpiled_observables)

        for batch_idx in _track(range(bsz), "Preparing estimator inputs", self.args):
            param_values = []
            for source in self.transpiled_param_sources:
                if source[0] == 'data':
                    _, qubit, idx = source
                    param_values.append(float(x_np[batch_idx, qubit, idx]))
                elif source[0] == 'u3':
                    _, layer, qubit, idx = source
                    param_values.append(float(u3_np[layer, qubit, idx]))
                else:
                    _, layer, control_qubit, idx = source
                    param_values.append(float(cu3_np[layer, control_qubit, idx]))
            sample_parameter_values.append(param_values)

        batch_results = []
        with _make_progress(self.args) as progress:
            task_id = progress.add_task("Executing estimator jobs", total=bsz)
            for chunk_start in range(0, bsz, self.estimator_job_chunk_size):
                chunk_end = min(chunk_start + self.estimator_job_chunk_size, bsz)
                chunk_parameter_values = []
                chunk_circuits = []
                chunk_observables = []

                for sample_idx in range(chunk_start, chunk_end):
                    param_values = sample_parameter_values[sample_idx]
                    for observable in self.transpiled_observables:
                        chunk_circuits.append(self.transpiled_qc_template)
                        chunk_observables.append(observable)
                        chunk_parameter_values.append(param_values)

                job = self.estimator.run(
                    chunk_circuits,
                    chunk_observables,
                    parameter_values=chunk_parameter_values,
                )
                result = job.result()
                chunk_results = np.asarray(result.values, dtype=np.float32).reshape(chunk_end - chunk_start, num_observables)
                batch_results.append(chunk_results)
                progress.update(task_id, advance=chunk_end - chunk_start)
                per_chunk_callback = getattr(self.args, 'per_chunk_callback', None)
                if per_chunk_callback is not None:
                    per_chunk_callback(chunk_start, chunk_end, chunk_results)

        batch_results = np.concatenate(batch_results, axis=0)

        # Convert results to PyTorch tensor
        output = torch.tensor(batch_results, dtype=torch.float32, device=device)
        return output

#-------------------------------------------------------------------------------------
def _pennylane_preprocess_x(x, args, n_wires):
    bsz = x.shape[0]
    kernel_size = args.kernel
    task_name = args.task
    if not task_name.startswith('QML'):
        x = F.avg_pool2d(x, kernel_size)
        if kernel_size == 4:
            x = x.view(bsz, 6, 6)
            tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4, device=x.device)), dim=-1)
            x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
        else:
            x = x.view(bsz, 4, 4).transpose(1, 2)
    else:
        x = x.view(bsz, n_wires, -1)
    return x


def _apply_pennylane_design(layer, x, qml):
    for i in range(len(layer.design)):
        if layer.design[i][0] == 'U3':
            design_layer = layer.design[i][2]
            qubit = layer.design[i][1][0]
            phi = layer.q_params_rot[design_layer, qubit, 0]
            theta = layer.q_params_rot[design_layer, qubit, 1]
            omega = layer.q_params_rot[design_layer, qubit, 2]
            qml.Rot(phi, theta, omega, wires=qubit)
        elif layer.design[i][0] == 'C(U3)':
            design_layer = layer.design[i][2]
            control_qubit = layer.design[i][1][0]
            target_qubit = layer.design[i][1][1]
            phi = layer.q_params_enta[design_layer, control_qubit, 0]
            theta = layer.q_params_enta[design_layer, control_qubit, 1]
            omega = layer.q_params_enta[design_layer, control_qubit, 2]
            qml.CRot(phi, theta, omega, wires=[control_qubit, target_qubit])
        else:  # data uploading: if self.design[i][0] == 'data'
            j = int(layer.design[i][1][0])
            qml.RY(x[j, 0].detach(), wires=j)
            qml.RZ(x[j, 1].detach(), wires=j)
            qml.RX(x[j, 2].detach(), wires=j)
            qml.RY(x[j, 3].detach(), wires=j)

    return [qml.expval(qml.PauliZ(i)) for i in range(layer.args.n_qubits)]

class PennylaneLayer(nn.Module):
    def __init__(self, arguments, design):
        super().__init__()
        self.qml = _require_pennylane()
        self.args = arguments
        self.design = design
        self.u3_params = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3), requires_grad=True)  # Each U3 gate needs 3 parameters
        self.cu3_params = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3), requires_grad=True) # Each CU3 gate needs 3 parameters
        self.quantum_net = self._build_quantum_net()

    def _build_quantum_net(self):
        dev = self.qml.device("lightning.qubit", wires=self.args.n_qubits)

        @self.qml.qnode(dev, interface="torch", diff_method="adjoint")
        def quantum_net(x):
            return _apply_pennylane_design(self, x, self.qml)

        return quantum_net

    def forward(self, x):
        x = _pennylane_preprocess_x(x, self.args, self.args.n_qubits)
        output_list = []
        for batch in range(x.size(0)):  # Use actual batch size
            x_batch = x[batch]
            output = self.quantum_net(x_batch)
            q_out = torch.stack([output[i] for i in range(len(output))]).float()
            output_list.append(q_out)
        outputs = torch.stack(output_list)

        return outputs


class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = normalize_arguments(arguments)
        self.design = design
        self.enable_group_avg = getattr(self.args, 'enable_group_avg', False)
        self.qubit_group_sizes = getattr(self.args, 'qubit_group_sizes', None)
        self.group_avg_num_groups = getattr(self.args, 'group_avg_num_groups', 2)
        if self.args.backend == 'tq':
            _log("Run with TorchQuantum backend.", self.args)
            self.QuantumLayer = TQLayer(self.args, self.design)
        elif self.args.backend == 'qi':
            _log("Run with Qiskit quantum backend.", self.args)
            self.QuantumLayer = EstimatorQiskitLayer(
                self.args, self.design, shots=self.args.shots
            )
        else:   # PennyLane or others
            _log("Run with PennyLane quantum backend or others.", self.args)
            self.QuantumLayer = PennylaneLayer(self.args, self.design)

    def _resolve_group_sizes(self, n_qubits):
        if not self.enable_group_avg and self.qubit_group_sizes is None:
            return None

        if self.qubit_group_sizes is not None:
            group_sizes = [int(size) for size in self.qubit_group_sizes]
            if any(size <= 0 for size in group_sizes):
                raise ValueError("All values in qubit_group_sizes must be positive.")
            if sum(group_sizes) != n_qubits:
                raise ValueError(
                    f"Sum of qubit_group_sizes ({sum(group_sizes)}) must equal n_qubits ({n_qubits})."
                )
            return group_sizes

        num_groups = int(self.group_avg_num_groups)
        if num_groups <= 0:
            raise ValueError("group_avg_num_groups must be a positive integer.")
        if n_qubits % num_groups != 0:
            raise ValueError(
                f"n_qubits ({n_qubits}) must be divisible by group_avg_num_groups ({num_groups})."
            )
        group_size = n_qubits // num_groups
        return [group_size] * num_groups

    def _apply_group_average(self, exp_val, n_qubits):
        group_sizes = self._resolve_group_sizes(n_qubits)
        if group_sizes is None:
            return exp_val

        if exp_val.dim() != 2 or exp_val.size(1) != n_qubits:
            raise ValueError(
                f"Expected quantum output shape [batch, {n_qubits}], got {tuple(exp_val.shape)}."
            )

        grouped = torch.split(exp_val, group_sizes, dim=1)
        return torch.stack([chunk.mean(dim=1) for chunk in grouped], dim=1)

    def forward(self, x_image, n_qubits, task_name):
        # exp_val = self.QuantumLayer(x_image, n_qubits, task_name)
        exp_val = self.QuantumLayer(x_image)
        exp_val = self._apply_group_average(exp_val, n_qubits)
        output = F.log_softmax(exp_val, dim=1)
        
        return output
