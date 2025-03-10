import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import Operator


# -------------------------------------------------------------------------
# Constants for encoding Lipschitz (fixed encodings)
ENCODING_LIPSCHITZ = {
    "amplitude": 2.0,
    "angle": np.pi,
    "basis": lambda num_qubits: 2.0 * np.sqrt(num_qubits),
}


class ClassicalLayer:
    def __init__(
        self, W: np.ndarray, b: np.ndarray, activation: str = "relu", slope_bounds=None
    ):
        self.W = W
        self.b = b
        self.activation = activation.lower()
        if slope_bounds is not None:
            self.alpha, self.beta = slope_bounds
        else:
            if self.activation == "relu":
                self.alpha, self.beta = 0.0, 1.0
            elif self.activation == "linear":
                self.alpha, self.beta = 1.0, 1.0
            elif self.activation == "tanh":
                self.alpha, self.beta = 0.0, 1.0
            else:
                self.alpha, self.beta = 0.0, 1.0

    def lipschitz_constant(self):
        if self.alpha == self.beta:
            normW = np.linalg.norm(self.W, 2)
            return float(self.beta * normW)

        m = self.W.shape[0]
        gamma = cp.Variable(nonneg=True)
        constraints = []
        for j in range(m):
            wj = self.W[j].reshape(1, -1)
            constraints.append(cp.norm(wj, 2) * self.beta <= gamma)
        problem = cp.Problem(cp.Minimize(gamma), constraints)
        problem.solve(solver=cp.SCS, verbose=False)
        return float(gamma.value)


class QuantumLayer:
    def __init__(
        self,
        measurement_ops,
        encoding="amplitude",
        num_qubits=None,
        channel=None,
        q_weights=None,
    ):
        self.measurement_ops = measurement_ops
        self.encoding = encoding.lower()
        if num_qubits is not None:
            self.num_qubits = num_qubits
        else:
            if measurement_ops is not None and len(measurement_ops) > 0:
                d = measurement_ops[0].shape[0]
                self.num_qubits = int(np.log2(d))
            else:
                self.num_qubits = 1
        self.channel = channel
        self.q_weights = q_weights  # trainable quantum weights (as a numpy array)
        if self.encoding in ENCODING_LIPSCHITZ:
            if self.encoding == "basis":
                self.L_enc = ENCODING_LIPSCHITZ["basis"](self.num_qubits)
            else:
                self.L_enc = float(ENCODING_LIPSCHITZ[self.encoding])
        else:
            self.L_enc = 1.0

    def lipschitz_constant(self):
        if self.measurement_ops is None or len(self.measurement_ops) == 0:
            return 1.0

        d = self.measurement_ops[0].shape[0]
        delta = cp.Variable((d, d), complex=True)
        constraints = [delta == cp.conj(delta).T, cp.trace(delta) == 0]
        I = np.eye(d)
        constraints += [I + delta >> 0, I - delta >> 0]
        m_count = len(self.measurement_ops)
        v = []
        for M in self.measurement_ops:
            v.append(cp.trace(M @ delta))
        u = cp.Variable(m_count)
        for i in range(m_count):
            constraints += [u[i] >= cp.real(v[i]), u[i] >= -cp.real(v[i])]
        objective = cp.Maximize(cp.sum(u))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=False)
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            K_q = 1.0
        else:
            max_diff = problem.value
            K_q = 0.5 * max_diff
        return float(K_q * self.L_enc)


# Hybrid Network: computes overall Lipschitz constant by composing layer bounds.
class HybridNetworkLipschitz:
    def __init__(self, layers):
        self.layers = layers

    def compute_lipschitz_constant(self):
        """
        Complete this section according to Theorem 2
        Initially, We are using a plain implementation
        """
        K_total = 1.0
        for layer in self.layers:
            K_total *= layer.lipschitz_constant()
        return K_total


def compute_network_lipschitz(layers):
    H = HybridNetworkLipschitz(layers)
    return H.compute_lipschitz_constant()


# Hybrid Model Extractor: extracts layer parameters from a PyTorch+Qiskit QNN hybrid model.
class HybridModelExtractor:
    def __init__(self, model):
        self.model = model

    def extract(self):
        layers = []
        if hasattr(self.model, "layer_list"):
            modules = self.model.layer_list
        elif isinstance(self.model, nn.Sequential):
            modules = list(self.model)
        else:
            modules = list(self.model.children())

        i = 0
        while i < len(modules):
            mod = modules[i]

            # --- Classical layer extraction ---
            if isinstance(mod, nn.Linear):
                activation = "linear"
                slope_bounds = None
                if i + 1 < len(modules):
                    next_mod = modules[i + 1]
                    if isinstance(next_mod, nn.ReLU):
                        activation = "relu"
                        slope_bounds = (0.0, 1.0)
                    elif isinstance(next_mod, nn.Tanh):
                        activation = "tanh"
                        slope_bounds = (0.0, 1.0)
                    elif isinstance(next_mod, nn.Linear):
                        activation = "linear"
                        slope_bounds = (1.0, 1.0)
                    else:
                        activation = "softmax"
                        slope_bounds = (0.0, 1.0)
                
                W = mod.weight.detach().cpu().numpy()
                b = (
                    mod.bias.detach().cpu().numpy()
                    if mod.bias is not None
                    else np.zeros(W.shape[0])
                )
                clayer = ClassicalLayer(
                    W, b, activation=activation, slope_bounds=slope_bounds
                )
                layers.append(clayer)

            # --- Quantum layer extraction ---
            # We assume quantum layers are wrapped in a TorchConnector.
            elif isinstance(mod, TorchConnector):
                try:
                    # Retrieve the trainable parameters from the TorchConnector.
                    params = {}
                    for name, param in mod.named_parameters():
                        params[name] = param.detach().cpu().numpy()
                    if "weight" not in params:
                        raise ValueError(
                            "No 'weight' parameter found in TorchConnector"
                        )
                    trained_values = params["weight"]

                    # Get the underlying quantum circuit.
                    qc = mod.neural_network.circuit

                    # For a fully bound circuit, assign numerical values to both input and weight parameters.
                    # Here, we set input parameters to zeros.
                    input_values = [0.0] * len(self.model.input_params)
                    param_dict = {
                        p: v for p, v in zip(self.model.input_params, input_values)
                    }
                    param_dict.update(
                        {p: v for p, v in zip(self.model.weight_params, trained_values)}
                    )
                    bound_circuit = qc.assign_parameters(param_dict)

                    # Compute the unitary matrix of the bound circuit.
                    unitary_matrix = Operator(bound_circuit).data

                    # Extract measurement operators from the QNN observables.
                    measurement_ops = [
                        obs.to_matrix() for obs in mod.neural_network.observables
                    ]

                    # Get the encoding; default to "amplitude" if not provided.
                    enc = getattr(mod.neural_network, "encoding", "amplitude")
                    num_qubits = qc.num_qubits

                    # Create a QuantumLayer instance.
                    qlayer = QuantumLayer(
                        measurement_ops=measurement_ops,
                        encoding=enc,
                        num_qubits=num_qubits,
                        channel=None,
                        q_weights=trained_values,
                    )
                    # Optionally, store the computed unitary in the quantum layer.
                    qlayer.unitary_matrix = unitary_matrix

                    layers.append(qlayer)
                except Exception as e:
                    print("Warning: could not extract quantum layer parameters:", e)

            elif isinstance(mod, nn.Sequential):
                sub_extractor = HybridModelExtractor(mod)
                layers.extend(sub_extractor.extract())
            
            i += 1
        return layers


if __name__ == "__main__":
    from robust_training import iris_hybrid_model as model

    extractor = HybridModelExtractor(model())
    extracted_layers = extractor.extract()
    print("Extracted", len(extracted_layers), "layers from the model.")
    for idx, layer in enumerate(extracted_layers):
        if isinstance(layer, ClassicalLayer):
            print(f"Layer {idx + 1}: ClassicalLayer with activation {layer.activation}")
        elif isinstance(layer, QuantumLayer):
            print(
                f"Layer {idx + 1}: QuantumLayer with encoding {layer.encoding}, weights: {layer.q_weights}"
            )

    overall_K = compute_network_lipschitz(extracted_layers)
    print("Overall Lipschitz constant from extracted model:", overall_K)
