import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

# Import only transpile from Qiskit and the Aer backend.
from qiskit import transpile
from qiskit_aer import Aer


# -------------------------------
# Qiskit: Quantum simulation functions for a 5-qubit circuit
# -------------------------------
def simulate_circuit_multi_vector(params, n_qubits=5):
    """
    Build and simulate a 5-qubit variational quantum circuit.
    The circuit uses 10 parameters:
      - First 5 parameters: RX rotations on each qubit.
      - Next 5 parameters: RY rotations on each qubit.
    The circuit applies:
      1. A layer of RX rotations,
      2. A ring of entangling CNOT gates,
      3. A layer of RY rotations.

    The circuit is transpiled for the Aer statevector_simulator backend.
    Returns a numpy array of shape (n_qubits,) with the expectation value of Pauli-Z for each qubit.
    """
    if len(params) != 10:
        raise ValueError("Expected 10 parameters for the 5-qubit VQC.")

    from qiskit import QuantumCircuit  # Local import for clarity.

    qc = QuantumCircuit(n_qubits)

    # First layer: RX rotations on each qubit.
    for i in range(n_qubits):
        qc.rx(params[i], i)

    # Entangling layer: CNOTs in a ring.
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(n_qubits - 1, 0)

    # Second layer: RY rotations on each qubit.
    for i in range(n_qubits):
        qc.ry(params[n_qubits + i], i)

    # Transpile the circuit for the Aer backend.
    backend = Aer.get_backend("statevector_simulator")
    qc_transpiled = transpile(qc, backend)
    # Run the circuit using backend.run(...)
    result = backend.run(qc_transpiled).result()
    state = result.get_statevector(qc_transpiled)
    state = np.asarray(state)  # Explicitly cast to a numpy array

    # Compute the expectation value <Z> for each qubit.
    expectations = np.zeros(n_qubits)
    for qubit in range(n_qubits):
        exp_val = 0.0
        for i, amplitude in enumerate(state):
            prob = np.abs(amplitude) ** 2
            # Extract the bit corresponding to the qubit.
            bit = (i >> qubit) & 1
            exp_val += (1 if bit == 0 else -1) * prob
        expectations[qubit] = exp_val
    return expectations


# -------------------------------
# PyTorch: Custom differentiable quantum module using parameter-shift rule
# -------------------------------
class QuantumFunction5Qubits(nn.Module):
    def forward(self, params):
        # params is a tensor of shape (10,)
        params_np = params.detach().cpu().numpy()
        result = simulate_circuit_multi_vector(params_np)
        return torch.tensor(result, dtype=params.dtype, device=params.device)

    def backward_shift(self, params, grad_output):
        """
        Compute the gradient via the parameter-shift rule.
        For each parameter index, shift that parameter by ±π/2, evaluate the circuit,
        and combine the results with grad_output to yield the gradient.
        """
        shift = np.pi / 2
        params_np = params.detach().cpu().numpy()
        grad_est = np.zeros_like(params_np)
        for i in range(len(params_np)):
            params_plus = params_np.copy()
            params_minus = params_np.copy()
            params_plus[i] += shift
            params_minus[i] -= shift
            f_plus = simulate_circuit_multi_vector(params_plus)
            f_minus = simulate_circuit_multi_vector(params_minus)
            deriv = (f_plus - f_minus) / 2.0  # This is a vector of length 5.
            grad_est[i] = np.dot(grad_output.detach().cpu().numpy(), deriv)
        return torch.tensor(grad_est, dtype=params.dtype, device=params.device)


# Wrap the above functionality in an autograd.Function.
class QuantumModuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params):
        quantum = QuantumFunction5Qubits()
        output = quantum.forward(params)
        ctx.save_for_backward(params)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (params,) = ctx.saved_tensors
        quantum = QuantumFunction5Qubits()
        grad_input = quantum.backward_shift(params, grad_output)
        return grad_input


# -------------------------------
# PyTorch: Define the quantum model module
# -------------------------------
class QuantumModel5Qubits(nn.Module):
    def __init__(self):
        super(QuantumModel5Qubits, self).__init__()
        # This module does not have its own learnable parameters; the parameters come from the encoder.

    def forward(self, params_override):
        # params_override should be a tensor of shape (10,)
        return QuantumModuleFunction.apply(params_override)


# -------------------------------
# Hybrid Quantum-Classical MNIST Classifier with Robust Training
# -------------------------------
class HybridQuantumClassifier(nn.Module):
    def __init__(self):
        super(HybridQuantumClassifier, self).__init__()
        # Encoder: Map 28x28 images to 10 parameters.
        self.encoder = nn.Linear(28 * 28, 10)
        # Quantum module: 5-qubit VQC.
        self.quantum = QuantumModel5Qubits()
        # Classifier: Map quantum outputs (5 expectation values) to 10 logits.
        self.classifier = nn.Linear(5, 10)

    def forward(self, x, theta_override=None):
        # Flatten input images.
        x = x.view(x.size(0), -1)
        theta = self.encoder(x)  # Shape: (batch, 10)
        quantum_outs = []
        for i in range(theta.shape[0]):
            # Use theta_override if provided (for robust training); otherwise, use theta.
            param_vec = theta[i] if theta_override is None else theta_override[i]
            quantum_out = self.quantum(param_vec)  # Output shape: (5,)
            quantum_outs.append(quantum_out)
        quantum_outs = torch.stack(quantum_outs, dim=0)  # Shape: (batch, 5)
        logits = self.classifier(quantum_outs)  # Shape: (batch, 10)
        return logits, theta


# -------------------------------
# Robust training routine on the quantum parameters (theta)
# -------------------------------
def robust_loss(
    model, images, labels, criterion, epsilon=0.1, num_steps=5, step_size=0.01
):
    """
    For a given batch of images and labels, this function computes the robust loss.
    An adversarial perturbation delta is applied to the quantum parameters (theta) produced by the encoder.
    An inner loop performs projected gradient ascent on delta (within an L∞ ball) to maximize the loss.
    """
    logits, theta = model(images)
    loss = criterion(logits, labels)

    delta = torch.zeros_like(theta, requires_grad=True)

    for _ in range(num_steps):
        theta_perturbed = theta + delta
        batch_logits = []
        for i in range(theta_perturbed.shape[0]):
            quantum_out = model.quantum(theta_perturbed[i])
            batch_logits.append(quantum_out)
        quantum_outs = torch.stack(batch_logits, dim=0)
        robust_logits = model.classifier(quantum_outs)
        robust_loss_val = criterion(robust_logits, labels)
        robust_loss_val.backward()

        with torch.no_grad():
            delta.add_(step_size * delta.grad.sign())
            delta.clamp_(-epsilon, epsilon)
        model.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()

    theta_perturbed = theta + delta
    batch_logits = []
    for i in range(theta_perturbed.shape[0]):
        quantum_out = model.quantum(theta_perturbed[i])
        batch_logits.append(quantum_out)
    quantum_outs = torch.stack(batch_logits, dim=0)
    robust_logits = model.classifier(quantum_outs)
    final_loss = criterion(robust_logits, labels)
    return final_loss


# -------------------------------
# Training loop using the MNIST dataset
# -------------------------------
def train_mnist(batch_size=64, epochs=5, lr=0.01):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    subset_train_dataset = Subset(train_dataset, list(range(1000)))
    train_loader = DataLoader(
        dataset=subset_train_dataset, batch_size=batch_size, shuffle=True
    )

    model = HybridQuantumClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("Begin Training....")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = robust_loss(
                model,
                images,
                labels,
                criterion,
                epsilon=0.1,
                num_steps=3,
                step_size=0.01,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )
        print(
            f"Epoch [{epoch + 1}/{epochs}] Average Loss: {total_loss / len(train_loader):.4f}"
        )

    return model


if __name__ == "__main__":
    trained_model = train_mnist()
