import numpy as np
import pennylane as qml
import functools


def get_unitary_matrix(params, n_qubits):
    """
    Compute the unitary matrix of a trained quantum circuit.

    Args:
        params (np.ndarray): Circuit parameters.
        n_qubits (int): Number of qubits.

    Returns:
        np.ndarray: Unitary matrix representation of the circuit.
    """
    U = np.eye(2**n_qubits, dtype=complex)
    for layer in params:
        for i in range(n_qubits):
            angle = layer[i][0]
            rx = qml.RX(angle, wires=i).matrix()
            expanded_rx = functools.reduce(np.kron, [rx if j == i else np.eye(2) for j in range(n_qubits)])
            U = expanded_rx @ U

        for i in range(n_qubits):
            entangle_angle = layer[i][1]
            ry = qml.RY(entangle_angle, wires=i).matrix()
            expanded_ry = functools.reduce(np.kron, [ry if j == i else np.eye(2) for j in range(n_qubits)])
            U = expanded_ry @ U

    return U
