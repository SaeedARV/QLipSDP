import numpy as np
from scipy.optimize import minimize
import time
import functools
from utils import get_unitary_matrix


class FairnessEvaluator:
    def __init__(self, circuit_function, n_qubits):
        """
        Initialize the fairness evaluator.

        Args:
            circuit_function (callable): User-defined variational circuit function.
                It must take parameters and data as input and return a prediction.
            n_qubits (int): Number of qubits in the circuit.
        """
        self.circuit_function = circuit_function
        self.n_qubits = n_qubits

    def train(self, train_data, train_labels, params, optimizer, epochs=10, batch_size=16):
        """
        Train the circuit on the provided data.

        Args:
            train_data (np.ndarray): Training data.
            train_labels (np.ndarray): Training labels.
            params (np.ndarray): Initial parameters for the circuit.
            optimizer (callable): Optimizer function (e.g., from Pennylane or Scipy).
            epochs (int): Number of training epochs.
            batch_size (int): Size of training batches.

        Returns:
            np.ndarray: Trained parameters.
        """
        for epoch in range(epochs):
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i : i + batch_size]
                batch_labels = train_labels[i : i + batch_size]
                for x, y in zip(batch_data, batch_labels):
                    params = optimizer.step(lambda p: self._cost(p, x, y), params)
        return params

    def _cost(self, params, x, y):
        """Calculate the cost function for training."""
        prediction = self.circuit_function(params, x)
        return (prediction - y) ** 2

    def evaluate(self, params, test_data, test_labels, sensitive_attribute):
        """
        Evaluate the circuit for fairness and accuracy.

        Args:
            params (np.ndarray): Trained circuit parameters.
            test_data (np.ndarray): Test dataset inputs.
            test_labels (np.ndarray): Test dataset labels.
            sensitive_attribute (np.ndarray): Sensitive attribute for fairness evaluation.

        Returns:
            dict: Evaluation results including accuracy, Lipschitz constant, and fairness metrics.
        """
        # Compute accuracy
        correct_predictions = sum(
            1 for x, y in zip(test_data, test_labels) if round(float(self.circuit_function(params, x))) == y
        )
        accuracy = correct_predictions / len(test_labels)

        # Compute Lipschitz constant
        lipschitz_constant, lip_time = self._compute_lipschitz_constant(params)

        return {
            "accuracy": accuracy,
            "lipschitz_constant": lipschitz_constant,
            "lipschitz_time": lip_time,
        }

    def _compute_lipschitz_constant(self, params):
        """
        Compute the Lipschitz constant for the trained circuit.

        Args:
            params (np.ndarray): Trained circuit parameters.

        Returns:
            tuple: Lipschitz constant and computation time.
        """
        U_matrix = get_unitary_matrix(params, self.n_qubits)
        dim = 2 ** self.n_qubits
        M = functools.reduce(np.kron, [np.diag([1, -1]) for _ in range(self.n_qubits)])

        def objective(x):
            rho_matrix = x[:dim**2].reshape((dim, dim))
            rho_matrix = rho_matrix @ rho_matrix.T.conj()
            rho_matrix /= np.trace(rho_matrix)

            sigma_matrix = x[dim**2:].reshape((dim, dim))
            sigma_matrix = sigma_matrix @ sigma_matrix.T.conj()
            sigma_matrix /= np.trace(sigma_matrix)

            trace_term = np.abs(np.trace(M @ U_matrix.conj() @ (rho_matrix - sigma_matrix) @ M.conj()))
            return -trace_term

        x0 = np.random.rand(2 * dim**2)
        start_time = time.time()
        result = minimize(objective, x0, method="L-BFGS-B")
        elapsed_time = time.time() - start_time
        K_star = 0.5 * -result.fun
        return K_star, elapsed_time
