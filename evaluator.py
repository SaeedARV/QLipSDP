import functools
import numpy as np
from scipy.optimize import minimize
from utils import get_unitary_matrix  # Importing from utils

class Evaluator:
    def __init__(self, circuit_function, n_qubits):
        self.circuit_function = circuit_function
        self.n_qubits = n_qubits

    def evaluate(self, params):
        """
        Evaluate the trained circuit.

        Args:
            params (np.ndarray): Trained circuit parameters.

        Returns:
            dict: Evaluation results containing the Lipschitz constant and computation time.
        """
        lipschitz_constant = self._compute_lipschitz_constant(params)
        return lipschitz_constant

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
        result = minimize(objective, x0, method="L-BFGS-B")
        lipschitz_constant = 0.5 * -result.fun

        return lipschitz_constant
