import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pennylane as qml
import numpy as np
from evaluator import FairnessEvaluator
from pennylane.optimize import AdamOptimizer

# Define your quantum circuit
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="autograd")
def custom_circuit(params, x):
    qml.AngleEmbedding(x[:n_qubits], wires=range(n_qubits))
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))


# Instantiate evaluator
evaluator = FairnessEvaluator(circuit_function=custom_circuit, n_qubits=n_qubits)

# Load your data (dummy example)
data = np.random.rand(100, 4)
labels = np.random.choice([0, 1], size=100)
sensitive_attribute = np.random.choice([0, 1], size=100)
params = np.random.uniform(size=(2, n_qubits, 3))

# Train the model
trained_params = evaluator.train(data, labels, params, AdamOptimizer(stepsize=0.01), epochs=50)

# Evaluate fairness
results = evaluator.evaluate(trained_params, data, labels, sensitive_attribute)
print("Fairness Results:", results)
