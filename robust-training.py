import numpy as np
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

from sklearn import datasets as sklearn_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def create_quantum_circuit(num_qubits):
    qc = QuantumCircuit(num_qubits)
    input_params = ParameterVector("x", num_qubits)
    weight_params = ParameterVector("θ", num_qubits)

    # Data encoding: apply RX rotations using the classical input features.
    for i in range(num_qubits):
        qc.rx(input_params[i], i)

    # Variational layer: apply RY rotations with trainable parameters.
    for i in range(num_qubits):
        qc.ry(weight_params[i], i)

    return qc, list(input_params), list(weight_params)


class HybridQuantumModel(nn.Module):
    def __init__(self, num_qubits, num_features, num_labels, encoding_size=2):
        super(HybridQuantumModel, self).__init__()

        # Classical Neural Network Encoding: map raw features to a vector of length num_qubits.
        self.encoding_net = nn.Sequential(
            nn.Linear(num_features, 3 * num_qubits),
            nn.ReLU(),
            nn.Linear(3 * num_qubits, num_qubits),
            nn.ReLU(),
        )

        qc, input_params, weight_params = create_quantum_circuit(num_qubits)
        # Build a list of #num_qubits observables.
        observables = []
        for i in range(num_qubits):
            op_str = "I" * i + "Z" + "I" * (num_qubits - i - 1)
            observables.append(SparsePauliOp.from_list([(op_str, 1)]))
        # Create an EstimatorV2 for statevector simulation.
        estimator = EstimatorV2(options={"run_options": {"method": "statevector"}})

        # Create the quantum neural network using EstimatorQNN.
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            observables=observables,
            estimator=estimator,
        )
        # Wrap the QNN as a PyTorch layer.
        self.quantum_layer = TorchConnector(qnn)

        # Classical Neural Network after the quantum network.
        self.classical_output_net = nn.Sequential(
            nn.Linear(num_qubits, 2**num_qubits),
            nn.ReLU(),
            nn.Linear(2**num_qubits, num_labels),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        encoded_input = self.encoding_net(x)
        quantum_output = self.quantum_layer(encoded_input)
        output = self.classical_output_net(quantum_output)
        # output = self.classical_output_net(encoded_input)
        return output


class RobustQuantumTrainer:
    def __init__(self, model, learning_rate=0.01, lambda_reg=0.1):
        self.model = model
        self.lambda_reg = lambda_reg
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        # criterion = nn.BCELoss()

    def train(self, X_train, y_train, epochs=100, batch_size=16):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 3)
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()

                output = self.model(data)
                loss = self.criterion(output, target)

                # -----------------------------------------------------------
                # Regularization term based on the Lipschitz **bound**
                reg_loss = 0
                for param in self.model.parameters():
                    reg_loss += torch.sum(param**2)
                # loss += self.lambda_reg * reg_loss
                # -----------------------------------------------------------

                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


# Data Preparation
def prepare_data(X, y):
    num_classes = max(y) - min(y) + 1
    y = np.eye(num_classes)[y]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, num_classes


def evaluate(model, X, y):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)

    with torch.no_grad():
        output = model(X)
        _, predicted = torch.max(output, 1)
        _, y_label = torch.max(y, 1)
        accuracy = (predicted == y_label).float().mean()
        print(f"Accuracy: {accuracy.item() * 100:.2f}%")


if __name__ == "__main__":
    # Hyperparameters
    num_qubits = 5

    # Prepare data
    iris = sklearn_datasets.load_iris()
    X = iris.data
    y = iris.target
    num_features = len(X[0])
    X_train, X_test, y_train, y_test, num_labels = prepare_data(X, y)

    # Initialize model
    model = HybridQuantumModel(
        num_qubits=num_qubits,
        num_features=num_features,
        num_labels=num_labels,
        encoding_size=2,
    )

    # Train and evaluate the model
    trainer = RobustQuantumTrainer(
        model, learning_rate=0.01, lambda_reg=0.1
    )  # lambda_reg := lipsdp

    trainer.train(X_train, y_train, epochs=10, batch_size=16)

    evaluate(model, X_test, y_test)
