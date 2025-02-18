import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, Aer


from sklearn import datasets as sklearn_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


# Hybrid Quantum-Classical Model
class HybridQuantumModel(nn.Module):
    def __init__(self, num_qubits, num_features, num_labels, encoding_size=2):
        super(HybridQuantumModel, self).__init__()

        # Classical Neural Network Encoding
        self.encoding_net = nn.Sequential(
            nn.Linear(num_features, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU(),
        )

        # Quantum Circuit
        self.num_qubits = num_qubits
        self.quantum_circuit = QuantumCircuit(num_qubits)

        # Classical net after quantum model
        self.classical_output_net = nn.Sequential(
            nn.Linear(2**num_qubits, num_qubits),
            nn.ReLU(),
            nn.Linear(num_qubits, num_labels),
            nn.Sigmoid(),
            nn.Linear(num_labels, 1),
        )
        # Classical layer after quantum model
        # self.classical_output_layer = nn.Linear(2**num_qubits, 1)

    def forward(self, x):
        encoded_input = self.encoding_net(x)
        for qubit in range(self.num_qubits):
            self.quantum_circuit.rx(encoded_input[0][0].item(), qubit)

        # ---------------------------------------------------------------------------
        # Simulate the quantum circuit using Aer simulator
        self.quantum_circuit.save_statevector()
        aer_simulator = AerSimulator(method="statevector")
        compiled_circuit = transpile(self.quantum_circuit, aer_simulator)
        result = aer_simulator.run(compiled_circuit).result()
        # statevector = result.get_statevector(compiled_circuit)
        statevector = result.data(0)["statevector"]
        # counts = result.get_counts(compiled_circuit)
        # print(statevector)
        # ---------------------------------------------------------------------------

        # Classical part to process quantum statevector
        counts_tensor = torch.tensor(statevector, dtype=torch.float)
        output = self.classical_output_net(counts_tensor)

        return output


class RobustQuantumTrainer:
    def __init__(self, model, learning_rate=0.01, lambda_reg=0.1):
        self.model = model
        self.lambda_reg = lambda_reg
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        # criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multilabel classification

    def train(self, X_train, y_train, epochs=100, batch_size=16):
        # Convert dataset to torch tensors & Create DataLoader
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
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
                # Regularization term based on the Lipschitz bound
                reg_loss = 0
                for param in self.model.parameters():
                    reg_loss += torch.sum(param**2)
                loss += self.lambda_reg * reg_loss

                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


# Data Preparation
def prepare_data(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def evaluate(model, X, y):
    with torch.no_grad():
        output = model(X)
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == y).float().mean()
        print(f"Accuracy: {accuracy.item() * 100:.2f}%")


if __name__ == "__main__":
    # Hyperparameters
    num_qubits = 2

    # Prepare data
    iris = sklearn_datasets.load_iris()
    X = iris.data
    y = iris.target
    num_labels = max(y)
    num_features = len(X[0])
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Initialize model
    model = HybridQuantumModel(
        num_qubits=num_qubits,
        num_features=num_features,
        num_labels=num_labels,
        encoding_size=2,
    )

    # Train and evaluate the model
    trainer = RobustQuantumTrainer(model, learning_rate=0.01, lambda_reg=0.1)
    trainer.train(X_train, y_train, epochs=100, batch_size=16)

    evaluate(model, X, y)
