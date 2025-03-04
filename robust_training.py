import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.visualization import circuit_drawer
from qiskit.quantum_info import Operator


from sklearn import datasets as sklearn_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed = 23423
set_seed(seed)

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
    def __init__(self, num_qubits, num_features, num_labels):
        super(HybridQuantumModel, self).__init__()

        # Classical Neural Network Encoding: map raw features to a vector of length num_qubits.
        self.encoding_net = nn.Sequential(
            nn.Linear(num_features, num_qubits),
            # nn.ReLU(),
            # nn.Linear(2 * num_qubits, num_qubits),
            nn.ReLU(),
        )

        self.qc, self.input_params, self.weight_params = create_quantum_circuit(
            num_qubits
        )
        # Build a list of #num_qubits observables.
        observables = []
        for i in range(num_qubits):
            op_str = "I" * i + "Z" + "I" * (num_qubits - i - 1)
            observables.append(SparsePauliOp.from_list([(op_str, 1)]))
        # Create an EstimatorV2 for statevector simulation.
        estimator = EstimatorV2(options={"run_options": {"method": "statevector"}})

        # Create the quantum neural network using EstimatorQNN.
        qnn = EstimatorQNN(
            circuit=self.qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            observables=observables,
            estimator=estimator,
        )
        # Wrap the QNN as a PyTorch layer.
        self.quantum_layer = TorchConnector(qnn)

        # Classical Neural Network after the quantum network.
        self.classical_output_net = nn.Sequential(
            nn.Linear(num_qubits, num_labels),
            # nn.ReLU(),
            # nn.Linear(num_qubits, num_labels),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        encoded_input = self.encoding_net(x)
        quantum_output = self.quantum_layer(encoded_input)
        output = self.classical_output_net(quantum_output)
        # output = self.classical_output_net(encoded_input)
        return output


class RobustQuantumTrainer:
    def __init__(self, model, learning_rate=0.01, lambda_reg=0.1, loss_metric="l2"):
        self.model = model
        self.lambda_reg = lambda_reg
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=lambda_reg, amsgrad=True
        )

        # Define the loss function based on the metric
        if loss_metric == "l1":
            self.criterion = nn.L1Loss()
        elif loss_metric == "l2":
            self.criterion = nn.MSELoss()
        elif loss_metric == "linf":
            self.criterion = lambda y_pred, y_true: torch.max(torch.abs(y_pred - y_true))
        else:
            raise ValueError("Invalid loss metric. Choose 'l1', 'l2', or 'linf'.")

    def train(self, X_train, y_train, epochs=100, batch_size=16):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 3)
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )

        # Track loss values
        lipschitz_values = []

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()

                output = self.model(data)
                loss = self.criterion(output, target)

                # -----------------------------------------------------------
                # Regularization term based on the Lipschitz **bound**
                reg_loss = 0
                for param in self.model.parameters():
                    # for param in self.model.encoding_net.parameters():
                    reg_loss += torch.sum(param**2)
                print(
                    "True Loss: ",
                    loss.item(),
                    ",\t Regularization Penalty: ",
                    self.lambda_reg * reg_loss.item(),
                )
                # loss += self.lambda_reg * reg_loss
                # -----------------------------------------------------------

                loss.backward()
                self.optimizer.step()


            from QlipSDP import HybridModelExtractor, compute_network_lipschitz
            # Compute Lipschitz constant for the current model state
            extractor = HybridModelExtractor(self.model)
            extracted_layers = extractor.extract()
            overall_K = compute_network_lipschitz(extracted_layers)
            lipschitz_values.append(overall_K)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        return lipschitz_values

# Plotting function
def plot_lipschitz_values(lipschitz_values_l1, lipschitz_values_l2, lipschitz_values_linf, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))

    # Plot Lipschitz values for all three models
    plt.plot(epochs, lipschitz_values_l1, label="ℓ1 Loss", marker="o")
    plt.plot(epochs, lipschitz_values_l2, label="ℓ2 Loss", marker="x")
    plt.plot(epochs, lipschitz_values_linf, label="ℓ∞ Loss", marker="s")

    plt.xlabel("Training Epochs")
    plt.ylabel("Lipschitz Constant")
    plt.title(f"Lipschitz Constant Over Training Epochs for Different Loss Metrics with seed {seed}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Data Preparation
def prepare_data(X, y):
    num_classes = max(y) - min(y) + 1
    num_features = len(X[0])
    y = np.eye(num_classes)[y]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    return X_train, X_test, num_features, y_train, y_test, num_classes


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


def hybrid_model(
    # Hyperparameters
    X_train,
    X_test,
    num_features,
    y_train,
    y_test,
    num_labels,
    num_qubits=4,
    learning_rate=0.01,
    lambda_reg=0.001,
    epochs=15,
    batch_size=16,
    loss_metric="l2",
):
    # Initialize model
    model = HybridQuantumModel(
        num_qubits=num_qubits,
        num_features=num_features,
        num_labels=num_labels,
    )

    # Train and evaluate the model
    trainer = RobustQuantumTrainer(
        model, learning_rate=learning_rate, lambda_reg=lambda_reg, loss_metric=loss_metric
    )  # lambda_reg := lipsdp

    lipschitz_values = trainer.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    print("Learning Rate:", learning_rate, ",\t Regularization Constant:", lambda_reg)
    print("#Epochs:", epochs, ",\t |Batch|:", batch_size)
    evaluate(model, X_test, y_test)
    return model, lipschitz_values


def train_iris(iris=sklearn_datasets.load_iris()):
    # Prepare data
    X = iris.data
    y = iris.target
    X_train, X_test, num_features, y_train, y_test, num_labels = prepare_data(X, y)
    num_epochs = 30

    # Train three models with different loss metrics
    model_, lipschitz_values_l1 = hybrid_model(
        X_train, X_test, num_features, y_train, y_test, num_labels, loss_metric="l1", epochs=num_epochs
    )
    model, lipschitz_values_l2 = hybrid_model(
        X_train, X_test, num_features, y_train, y_test, num_labels, loss_metric="l2", epochs=num_epochs
    )
    model_, lipschitz_values_linf = hybrid_model(
        X_train, X_test, num_features, y_train, y_test, num_labels, loss_metric="linf", epochs=num_epochs
    )

    # Plot the Lipschitz values for all three models
    plot_lipschitz_values(lipschitz_values_l1, lipschitz_values_l2, lipschitz_values_linf, num_epochs=num_epochs)

    torch.save(model.state_dict(), "./model/iris.pt")
    return model


def iris_hybrid_model(num_qubits=4, iris=sklearn_datasets.load_iris()):
    X = iris.data
    y = iris.target
    X_train, X_test, num_features, y_train, y_test, num_labels = prepare_data(X, y)
    model = HybridQuantumModel(
        num_qubits=num_qubits,
        num_features=num_features,
        num_labels=num_labels,
    )
    model.load_state_dict(torch.load("./model/iris.pt", weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    model = train_iris()
    # model, loss_values = iris_hybrid_model()
    # print(model)
    layers = []
    if hasattr(model, "layer_list"):
        modules = model.layer_list
    elif isinstance(model, nn.Sequential):
        modules = list(model)
    else:
        modules = list(model.children())

    i = 0
    while i < len(modules):
        mod = modules[i]
        i += 1
        if isinstance(mod, TorchConnector):
            params = {}
            for name, param in mod.named_parameters():
                params[name] = param.detach().cpu().numpy()
            qc = mod.neural_network.circuit
            trained_values = params["weight"]
            input_values = [0.0] * len(model.input_params)

            # Create a dictionary mapping each parameter to its value.
            param_dict = {p: v for p, v in zip(model.input_params, input_values)}
            param_dict.update(
                {p: v for p, v in zip(model.weight_params, trained_values)}
            )
            bound_circuit = qc.assign_parameters(param_dict)
            unitary_matrix = Operator(bound_circuit).data
            # print(unitary_matrix)
            print(
                mod,
                "\n",
                type(mod),
                "\n",
                mod.named_parameters(),
                "\n",
                unitary_matrix,
                "\n-----------",
            )
            print(params)
