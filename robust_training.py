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
from tqdm import tqdm

# Set device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed = 1
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
    def __init__(self, num_qubits, num_features, num_labels, width):
        super(HybridQuantumModel, self).__init__()
        self.num_qubits = num_qubits
        self.width = width 

        # Classical Neural Network Encoding: map raw features to a vector of length num_qubits.
        self.encoding_net = nn.Sequential(
            nn.Linear(num_features, width),
            nn.Linear(width, num_qubits),
            nn.ReLU(),
        )

        self.qc, self.input_params, self.weight_params = create_quantum_circuit(num_qubits)

        # Define the observables for the quantum neural network.
        observables = [SparsePauliOp.from_list([("I" * i + "Z" + "I" * (num_qubits - i - 1), 1)]) for i in range(num_qubits)]
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
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        encoded_input = self.encoding_net(x)
        quantum_output = self.quantum_layer(encoded_input)
        output = self.classical_output_net(quantum_output)
        return output


class RobustQuantumTrainer:
    def __init__(self, model, X_test, y_test, learning_rate=0.01, lambda_reg=0.1, loss_metric="l2"):
        self.model = model
        self.lambda_reg = lambda_reg
        self.X_test = X_test
        self.y_test = y_test
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=lambda_reg, amsgrad=True
        )
        self.criterion = self._get_loss_function(loss_metric)

    def _get_loss_function(self, loss_metric):
        if loss_metric == "l1":
            return nn.L1Loss()
        elif loss_metric == "l2":
            return nn.MSELoss()
        elif loss_metric == "linf":
            return lambda y_pred, y_true: torch.max(torch.abs(y_pred - y_true))
        else:
            raise ValueError("Invalid loss metric. Choose 'l1', 'l2', or 'linf'.")

    def train(self, X_train, y_train, epochs=100, batch_size=16):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 3).to(device)
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )

        lipschitz_values = []
        test_accuracies = []  # Track test accuracy over epochs

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()

                output = self.model(data)
                loss = self.criterion(output, target)

                # Regularization term based on the Lipschitz **bound**
                reg_loss = sum(torch.sum(param**2) for param in self.model.parameters())
                print(
                    "True Loss: ",
                    loss.item(),
                    ",\t Regularization Penalty: ",
                    self.lambda_reg * reg_loss.item(),
                )

                loss.backward()
                self.optimizer.step()

            # Compute Lipschitz constant for the current model state
            from QlipSDP import HybridModelExtractor, compute_network_lipschitz
            extractor = HybridModelExtractor(self.model)
            extracted_layers = extractor.extract()
            overall_K = compute_network_lipschitz(extracted_layers)
            lipschitz_values.append(overall_K)

            # Evaluate test accuracy at the end of each epoch
            accuracy = self.evaluate_test_accuracy()
            test_accuracies.append(accuracy)
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Test Accuracy: {accuracy:.2f}%")

        return lipschitz_values, test_accuracies

    def evaluate_test_accuracy(self):
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.long).to(device)

        with torch.no_grad():
            output = self.model(X_test_tensor)
            _, predicted = torch.max(output, 1)
            _, y_label = torch.max(y_test_tensor, 1)
            accuracy = (predicted == y_label).float().mean().item()
        return accuracy

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
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(X)
        _, predicted = torch.max(output, 1)
        _, y_label = torch.max(y, 1)
        accuracy = (predicted == y_label).float().mean()
        print(f"Accuracy: {accuracy.item() * 100:.2f}%")


def train_and_save_model(
    # Hyperparameters
    X_train,
    X_test,
    num_features,
    y_train,
    y_test,
    num_labels,
    num_qubits=4,
    width = 4,
    learning_rate=0.01,
    lambda_reg=0.001,
    epochs=30,
    batch_size=16,
    loss_metric="l2",
    save_path="./model/model.pt"
):
    # Initialize model
    model = HybridQuantumModel(
        num_qubits=num_qubits,
        num_features=num_features,
        num_labels=num_labels,
        width=width
    ).to(device)

    # Train and evaluate the model
    trainer = RobustQuantumTrainer(
        model, X_test, y_test, learning_rate=learning_rate, lambda_reg=lambda_reg, loss_metric=loss_metric
    )  # lambda_reg := lipsdp

    lipschitz_values, accuracies  = trainer.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    print("Learning Rate:", learning_rate, ",\t Regularization Constant:", lambda_reg)
    print("#Epochs:", epochs, ",\t |Batch|:", batch_size)

    torch.save(model.state_dict(), save_path)
    evaluate(model, X_test, y_test)
    return lipschitz_values, accuracies

def train_naive(X_train, X_test, y_train, y_test, num_features, num_labels, num_qubits=4, width=4, learning_rate=0.01, epochs=30, batch_size=16):
    # Train without regularization or noise
    model = HybridQuantumModel(
        num_qubits=num_qubits,
        num_features=num_features,
        num_labels=num_labels,
        width=width
    ).to(device)
    trainer = RobustQuantumTrainer(model, X_test, y_test, learning_rate=learning_rate, lambda_reg=0.0)  # No regularization
    lipschitz_values, accuracies = trainer.train(X_train, y_train, epochs=epochs, batch_size=batch_size) 
    return lipschitz_values, accuracies

def train_pdg(X_train, X_test, y_train, y_test, num_features, num_labels, num_qubits=4, width=4, learning_rate=0.01, epochs=30, batch_size=16, noise_level=0.1):
    # Add noise to the data to increase dataset size (data augmentation)
    X_train_noisy = X_train + noise_level * np.random.randn(*X_train.shape)
    X_train_augmented = np.vstack([X_train, X_train_noisy])
    y_train_augmented = np.vstack([y_train, y_train])

    # Train with augmented data
    model = HybridQuantumModel(
        num_qubits=num_qubits,
        num_features=num_features,
        num_labels=num_labels,
        width=width
    ).to(device)
    trainer = RobustQuantumTrainer(model, X_test, y_test, learning_rate=learning_rate, lambda_reg=0.0)  # No regularization
    lipschitz_values, accuracies = trainer.train(X_train_augmented, y_train_augmented, epochs=epochs, batch_size=batch_size)
    return lipschitz_values, accuracies

def train_and_compare_models(X_train, X_test, y_train, y_test, num_features, num_labels):
    metrics = ["l1", "l2", "linf"]
    lipschitz_values = {}
    for metric in tqdm(metrics):
        print(f"Training model with {metric} loss...")
        lipschitz_values[metric] = train_and_save_model(
            X_train, X_test, num_features, y_train, y_test, num_labels, loss_metric=metric
        )
    plot_lipschitz_values(lipschitz_values["l1"], lipschitz_values["l2"], lipschitz_values["linf"], num_epochs=30)

def train_with_regularization(X_train, X_test, y_train, y_test, num_features, num_labels):
    lambda_reg_values = np.linspace(0.0001, 0.5, 20)
    lipschitz_values = {"l1": [], "l2": [], "linf": []}
    for lambda_reg in tqdm(lambda_reg_values):
        print(f"Training with λ = {lambda_reg}")
        for metric in lipschitz_values.keys():
            l_values = train_and_save_model(
            X_train, X_test, num_features, y_train, y_test, num_labels, loss_metric=metric, lambda_reg=lambda_reg
            )
            lipschitz_values[metric].append(l_values[-1])
    plot_lipschitz_vs_regularization(lipschitz_values["l1"], lipschitz_values["l2"], lipschitz_values["linf"], lambda_reg_values)

def train_with_varying_qubits(X_train, X_test, y_train, y_test, num_features, num_labels):
    num_qubits_list = range(1, 7)  # Number of qubits from 2 to 20 in steps of 2
    metrics = ["l1", "l2", "linf"]
    lipschitz_values = {metric: [] for metric in metrics}

    for num_qubits in tqdm(num_qubits_list):
        print(f"Training model with {num_qubits} qubits...")
        for metric in metrics:
            l_values = train_and_save_model(
                X_train, X_test, num_features, y_train, y_test, num_labels,
                num_qubits=num_qubits, loss_metric=metric
            )
            lipschitz_values[metric].append(l_values[-1]) 

    # Plot the results
    plot_lipschitz_vs_qubits(
        lipschitz_values["l1"], lipschitz_values["l2"], lipschitz_values["linf"], num_qubits_list
    )

def train_with_varying_width(X_train, X_test, y_train, y_test, num_features, num_labels):
    widths = range(5, 56, 5)
    metrics = ["l1", "l2", "linf"]
    lipschitz_values = {metric: [] for metric in metrics}

    for width in tqdm(widths):
        print(f"Training model with width = {width}...")
        for metric in metrics:
            l_values = train_and_save_model(
                X_train, X_test, num_features, y_train, y_test, num_labels,
                width=width, loss_metric=metric
            )
            lipschitz_values[metric].append(l_values[-1])  # Use the final Lipschitz value

    # Plot the results
    plot_lipschitz_vs_width(
        lipschitz_values["l1"], lipschitz_values["l2"], lipschitz_values["linf"], widths
    )

# Train and compare the models with the desired functionality
def train_and_plot_comparison(X_train, X_test, y_train, y_test, num_features, num_labels):
    print("Training naive model...")
    lipschitz_naive, accuracy_naive = train_naive(X_train, X_test, y_train, y_test, num_features, num_labels)
    print("Training model with regularization...")
    lipschitz_reg, accuracy_reg = train_and_save_model(X_train, X_test, num_features, y_train, y_test, num_labels)
    print("Training model with PDG...")
    lipschitz_pdg, accuracy_pdg = train_pdg(X_train, X_test, y_train, y_test, num_features, num_labels)

    # Plotting Lipschitz constant vs. epochs
    plot_lipschitz_accuracy_vs_epochs(lipschitz_naive, lipschitz_reg, lipschitz_pdg, accuracy_naive, accuracy_reg, accuracy_pdg)


# Plotting function
def plot_lipschitz_values(lipschitz_values_l1, lipschitz_values_l2, lipschitz_values_linf, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))

    # Plot Lipschitz values for all three models
    plt.plot(epochs, lipschitz_values_l1, label=r"$\ell_1$ Loss", marker="o")
    plt.plot(epochs, lipschitz_values_l2, label=r"$\ell_2$ Loss", marker="x")
    plt.plot(epochs, lipschitz_values_linf, label=r"$\ell_\infty$ Loss", marker="s")

    plt.xlabel("Training Epochs")
    plt.ylabel("Lipschitz Constant")
    plt.title(f"Lipschitz Constant Over Training Epochs for Different Loss Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_lipschitz_vs_regularization(lipschitz_values_l1, lipschitz_values_l2, lipschitz_values_linf, lambda_reg_values):
    plt.figure(figsize=(10, 6))

    # Plot Lipschitz values for all three loss metrics
    plt.plot(lambda_reg_values, lipschitz_values_l1, label=r"$\ell_1$ Loss", marker="o")
    plt.plot(lambda_reg_values, lipschitz_values_l2, label=r"$\ell_2$ Loss", marker="x")
    plt.plot(lambda_reg_values, lipschitz_values_linf, label=r"$\ell_\infty$ Loss", marker="s")

    plt.xlabel("Regularization Parameter (λ)")
    plt.ylabel("Lipschitz Constant")
    plt.title(f"Lipschitz Constant vs. Regularization Parameter for Different Loss Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_lipschitz_vs_qubits(lipschitz_values_l1, lipschitz_values_l2, lipschitz_values_linf, num_qubits_list):
    plt.figure(figsize=(10, 6))
    plt.plot(num_qubits_list, lipschitz_values_l1, label=r"$\ell_1$ Loss", marker="o")
    plt.plot(num_qubits_list, lipschitz_values_l2, label=r"$\ell_2$ Loss", marker="x")
    plt.plot(num_qubits_list, lipschitz_values_linf, label=r"$\ell_\infty$ Loss", marker="s")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Lipschitz Constant")
    plt.title("Lipschitz Constant vs. Number of Qubits for Different Loss Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_lipschitz_vs_width(lipschitz_values_l1, lipschitz_values_l2, lipschitz_values_linf, widths):
    plt.figure(figsize=(10, 6))
    plt.plot(widths, lipschitz_values_l1, label=r"$\ell_1$ Loss", marker="o")
    plt.plot(widths, lipschitz_values_l2, label=r"$\ell_2$ Loss", marker="x")
    plt.plot(widths, lipschitz_values_linf, label=r"$\ell_\infty$ Loss", marker="s")
    plt.xlabel("Number of Parameters in Classical Encoding Network")
    plt.ylabel("Lipschitz Constant")
    plt.title("Lipschitz Constant vs. Number of Parameters in Classical Encoding Network for Different Loss Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_lipschitz_accuracy_vs_epochs(lipschitz_naive, lipschitz_reg, lipschitz_pdg, accuracy_naive, accuracy_reg, accuracy_pdg):
    epochs = range(1, len(lipschitz_naive) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 12))
    
    # Top plot: Lipschitz Constant vs Epochs
    ax1.plot(epochs, lipschitz_naive, label="Naive", marker="o")
    ax1.plot(epochs, lipschitz_reg, label="Regularization", marker="x")
    ax1.plot(epochs, lipschitz_pdg, label="PDG", marker="s")
    ax1.set_ylabel("Lipschitz Constant")
    ax1.set_title("Lipschitz Constant vs Epochs")
    ax1.legend()
    ax1.grid(True)
    
    # Bottom plot: Test Accuracy vs Epochs
    ax2.plot(epochs, accuracy_naive, label="Naive", marker="o")
    ax2.plot(epochs, accuracy_reg, label="Regularization", marker="x")
    ax2.plot(epochs, accuracy_pdg, label="PDG", marker="s")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Test Accuracy vs Epochs")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()



    
if __name__ == "__main__":
    iris = sklearn_datasets.load_iris()
    X_train, X_test, num_features, y_train, y_test, num_labels = prepare_data(iris.data, iris.target)
    # Train and save a single model
    # model, _ = train_and_save_model(X_train, X_test, num_features, y_train, y_test, num_labels, loss_metric=metric)

    # Train and compare three models with different loss metrics
    # train_and_compare_models(X_train, X_test, y_train, y_test, num_features, num_labels)

    # Train with different regularization parameters
    # train_with_regularization(X_train, X_test, y_train, y_test, num_features, num_labels)

    # Train and plot Lipschitz vs. Number of Qubitss for different loss metrics
    # train_with_varying_qubits(X_train, X_test, y_train, y_test, num_features, num_labels)

    # Train and plot Lipschitz vs. Width of Classical Encoding Network for different loss metrics
    # train_with_varying_width(X_train, X_test, y_train, y_test, num_features, num_labels)

    # Train and compare naive, regularized, and PDG models
    train_and_plot_comparison(X_train, X_test, y_train, y_test, num_features, num_labels)