import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os, sys
import seaborn as sns

#! Change to 'cuda' if you are using Nvidia, change to mps if you are using Mac, otherwise 'cpu'
device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

# Constants
m = 1.0  # Mass (kg)
b = 10  # Friction coefficient
k_p = 50  # Proportional gain
k_d = 10   # Derivative gain
dt = 0.01  # Time step
num_samples = 1250  # Number of samples in dataset

# Generate synthetic data for trajectory tracking
t = np.linspace(0, 10, num_samples)
q_target = np.sin(t)
dot_q_target = np.cos(t)

# Initial conditions for training data generation
q = 0
dot_q = 0
X = []
Y = []

for i in range(num_samples):
    # PD control output
    tau = k_p * (q_target[i] - q) + k_d * (dot_q_target[i] - dot_q)
    # Ideal motor dynamics (variable mass for realism)
    #m_real = m * (1 + 0.1 * np.random.randn())  # Mass varies by +/-10%
    ddot_q_real = (tau - b * dot_q) / m
    
    # Calculate error
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    
    # Store data
    X.append([q, dot_q, q_target[i], dot_q_target[i]])
    Y.append(ddot_q_error)
    
    # Update state
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Convert data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# Define a deep MLP model, 2 hidden layers
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Define the Deeper CorrectorMLP model, 3 hidden layers
class DeeperMLP(nn.Module):
    def __init__(self, hidden_size):
        super(DeeperMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.layers(x)
        

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_heatmap(data, x_labels, y_labels, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap='coolwarm', xticklabels=x_labels, yticklabels=y_labels, fmt=".5f")
    plt.title(title)
    plt.xlabel('Learning Rates')
    plt.ylabel('Hidden Layer Sizes')
    plt.savefig(filename)
    plt.show()

def main(model_type='deep'):
    device = torch.device("cpu")
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
    hidden_sizes = [32, 64, 96, 128]
    epochs = 500
    results = {}

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    for lr in learning_rates:
        results[lr] = {}
        for hidden_size in hidden_sizes:
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            if model_type == 'deep':
                model = MLP(hidden_size=hidden_size).to(device)
            else:
                model = DeeperMLP(hidden_size=hidden_size).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_losses = []
            test_losses = []
            for epoch in range(epochs):
                epoch_loss = 0
                model.train()
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_train_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = criterion(output, target)
                        test_loss += loss.item()
                avg_test_loss = test_loss / len(test_loader)
                test_losses.append(avg_test_loss)

                print(f'Epoch {epoch + 1}/{epochs}, Hidden Size {hidden_size}, LR {lr}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')

            results[lr][hidden_size] = {'train_losses': train_losses, 'test_losses': test_losses}

            # Trajectory Tracking Simulation
            q_test = 0.0
            dot_q_test = 0.0
            q_real = []
            q_real_corrected = []

            # Integration with only PD Control
            for k in range(len(t)):
                tau = k_p * (q_target[k] - q_test) + k_d * (dot_q_target[k] - dot_q_test)
                ddot_q_real = (tau - b * dot_q_test) / m
                dot_q_test += ddot_q_real * dt
                q_test += dot_q_test * dt
                q_real.append(q_test)

            # Reset state variables for corrected simulation
            q_test = 0.0
            dot_q_test = 0.0

            # Integration with PD Control and MLP Correction
            for j in range(len(t)):
                tau = k_p * (q_target[j] - q_test) + k_d * (dot_q_target[j] - dot_q_test)
                inputs = torch.tensor([q_test, dot_q_test, q_target[j], dot_q_target[j]], dtype=torch.float32).to(device)
                correction = model(inputs.unsqueeze(0)).item()
                ddot_q_corrected = (tau - b * dot_q_test + correction) / m
                dot_q_test += ddot_q_corrected * dt
                q_test += dot_q_test * dt
                q_real_corrected.append(q_test)

            # Plotting the trajectories
            ensure_dir(f'images/task1/{model_type}/lr{lr}')
            plt.figure(figsize=(10, 6))
            plt.plot(t, q_target, 'r-', label='Target')
            plt.plot(t, q_real, 'b--', label='PD Only')
            plt.plot(t, q_real_corrected, 'g--', label='PD + MLP Correction')
            plt.title(f'Trajectory Tracking with Hidden Size {hidden_size}, LR {lr}')
            plt.xlabel('Time [s]')
            plt.ylabel('Position')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'images/task1/{model_type}/lr{lr}/trajectory_hidden{hidden_size}.png')
            plt.close()
        
        # Plot training and test losses for this learning rate
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        for hidden_size in hidden_sizes:
            plt.plot(results[lr][hidden_size]['train_losses'], label=f'Hidden Size {hidden_size}')
        plt.title(f'Training Loss for LR {lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for hidden_size in hidden_sizes:
            plt.plot(results[lr][hidden_size]['test_losses'], label=f'Hidden Size {hidden_size}')
        plt.title(f'Test Loss for LR {lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        ensure_dir(f'images/task1/{model_type}/lr{lr}')
        plt.tight_layout()
        plt.savefig(f'images/task1/{model_type}/lr{lr}/training_test_loss.png')
        plt.close()

    # Generate heatmap for final test losses
    heatmap_data = np.array([[results[lr][hs]['test_losses'][-1] for lr in learning_rates] for hs in hidden_sizes])
    generate_heatmap(heatmap_data, y_labels=hidden_sizes, x_labels=learning_rates, title="Final Test Loss Heatmap", filename=f"images/task1/{model_type}/final_test_loss_heatmap.png")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'deep':
            main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'deeper':
        main('deeper')
    else:
        main()