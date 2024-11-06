import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os, sys

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

# Define the MLP model
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

# Define the DeepCorrectorMLP model
class DeepCorrectorMLP(nn.Module):
    def __init__(self, hidden_size_1, hidden_size_2):
        super(DeepCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 1)
        )
    def forward(self, x):
        return self.layers(x)
        

def main_1(lr=0.1):
    #! For this task, CPU is faster than GPU
    device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
    # Hidden layer sizes to evaluate
    hidden_sizes = [32, 64, 96, 128]
    # Dictionary to store results
    results = {}
    results_str = ''

    # Split data into training and test sets (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
    
    # Create datasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create subplots for trajectory plots
    _, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = axes.flatten()

    for i, hidden_size in enumerate(hidden_sizes):
        model = MLP(hidden_size=hidden_size).to(device)
        criterion = nn.MSELoss()
        #! Change learning rate below
        optimizer = optim.Adam(model.parameters(), lr=lr)
        epochs = 500
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            # Training loop
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
            
            # Evaluate on test set
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
            model.train()
            
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')

        # Store losses for plotting
        results[hidden_size] = {'train_losses': train_losses, 'test_losses': test_losses}
        results_str += f'Hidden Size: {hidden_size}, Final Train Loss: {train_losses[-1]:.6f}, Final Test Loss: {test_losses[-1]:.6f}\n'
        
        # Testing Phase: Simulate trajectory tracking
        q_test = 0
        dot_q_test = 0
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
        q_test = 0
        dot_q_test = 0
        
        # Integration with PD Control and MLP Correction
        for j in range(len(t)):
            tau = k_p * (q_target[j] - q_test) + k_d * (dot_q_target[j] - dot_q_test)
            inputs = torch.tensor([q_test, dot_q_test, q_target[j], dot_q_target[j]], dtype=torch.float32).to(device)
            correction = model(inputs.unsqueeze(0)).item()
            ddot_q_corrected = (tau - b * dot_q_test + correction) / m
            dot_q_test += ddot_q_corrected * dt
            q_test += dot_q_test * dt
            q_real_corrected.append(q_test)

        # Plot results
        axes[i].plot(t, q_target, 'r-', label='Target')
        axes[i].plot(t, q_real, 'b--', label='PD Only')
        axes[i].plot(t, q_real_corrected, 'g--', label='PD + MLP Correction')
        axes[i].set_title(f'Trajectory with Hidden Size {hidden_size}')
        axes[i].set_xlabel('Time [s]')
        axes[i].set_ylabel('Position')
        axes[i].legend()
    # Adjust layout and save trajectory plot
    plt.tight_layout()
    if not os.path.exists(f'images/task1/shallow/lr{lr}'):
        os.makedirs(f'images/task1/shallow/lr{lr}')
    #! Uncomment the following line to save the plot
    # plt.savefig(f'images/task1/shallow/lr{lr}/trajectories.png')
    
    # Log training results
    print("\033[92m=============================== Training/Testing loss ================================\033[0m")
    print(results_str.strip())
    print("\033[92m======================================================================================\033[0m")

    # Plot training and test losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for hidden_size in hidden_sizes:
        train_losses = results[hidden_size]['train_losses']
        plt.plot(train_losses, label=f'Train Loss Hidden Size {hidden_size}')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    for hidden_size in hidden_sizes:
        test_losses = results[hidden_size]['test_losses']
        plt.plot(test_losses, label=f'Test Loss Hidden Size {hidden_size}')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #! Uncomment the following line to save the plot
    # plt.savefig(f'images/task1/shallow/lr{lr}/training_test_loss.png')
    plt.show()

# Main function for Task 1.2 (Deep MLP with test loss logging and plotting)
def main_2(lr=0.1):
    device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
    # Hidden layer sizes to evaluate
    hidden_sizes = [16, 32, 64]
    results = {}
    results_str = ''

    # Split data into training and test sets (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
    
    # Create datasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create subplots for trajectory plots
    _, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    index = 0
    for hidden_size_1 in hidden_sizes:
        for hidden_size_2 in hidden_sizes:
            # Initialize the model
            model = DeepCorrectorMLP(hidden_size_1, hidden_size_2).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            epochs = 500
            train_losses = []
            test_losses = []
            for epoch in range(epochs):
                epoch_loss = 0
                # Training loop
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                # Compute average training loss
                avg_train_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # Evaluate on test set
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
                model.train()
                
                print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')

            # Store losses for plotting
            key = f'{hidden_size_1}_{hidden_size_2}'
            results[key] = {'train_losses': train_losses, 'test_losses': test_losses}
            results_str += f'Hidden Layers Size {hidden_size_1}, {hidden_size_2} Final Train Loss: {train_losses[-1]:.6f}, Final Test Loss: {test_losses[-1]:.6f}\n'
                
            # Testing Phase: Simulate trajectory tracking
            q_test = 0
            dot_q_test = 0
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
            q_test = 0
            dot_q_test = 0
            
            # Integration with PD Control and MLP Correction
            for j in range(len(t)):
                tau = k_p * (q_target[j] - q_test) + k_d * (dot_q_target[j] - dot_q_test)
                inputs = torch.tensor([q_test, dot_q_test, q_target[j], dot_q_target[j]], dtype=torch.float32).to(device)
                correction = model(inputs.unsqueeze(0)).item()
                ddot_q_corrected = (tau - b * dot_q_test + correction) / m
                dot_q_test += ddot_q_corrected * dt
                q_test += dot_q_test * dt
                q_real_corrected.append(q_test)

            # Plot results
            axes[index].plot(t, q_target, 'r-', label='Target')
            axes[index].plot(t, q_real, 'b--', label='PD Only')
            axes[index].plot(t, q_real_corrected, 'g--', label='PD + MLP Correction')
            axes[index].set_title(f'Trajectory with Hidden Sizes {hidden_size_1}, {hidden_size_2}')
            axes[index].set_xlabel('Time [s]')
            axes[index].set_ylabel('Position')
            axes[index].legend()
            index += 1
    # Adjust layout and save trajectory plot
    plt.tight_layout()
    if not os.path.exists(f'images/task1/deep/lr{lr}'):
        os.makedirs(f'images/task1/deep/lr{lr}')
    #! Uncomment the following line to save the plot
    # plt.savefig(f'images/task1/deep/lr{lr}/trajectories.png')
    
    # Log training results
    print("\033[92m=============================== Training/Testing loss ================================\033[0m")
    print(results_str.strip())
    print("\033[92m======================================================================================\033[0m")

    # Plot training and test losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for hidden_size_1 in hidden_sizes:
        for hidden_size_2 in hidden_sizes:
            key = f'{hidden_size_1}_{hidden_size_2}'
            train_losses = results[key]['train_losses']
            plt.plot(train_losses, label=f'Train Loss {hidden_size_1},{hidden_size_2}')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    for hidden_size_1 in hidden_sizes:
        for hidden_size_2 in hidden_sizes:
            key = f'{hidden_size_1}_{hidden_size_2}'
            test_losses = results[key]['test_losses']
            plt.plot(test_losses, label=f'Test Loss {hidden_size_1},{hidden_size_2}')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #! Uncomment the following line to save the plot
    # plt.savefig(f'images/task1/deep/lr{lr}/training_test_loss.png')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'shallow':
        if len(sys.argv) > 2:
            lr = float(sys.argv[2])
            main_1(lr)
        else:
            main_1()
    elif len(sys.argv) > 1 and sys.argv[1] == 'deep':
        if len(sys.argv) > 2:
            lr = float(sys.argv[2])
            main_2(lr)
        else:
            main_2()
    else:
        if len(sys.argv) > 1:
            lr = float(sys.argv[1])
            main_1(lr)
        else:
            main_1()