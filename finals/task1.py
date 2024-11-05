import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

#! Change to 'cuda' if you are using Nvidia, change to mps if you are using Mac, otherwise 'cpu'
device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

# Constants
m = 1.0  # Mass (kg)
b = 10  # Friction coefficient
k_p = 50  # Proportional gain
k_d = 10   # Derivative gain
dt = 0.01  # Time step
num_samples = 1000  # Number of samples in dataset

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

# Dataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# MLP Model Definition
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

# Model, Loss, Optimizer
hidden_sizes = [32, 64, 96, 128]
results = {}

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
axes = axes.flatten()

for i, hidden_size in enumerate(hidden_sizes):
    model = MLP(hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 500
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]:.6f}')

    results[hidden_size] = train_losses
    print(f'Hidden size: {hidden_size}, Final Epoch Loss: {train_losses[-1]:.6f}')
    
    # Testing Phase: Simulate trajectory tracking
    q_test = 0
    dot_q_test = 0
    q_real = []
    q_real_corrected = []
    
    # integration with only PD Control
    for k in range(len(t)):
        tau = k_p * (q_target[k] - q_test) + k_d * (dot_q_target[k] - dot_q_test)
        ddot_q_real = (tau - b * dot_q_test) / m
        dot_q_test += ddot_q_real * dt
        q_test += dot_q_test * dt
        q_real.append(q_test)
    
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
    axes[i].plot(t, q_real, 'g--', label='PD Only')
    axes[i].plot(t, q_real_corrected, 'b--', label='PD + MLP Correction')
    axes[i].set_title(f'Trajectory with Hidden Size {hidden_size}')
    axes[i].set_xlabel('Time [s]')
    axes[i].set_ylabel('Position')
    axes[i].legend()
# modify the layout of trajectories plot
plt.tight_layout()
if not os.path.exists('images/task1.1/'):
    os.makedirs('images/task1.1/')
plt.savefig('images/task1.1/trajectories.png')

# Plot training loss results
plt.figure(figsize=(12, 6))
for hidden_size in hidden_sizes:
    plt.plot(results[hidden_size], label=f'Hidden Size {hidden_size}')
plt.title('Training Loss by Hidden Layer Size')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images/task1.1/training_loss.png')
plt.show()
