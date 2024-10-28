import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from FDEint import FDEint
from tests.mittag_leffler import ml as mittag_leffler

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Simple fully connected network with 3 layers
        self.fc1 = nn.Linear(1, 10)  # Input layer with 10 hidden units
        self.fc2 = nn.Linear(10, 10)  # Hidden layer
        self.fc3 = nn.Linear(10, 1)  # Output layer with a single output

    def forward(self, x):
        # Forward pass through the network using ReLU activation functions
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Generate training data based on the Mittag-Leffler function
alpha = 0.6  # Fractional order
num_points = 1000
t_train = torch.linspace(0, 20, num_points).unsqueeze(-1).unsqueeze(0)  # Time tensor (1, num_points, 1)

# Generate true solution for Mittag-Leffler function
y_train = np.array([mittag_leffler(-ti.item() ** alpha, alpha) for ti in t_train.squeeze()])
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)  # Shape (1, num_points, 1)
y0 = y_train[:, 0, :]  # Initial condition is the first point of y_train

# Initialize the neural network model
model = SimpleNN()

# Training parameters
epochs = 500
lr = 0.0025
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

# Define the differential equation function using the trained neural network
def neural_fractional_diff_eq(t, x):
    return model(x)  # The model approximates the Mittag-Leffler dynamics

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Clear gradients
    # Solve the NFDE with the model as the function f
    y_pred = FDEint(neural_fractional_diff_eq, t_train, y0, torch.tensor([alpha]).unsqueeze(0))
    # Calculate loss as the difference between the predicted and true solution
    loss = criterion(y_pred, y_train)
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Solve the NFDE with the trained model as the function f to get the final solution
solution = FDEint(neural_fractional_diff_eq, t_train, y0, torch.tensor([alpha]).unsqueeze(0))

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(t_train.squeeze().detach().numpy(), solution.squeeze().detach().numpy(), label="NFDE Solution with NN")
plt.plot(t_train.squeeze().detach().numpy(), y_train.squeeze().detach().numpy(), label="True Mittag-Leffler Solution", linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Solution")
plt.title("Solution of the NFDE with Neural Network Approximation")
plt.legend()
plt.show()