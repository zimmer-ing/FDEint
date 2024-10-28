# Simple example of solving a fractional differential equation using FDEint
import torch
from FDEint import FDEint
import matplotlib.pyplot as plt

def fractional_diff_eq(t, x):
    return -x

t = torch.linspace(0., 20., 2001).unsqueeze(-1).unsqueeze(0)
y0 = torch.tensor([1., 1.]).unsqueeze(0)
alpha = torch.tensor([0.6])

solution = FDEint(fractional_diff_eq, t, y0, alpha)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(t.squeeze(), solution.squeeze())
plt.xlabel('Time')
plt.ylabel('Solution')
plt.title('Solution of the fractional differential equation')
plt.show()