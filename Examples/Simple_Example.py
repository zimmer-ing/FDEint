# test_installation.py
import torch
from FDEint import FDEint

def fractional_diff_eq(t, x):
    return -x

t = torch.linspace(0., 20., 2001).unsqueeze(-1).unsqueeze(0)
y0 = torch.tensor([1., 1.]).unsqueeze(0)
alpha = torch.tensor([0.6])

solution = FDEint(fractional_diff_eq, t, y0, alpha)
print("Solution:", solution)