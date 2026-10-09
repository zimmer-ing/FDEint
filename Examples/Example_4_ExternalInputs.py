"""Solve the same forced equation with samples and a callback, without plotting.

Install the local project first with ``python -m pip install -e .``, then run
``python Examples/Example_4_ExternalInputs.py`` from the repository root.
"""

import torch

from FDEint import FDEint


def rhs(t, y, u):
    return -y + u


def forcing(t):
    # During solver evaluation t has shape (batch, 1).
    return 0.1 * (t - 2.0)


def main():
    t = torch.linspace(2.0, 4.0, 101)
    y0 = torch.tensor([[1.0], [0.5]])
    alpha = torch.tensor(0.7)

    # Shared samples: (n_time, dim_u), broadcast to both initial states.
    u = forcing(t.unsqueeze(-1))
    from_samples = FDEint(rhs, t, y0, alpha, u=u)
    from_callback = FDEint(rhs, t, y0, alpha, u_fn=forcing)

    print("Solution shape:", tuple(from_samples.shape))
    print("Final states with samples:", from_samples[:, -1, :])
    print("Final states with callback:", from_callback[:, -1, :])
    print("Maximum difference:", (from_samples - from_callback).abs().max().item())


if __name__ == "__main__":
    main()
