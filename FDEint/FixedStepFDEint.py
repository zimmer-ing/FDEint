# Copyright (C) 2024 Bernd Zimmering
# This work is licensed under a MIT License. Please see the LICENSE.txt file for details.
#
# If you find this solver useful in your research, please consider citing:
# @InProceedings{pmlr-v255-zimmering24a,
#   title = 	 {Optimising Neural Fractional Differential Equations for Performance and Efficiency},
#   author =       {Zimmering, Bernd and Coelho, Cec\'{i}lia and Niggemann, Oliver},
#   booktitle = 	 {Proceedings of the 1st ECAI Workshop on "Machine Learning Meets Differential Equations: From Theory to Applications"},
#   pages = 	 {1--22},
#   year = 	 {2024},
#   editor = 	 {Coelho, Cecı́lia and Zimmering, Bernd and Costa, M. Fernanda P. and Ferrás, Luı́s L. and Niggemann, Oliver},
#   volume = 	 {255},
#   series = 	 {Proceedings of Machine Learning Research},
#   month = 	 {20 Oct},
#   publisher =    {PMLR},
#   pdf = 	 {https://raw.githubusercontent.com/mlresearch/v255/main/assets/zimmering24a/zimmering24a.pdf},
#   url = 	 {https://proceedings.mlr.press/v255/zimmering24a.html},
#   abstract = 	 {Neural Ordinary Differential Equations (NODEs) are well-established architectures that fit an ODE, modelled by a neural network (NN), to data, effectively modelling complex dynamical systems. Recently, Neural Fractional Differential Equations (NFDEs) were proposed, inspired by NODEs, to incorporate non-integer order differential equations, capturing memory effects and long-range dependencies. In this work, we present an optimised implementation of the NFDE solver, achieving up to 570 times faster computations and up to 79 times higher accuracy. Additionally, the solver supports efficient multidimensional computations and batch processing. Furthermore, we enhance the experimental design to ensure a fair comparison of NODEs and NFDEs by implementing rigorous hyperparameter tuning and using consistent numerical methods. Our results demonstrate that for systems exhibiting fractional dynamics, NFDEs significantly outperform NODEs, particularly in extrapolation tasks on unseen time horizons. Although NODEs can learn fractional dynamics when time is included as a feature to the NN, they encounter difficulties in extrapolation due to reliance on explicit time dependence. The code is available at https://github.com/zimmer-ing/Neural-FDE}
# }

import torch
from tqdm import tqdm
import warnings
import torch.nn as nn

DEBUG = False


# Helper function to compute the Gamma function
def torchgamma(x):
    return torch.exp(torch.special.gammaln(x))


# Main FDE solver function
def FDEint(f, t, y0, alpha, h=None, dtype=torch.float32, DEBUG=False, return_internals=False):
    """
    Solves a fractional differential equation (FDE) using a predictor-corrector method.

    Parameters:
    - f: function (or an nn.Module) that defines the fractional differential equation (dy/dt = f(t, y)).
         It can be either a function or a neural network.
    - t: time points at which the solution is desired (torch.Tensor).
    - y0: initial condition (torch.Tensor).
    - alpha: fractional order of the differential equation (0 < alpha <= 1).
    - h: step size (optional). If not provided, it is set to the smallest time difference in t.
    - dtype: data type for computations (default: torch.float32).
    - DEBUG: if True, shows a progress bar for the integration steps.
    - return_internals: if True, returns additional internal time and solution values.

    Returns:
    - Approximate solution y at the time points t (and optionally internal time and solution values).
    """

    # Ensure alpha is in the correct range and h is positive
    assert 0 < alpha <= 1, "Alpha must be between 0 and 1"
    assert h is None or h > 0, "Step size must be greater than 0 if provided"

    # Check if the time tensor t needs to be expanded to have a batch dimension
    device = y0.device
    if len(t.shape) == 1:
        t = t.unsqueeze(-1).unsqueeze(0).repeat(y0.shape[0], 1, 1)
    elif len(t.shape) == 2 and t.shape[-1] == 1:
        t = t.unsqueeze(0)

    # Ensure y0 has the right shape for batch operations
    if len(y0.shape) == 1:
        y0 = y0.unsqueeze(0)

    # Check if alpha is a parameter in autograd (if so, clone some tensors to autograd avoid errors)
    alpha_is_in_autograd = isinstance(alpha, nn.Parameter)

    # Convert all relevant tensors to the specified dtype
    t, y0, alpha = t.to(dtype), y0.to(dtype), alpha.to(dtype)

    # Determine the smallest time step in t and adjust h if necessary
    dt_min = torch.min(torch.diff(t.squeeze()))
    if h is None:
        h = dt_min
    elif dt_min < h and dt_min - h > 1e-6:
        warnings.warn(
            f"The minimum time difference in desired time points ({dt_min}) is smaller than the step size h ({h}). Adjusting h to {dt_min}.",
            UserWarning)
        h = dt_min

    # Tensor initializations for storing intermediate results
    alpha = alpha.squeeze()
    batch_size, dim_y = y0.shape
    N = int(torch.ceil((t.max() - t.min()) / h).item()) + 1  # Number of time steps
    y_internal = torch.zeros((batch_size, N + 1, dim_y), device=device, dtype=dtype)
    t_internal = torch.zeros((batch_size, N + 1), device=device, dtype=dtype)
    fk_mem = torch.zeros_like(y_internal)  # Memory for previous evaluations of f
    y_internal[:, 0, :] = y0  # Set initial condition

    # Precompute coefficients for the predictor and corrector steps
    k = torch.arange(1, N + 1, device=device, dtype=dtype)
    b = ((k ** alpha) - (k - 1) ** alpha).unsqueeze(-1)  # Predictor coefficients
    a = ((k + 1) ** (alpha + 1) - 2 * k ** (alpha + 1) + (k - 1) ** (alpha + 1)).unsqueeze(-1)  # Corrector coefficients
    b = torch.cat([torch.zeros((1, 1), device=device, dtype=dtype), b], dim=0)
    a = torch.cat([torch.zeros((1, 1), device=device, dtype=dtype), a], dim=0)

    # Precompute Gamma functions
    gamma_alpha1 = torchgamma(alpha + 1)
    gamma_alpha2 = torchgamma(alpha + 2)

    # Initial function evaluation f(0, y0)
    f0 = f(t[:, 0, :], y0)
    fk_mem[:, 0, :] = f0.clone() if alpha_is_in_autograd else f0

    y_new = y0
    kn = torch.arange(0, N, device=device, dtype=torch.long)

    # Main loop for time stepping
    for j in tqdm(range(1, N + 1), desc="Time Steps Progress", disable=not DEBUG):
        t_act = (j * h).repeat(batch_size).to(dtype)  # Current time step
        t_internal[:, j] = t_act

        # Compute f at the current step (can be a neural network or a function)
        fkj = f(t_act.unsqueeze(-1), y_new)
        fk_mem[:, j, :] = fkj.clone() if alpha_is_in_autograd else fkj

        # Retrieve previously computed function values
        fk = fk_mem[:, :j, :].clone() if alpha_is_in_autograd else fk_mem[:, :j, :]

        # Predictor step: Estimate the next value using previous steps
        bjk = b[j - kn[:j]]
        y_p = y0 + (h ** alpha / gamma_alpha1) * torch.sum(bjk * fk, dim=1)

        # Corrector step: Refine the predicted value
        ajk = a[(j - kn[:j])[1:]]
        y_new = y0 + (h ** alpha / gamma_alpha2) * (
                f(t_act.unsqueeze(-1), y_p)
                + ((j - 1) ** (alpha + 1) - (j - 1 - alpha) * j ** alpha) * f0
                + torch.sum(ajk * fk[:, 1:, :], dim=1)
        )

        # Store the new value in the internal solution array
        y_internal[:, j, :] = y_new.clone() if alpha_is_in_autograd else y_new

    # Return the solution, optionally with internal values
    return get_outputs(y_internal, t_internal, t) if not return_internals else (
    get_outputs(y_internal, t_internal, t), t_internal, y_internal)


# Function to interpolate results at desired time points
def get_outputs(y_internal, t_internal, t):
    """
    Interpolates the solution values at the desired time points.

    Parameters:
    - y_internal: solution values at internal time points.
    - t_internal: internal time points corresponding to y_internal.
    - t: desired time points.

    Returns:
    - Interpolated solution values at the desired time points.
    """
    batch_size, num_internal_points = t_internal.shape
    _, num_desired_points, _ = t.shape
    _, _, features = y_internal.shape

    # Find indices of time points just before and after each desired time point
    idx = torch.searchsorted(t_internal, t.squeeze(-1), right=True)
    idx_y = idx.unsqueeze(-1).repeat(1, 1, features)

    # Gather the values just before and after the desired time points
    t0 = torch.gather(t_internal, 1, idx - 1).unsqueeze(-1)
    t1 = torch.gather(t_internal, 1, idx).unsqueeze(-1)
    y0 = torch.gather(y_internal, 1, idx_y - 1)
    y1 = torch.gather(y_internal, 1, idx_y)

    # Perform linear interpolation
    return y0 + (y1 - y0) * (t - t0) / (t1 - t0)


