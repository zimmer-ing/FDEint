# Copyright (C) 2024-2026 Bernd Zimmering
# Licensed under the MIT License; see LICENSE.txt.
# Please cite Zimmering, Coelho & Niggemann (2024):
# https://proceedings.mlr.press/v255/zimmering24a.html

import warnings

import torch
from tqdm import tqdm


def torchgamma(x):
    """Compute the Gamma function without leaving PyTorch's computation graph."""
    return torch.exp(torch.special.gammaln(x))


def _prepare_time_tensor(t, y0, dtype=torch.float32):
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor.")
    batch_size = 1 if y0.ndim == 1 else y0.shape[0]
    if t.ndim == 1:
        t = t[None, :, None]
    elif t.ndim == 2:
        t = t[None, :, :] if t.shape[-1] == 1 else t.unsqueeze(-1)
    elif t.ndim != 3 or t.shape[-1] != 1:
        raise ValueError("t must have shape (n_time,), (n_time, 1), (batch, n_time), or (batch, n_time, 1).")
    if t.shape[0] == 1:
        t = t.expand(batch_size, -1, -1)
    if t.shape[0] != batch_size:
        raise ValueError("Batch size of t must match batch size of y0.")
    t = t.to(device=y0.device, dtype=dtype).contiguous()
    if t.shape[1] < 2:
        raise ValueError("t must contain at least two time points.")
    if not torch.isfinite(t).all() or not (torch.diff(t, dim=1) > 0).all():
        raise ValueError("Time points must be finite and strictly increasing in each batch.")
    if not (t[:, 0, 0] == t[0, 0, 0]).all():
        raise ValueError("All batches must have the same initial time.")
    return t


def _prepare_input_tensor(u, t, y0, dtype=torch.float32):
    if u is None:
        return None
    if not isinstance(u, torch.Tensor):
        raise TypeError("u must be a torch.Tensor.")
    batch_size, n_time = t.shape[:2]
    if u.ndim == 1:
        u = u[None, :, None]
    elif u.ndim == 2:
        # Shared (n_time, dim_u) takes precedence for ambiguous 2D inputs.
        if u.shape[0] == n_time:
            u = u.unsqueeze(0)
        elif u.shape[0] in (1, batch_size) and u.shape[1] == n_time:
            u = u.unsqueeze(-1)
        else:
            raise ValueError("u must have shape (n_time, dim_u) or (batch, n_time).")
    elif u.ndim != 3:
        raise ValueError("u must have shape (n_time,), (n_time, dim_u), (batch, n_time), or (batch, n_time, dim_u).")
    if u.shape[0] == 1:
        u = u.expand(batch_size, -1, -1)
    if u.shape[0] != batch_size or u.shape[1] != n_time or u.shape[2] < 1:
        raise ValueError("u must match the batch and time dimensions of t and contain at least one input channel.")
    u = u.to(device=y0.device, dtype=dtype)
    if not torch.isfinite(u).all():
        raise ValueError("u must contain finite values.")
    return u


def _linear_interpolate_batched(t_query, t_grid, values):
    """Interpolate batched samples, holding endpoint values outside the grid."""
    grid = t_grid.squeeze(-1).contiguous()
    query = t_query.squeeze(-1).contiguous()
    query = torch.maximum(torch.minimum(query, grid[:, -1:]), grid[:, :1])
    idx = torch.searchsorted(grid, query, right=True).clamp(1, grid.shape[1] - 1)
    t0 = grid.gather(1, idx - 1)
    t1 = grid.gather(1, idx)
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    v0 = values.gather(1, gather_idx - 1)
    v1 = values.gather(1, gather_idx)
    weight = ((query - t0) / (t1 - t0)).unsqueeze(-1)
    return v0 + weight * (v1 - v0)


def _call_f(f, t_in, y_in, u_in=None):
    result = f(t_in, y_in) if u_in is None else f(t_in, y_in, u_in)
    if not isinstance(result, torch.Tensor) or result.shape != y_in.shape:
        raise ValueError("f must return a tensor with the same (batch, dim_y) shape as y.")
    return result.to(device=y_in.device, dtype=y_in.dtype)


def FDEint(
    f,
    t: torch.Tensor,
    y0: torch.Tensor,
    alpha,
    h=None,
    dtype=torch.float32,
    DEBUG=False,
    return_internals=False,
    u: torch.Tensor = None,
    u_fn=None,
):
    """Solve a Caputo FDE of scalar order 0 < alpha <= 1 using PECE.

    Existing calls with ``f(t, y)`` remain valid. With either ``u`` or ``u_fn``,
    the right-hand side is called as ``f(t, y, u)``. Sampled inputs are linearly
    interpolated on the requested time grid, with endpoint values held outside
    that grid. ``u_fn`` receives times of shape (batch, 1) and must return a
    tensor of shape (batch, dim_u).

    ``y0`` has shape (dim_y,) or (batch, dim_y). ``t`` may be (n_time,),
    (n_time, 1), (batch, n_time), or (batch, n_time, 1). All batches must start
    at the same time. ``u`` may be (n_time,), (n_time, dim_u), (batch, n_time),
    or (batch, n_time, dim_u). For ambiguous 2D inputs, (n_time, dim_u) takes
    precedence; use a 3D tensor for unambiguous batched inputs.

    ``h`` defaults to the minimum requested time spacing. Larger explicit
    values are reduced to that spacing with a warning. Computation uses
    ``dtype`` (float32 or float64) and the device of ``y0``. The fixed internal
    grid may extend less than one step beyond the last requested time, so an
    input callback must be defined through that grid endpoint.

    Returns a tensor of shape (batch, n_time, dim_y). If ``return_internals``
    is True, returns (solution, internal_times, internal_states), with shapes
    (batch, n_time, dim_y), (batch, n_internal), and
    (batch, n_internal, dim_y). Autograd supports y0, alpha, sampled inputs,
    and parameters in f or u_fn. Time grids and h are numerical controls.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64.")
    if not isinstance(y0, torch.Tensor):
        raise TypeError("y0 must be a torch.Tensor.")
    if y0.ndim not in (1, 2) or y0.numel() == 0:
        raise ValueError("y0 must have shape (dim_y,) or (batch, dim_y) and be nonempty.")
    if u is not None and u_fn is not None:
        raise ValueError("Use either u or u_fn, not both.")
    if not callable(f) or (u_fn is not None and not callable(u_fn)):
        raise TypeError("f and u_fn must be callable.")

    device = y0.device
    y0 = y0.unsqueeze(0) if y0.ndim == 1 else y0
    y0 = y0.to(dtype=dtype)
    if not torch.isfinite(y0).all():
        raise ValueError("y0 must contain finite values.")
    t = _prepare_time_tensor(t, y0, dtype=dtype)
    u = _prepare_input_tensor(u, t, y0, dtype=dtype)
    alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
    if alpha.numel() != 1:
        raise ValueError("alpha must be a scalar fractional order.")
    alpha = alpha.reshape(())
    if not torch.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and satisfy 0 < alpha <= 1.")

    dt_min = torch.diff(t, dim=1).min()
    if h is None:
        h = dt_min
    else:
        h = torch.as_tensor(h, device=device, dtype=dtype)
        if h.numel() != 1:
            raise ValueError("h must be a scalar step size.")
        h = h.reshape(())
        if not torch.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and greater than zero.")
        # Avoid warnings caused solely by rounding uniform time samples.
        if h > dt_min and not torch.isclose(h, dt_min, rtol=1e-5, atol=0):
            warnings.warn("h exceeds the minimum requested time spacing; reducing h to that spacing.", UserWarning, stacklevel=2)
            h = dt_min

    batch_size, dim_y = y0.shape
    t_start = t[0, 0, 0]
    span = t[:, -1, 0].max() - t_start
    step_count = span / h
    if not torch.isfinite(step_count) or not t_start + h > t_start:
        raise ValueError("h is too small to represent a finite integration grid in the requested dtype.")
    nearest = torch.round(step_count)
    # Rounded uniform samples can otherwise add a full, unnecessary step.
    if torch.isclose(step_count, nearest, rtol=32 * torch.finfo(dtype).eps, atol=0):
        n_steps = max(1, int(nearest.item()))
        h = span / n_steps
    else:
        n_steps = int(torch.ceil(step_count).item())
    offsets = torch.arange(n_steps + 1, device=device, dtype=dtype) * h
    internal_grid = t_start + offsets
    if not torch.isfinite(internal_grid).all() or not (torch.diff(internal_grid) > 0).all():
        raise ValueError("h is too small to represent increasing times in the requested dtype.")
    t_internal = internal_grid.unsqueeze(0).expand(batch_size, -1).contiguous()
    y_internal = torch.zeros((batch_size, n_steps + 1, dim_y), device=device, dtype=dtype)
    fk_mem = torch.zeros_like(y_internal)
    y_internal[:, 0, :] = y0

    k = torch.arange(1, n_steps + 1, device=device, dtype=dtype)
    b = (k ** alpha - (k - 1) ** alpha).unsqueeze(-1)
    a = ((k + 1) ** (alpha + 1) - 2 * k ** (alpha + 1) + (k - 1) ** (alpha + 1)).unsqueeze(-1)
    zero = torch.zeros((1, 1), device=device, dtype=dtype)
    b, a = torch.cat((zero, b)), torch.cat((zero, a))
    predictor_scale = h ** alpha / torchgamma(alpha + 1)
    corrector_scale = h ** alpha / torchgamma(alpha + 2)
    kn = torch.arange(n_steps, device=device)

    def input_at(time):
        if u_fn is not None:
            value = u_fn(time)
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != batch_size or value.shape[1] < 1:
                raise ValueError("u_fn must return a tensor of shape (batch, dim_u).")
            return value.to(device=device, dtype=dtype)
        if u is not None:
            return _linear_interpolate_batched(time.unsqueeze(1), t, u).squeeze(1)
        return None

    t_prev = t[:, 0, :]
    u_prev = input_at(t_prev)
    f0 = _call_f(f, t_prev, y0, u_prev)
    fk_mem[:, 0, :] = f0
    y_new = y0

    for j in tqdm(range(1, n_steps + 1), desc="Time Steps Progress", disable=not DEBUG):
        # The history entry must pair the corrected state with its own time/input.
        if j > 1:
            fk_mem[:, j - 1, :] = _call_f(f, t_prev, y_new, u_prev)
        t_act = t_internal[:, j:j + 1]
        u_act = input_at(t_act)
        # Later writes to fk_mem must not change saved tensors during backward,
        # including when alpha is fixed but f, y0, or inputs are trainable.
        fk = fk_mem[:, :j, :].clone() if torch.is_grad_enabled() else fk_mem[:, :j, :]
        bjk = b[j - kn[:j]]
        y_pred = y0 + predictor_scale * torch.sum(bjk * fk, dim=1)
        ajk = a[(j - kn[:j])[1:]]
        initial_weight = (j - 1) ** (alpha + 1) - (j - 1 - alpha) * j ** alpha
        y_new = y0 + corrector_scale * (
            _call_f(f, t_act, y_pred, u_act)
            + initial_weight * f0
            + torch.sum(ajk * fk[:, 1:, :], dim=1)
        )
        y_internal[:, j, :] = y_new
        t_prev, u_prev = t_act, u_act

    solution = get_outputs(y_internal, t_internal, t)
    if return_internals:
        return solution, t_internal, y_internal
    return solution


def get_outputs(y_internal, t_internal, t):
    """Linearly interpolate internal states at requested batched time points.

    ``y_internal`` and ``t_internal`` have shapes (batch, n_internal, dim_y)
    and (batch, n_internal). ``t`` has shape (batch, n_time, 1). Queries outside
    the internal grid are held at the nearest endpoint.
    """
    return _linear_interpolate_batched(t, t_internal.unsqueeze(-1), y_internal)
