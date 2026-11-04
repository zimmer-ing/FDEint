# FDEint

[![PyPI version](https://badge.fury.io/py/FDEint.svg)](https://badge.fury.io/py/FDEint)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FDEint** solves Caputo fractional differential equations with a fixed-step
predictor-corrector (PECE) method implemented in PyTorch. It supports batched states,
neural right-hand sides, and external inputs supplied as samples or a callable.
The solver is differentiable through the initial state, fractional order,
input values, and model parameters.

This package was developed as part of the work presented in the following paper:

> Zimmering, B., Coelho, C., & Niggemann, O. (2024). *Optimising Neural Fractional Differential Equations for Performance and Efficiency*. In *Proceedings of the 1st ECAI Workshop on Machine Learning Meets Differential Equations: From Theory to Applications*. [Available online](https://proceedings.mlr.press/v255/zimmering24a.html).

The paper describes the method and its evaluation; its reported performance
and accuracy refer to those experiments. The accompanying research code is
available on [GitHub](https://github.com/zimmer-ing/Neural-FDE).

## Installation

Python 3.8 or newer and PyTorch 2.2.1 or newer are required:

```bash
pip install FDEint
```

To use the source checkout:

```bash
python -m pip install -e .
```

The solver supports CPU and CUDA tensors. Put the initial state and any neural
network on the desired device before calling it. Computation defaults to
`torch.float32`; pass `dtype=torch.float64` for double precision and use matching
network weights. MPS is not a supported target for this release.

## Usage

### Solve an equation without inputs

For an equation of the form `D^alpha y(t) = f(t, y(t))`, keep the existing
two-argument right-hand side:

```python
import torch
from FDEint import FDEint

def rhs(t, y):
    return -y

t = torch.linspace(0.0, 2.0, 101)
y0 = torch.tensor([1.0])
alpha = torch.tensor(0.7)

solution = FDEint(rhs, t, y0, alpha)
print(solution.shape)  # torch.Size([1, 101, 1])
```

### Supply external input samples

For `D^alpha y(t) = f(t, y(t), u(t))`, supply samples at the requested observation
times with `u=`. The solver interpolates these samples linearly at its internal
integration times:

```python
def forced_rhs(t, y, u):
    return -y + u

u = torch.sin(t).unsqueeze(-1)  # (n_time, dim_u)
solution = FDEint(forced_rhs, t, y0, alpha, u=u)
```

### Supply an input function

Use `u_fn=` for a callable input, such as an analytic signal or a neural network.
It receives a time tensor of shape `(batch, 1)` and must return a tensor of
shape `(batch, dim_u)`:

```python
def input_function(t):
    return torch.sin(t)

solution = FDEint(forced_rhs, t, y0, alpha, u_fn=input_function)
```

Pass either `u` or `u_fn` in a single call. When neither is supplied, the solver
calls `f(t, y)`; otherwise it calls `f(t, y, u)`.

The runnable [external-input example](https://github.com/zimmer-ing/FDEint/blob/main/Examples/Example_4_ExternalInputs.py)
uses both interfaces, two initial states, and a time grid starting at 2.0:

```bash
python Examples/Example_4_ExternalInputs.py
```

### Learn a neural fractional differential equation

The right-hand side can be an ordinary callable or a `torch.nn.Module`.
Autograd tracks its parameters, `y0`, tensor-valued `alpha`, and tensor-valued
inputs. For `u_fn`, autograd tracks differentiable operations and parameters
inside the callable:

```python
class Dynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, t, y):
        return -self.rate * y

model = Dynamics()
alpha = torch.tensor(0.7, requires_grad=True)
solution = FDEint(model, t, y0, alpha)
loss = solution.square().mean()
loss.backward()
print(model.rate.grad, alpha.grad)
```

Keep a learned `alpha` within `(0, 1]`, for example with a constrained
parameterization. Gradients with respect to the observation grid `t` or step
size `h` are not a supported API.

Further training examples are in the [Examples folder](https://github.com/zimmer-ing/FDEint/tree/main/Examples).
Examples 1–3 use Matplotlib; the training examples also use SciPy and a reference
helper included in the source checkout. Install their optional dependencies
and run them as modules from the repository root, for example:

```bash
python -m pip install -e '.[examples]'
python -m Examples.Example_3_LearningNFDE_alpha
```

The examples and reference helper are included in the source archive; the
installed wheel contains the solver package.

## API

```python
FDEint(f, t, y0, alpha, h=None, dtype=torch.float32,
       DEBUG=False, return_internals=False, u=None, u_fn=None)
```

| Argument | Meaning |
| --- | --- |
| `f` | Callable or `torch.nn.Module`; returns a tensor of shape `(batch, dim_y)`. |
| `t` | Finite, strictly increasing observation times, with at least two points per trajectory. |
| `y0` | Initial state at the first observation time; shape `(dim_y,)` or `(batch, dim_y)`. |
| `alpha` | One scalar order, shared across the batch and state dimensions, with `0 < alpha <= 1`; a number or one-element tensor. |
| `h` | Optional positive, finite scalar step size. Defaults to the smallest spacing in `t`; larger values are reduced to that spacing with a warning. |
| `dtype` | Computation dtype, `torch.float32` (default) or `torch.float64`. |
| `DEBUG` | Show a progress bar when `True`. |
| `return_internals` | Return the observation solution together with the internal grid and states when `True`. |
| `u` | Optional sampled external inputs at the times in `t`; see shapes below. |
| `u_fn` | Optional input callable evaluated at internal times; mutually exclusive with `u`. |

During evaluation, `f` receives `t` with shape `(batch, 1)`, `y` with shape
`(batch, dim_y)`, and, when applicable, `u` with shape `(batch, dim_u)`.
All solver tensors use the device of `y0` and the requested computation dtype;
move any modules to that device and dtype as well.

### Tensor shapes

`batch`, `n_time`, `dim_y`, and `dim_u` denote the number of trajectories,
observation times, state components, and input components, respectively.

| Tensor | Accepted shape | Interpretation |
| --- | --- | --- |
| `y0` | `(dim_y,)` | One trajectory. |
| `y0` | `(batch, dim_y)` | A batch of initial states. |
| `t` | `(n_time,)` or `(n_time, 1)` | One observation grid shared across the batch. |
| `t` | `(batch, n_time)` or `(batch, n_time, 1)` | Observation grid per trajectory. |
| `u` | `(n_time,)` | One scalar signal shared across the batch. |
| `u` | `(n_time, dim_u)` | A vector signal shared across the batch. |
| `u` | `(batch, n_time)` | One scalar signal per trajectory. |
| `u` | `(batch, n_time, dim_u)` | Explicit batched vector inputs. |

A singleton batch dimension in `t` or `u` is broadcast to the batch of `y0`.
Batched observation grids must contain the same number of points and start at
the same time; their later observation times and final times may differ.
Samples in `u` correspond to the observation grid of each trajectory. When
`batch == n_time`, a two-dimensional `u` is interpreted as shared
`(n_time, dim_u)` samples. Use `(batch, n_time, dim_u)` to express batched inputs
unambiguously. `u_fn` must return the full `(batch, dim_u)` shape.

The normal return value always has shape `(batch, n_time, dim_y)`, including
single trajectories. With `return_internals=True`, the return value is
`(solution, t_internal, y_internal)`, where the latter two tensors have shapes
`(batch, n_internal)` and `(batch, n_internal, dim_y)`.

### Integration grid and input boundaries

The solver starts at the first time in `t`, including when that time is nonzero.
It integrates on a uniform internal grid and linearly interpolates the solution
at the requested observation times. Fractional memory starts at that initial
time; it does not include any history before it.

The internal grid may end less than one step beyond the last requested time.
Sampled inputs are held at their endpoint values for internal evaluations
outside each trajectory's sample range. Define `u_fn` over the complete internal
grid, including that possible final step. For discontinuous inputs, choose an
appropriate grid and remember that sampled inputs use linear interpolation.
Check the numerical sensitivity to `h` and `dtype` for your application.

The method retains dense fractional history. For `N` internal steps, work grows
quadratically with `N`, and solver arrays require storage proportional to
`batch * N * dim_y`. Autograd can retain additional intermediate values during
training. Long horizons or very small steps can therefore be expensive.
History-weight calculations involve subtraction of nearby values, which can
lose precision in `float32` on long grids. Prefer `dtype=torch.float64` for long
horizons and numerical validation, and verify convergence as you refine `h`;
double precision alone does not guarantee a desired error tolerance.

## Development and release

Install test dependencies with `python -m pip install -e '.[test]'`.
See [CHANGELOG.md](https://github.com/zimmer-ing/FDEint/blob/main/CHANGELOG.md) for release changes and
[RELEASING.md](https://github.com/zimmer-ing/FDEint/blob/main/RELEASING.md) for local checks and automatic publishing
through GitHub releases. The release workflow tests the installed wheel on Python 3.8–3.15 before uploading it to PyPI.

## Paper

The methodology and optimization strategies for the FDEint package are detailed in the paper *"Optimising Neural Fractional Differential Equations for Performance and Efficiency"*. You can access the paper in the following ways:

View the [official PMLR publication](https://proceedings.mlr.press/v255/zimmering24a.html) or the version from our github repo [here](https://github.com/zimmer-ing/Neural-FDE).

## Citation

If you find FDEint useful for your research, please consider citing the following paper:

```
@InProceedings{pmlr-v255-zimmering24a,
  title = 	 {Optimising Neural Fractional Differential Equations for Performance and Efficiency},
  author =       {Zimmering, Bernd and Coelho, Cec\'{i}lia and Niggemann, Oliver},
  booktitle = 	 {Proceedings of the 1st ECAI Workshop on "Machine Learning Meets Differential Equations: From Theory to Applications"},
  pages = 	 {1--22},
  year = 	 {2024},
  editor = 	 {Coelho, Cecı́lia and Zimmering, Bernd and Costa, M. Fernanda P. and Ferrás, Luı́s L. and Niggemann, Oliver},
  volume = 	 {255},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {20 Oct},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v255/main/assets/zimmering24a/zimmering24a.pdf},
  url = 	 {https://proceedings.mlr.press/v255/zimmering24a.html},
  abstract = 	 {Neural Ordinary Differential Equations (NODEs) are well-established architectures that fit an ODE, modelled by a neural network (NN), to data, effectively modelling complex dynamical systems. Recently, Neural Fractional Differential Equations (NFDEs) were proposed, inspired by NODEs, to incorporate non-integer order differential equations, capturing memory effects and long-range dependencies. In this work, we present an optimised implementation of the NFDE solver, achieving up to 570 times faster computations and up to 79 times higher accuracy. Additionally, the solver supports efficient multidimensional computations and batch processing. Furthermore, we enhance the experimental design to ensure a fair comparison of NODEs and NFDEs by implementing rigorous hyperparameter tuning and using consistent numerical methods. Our results demonstrate that for systems exhibiting fractional dynamics, NFDEs significantly outperform NODEs, particularly in extrapolation tasks on unseen time horizons. Although NODEs can learn fractional dynamics when time is included as a feature to the NN, they encounter difficulties in extrapolation due to reliance on explicit time dependence. The code is available at https://github.com/zimmer-ing/Neural-FDE}
}  
```

## License
This project is licensed under the MIT License. Please see the [LICENSE](https://github.com/zimmer-ing/FDEint/blob/main/LICENSE.txt) file for more details.
