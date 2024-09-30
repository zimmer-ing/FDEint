# FDEint

FDEint is a Python package for solving fractional differential equations using a predictor-corrector method. This package is part of the work presented in the paper “Optimising Neural Fractional Differential Equations for Performance and Efficiency” by Bernd Zimmering, Cecília Coelho, and Oliver Niggemann, featured at the 1st ECAI Workshop on “Machine Learning Meets Differential Equations: From Theory to Applications”.

## Installation

You can install the package using pip:

```bash
pip install .
```
## Usage
    
```python
import torch
from FDEint import FDEint

def fractional_diff_eq(t, x):
    return -x

t = torch.linspace(0., 20., 2001).unsqueeze(-1).unsqueeze(0)
y0 = torch.tensor([1., 1.]).unsqueeze(0)
alpha = torch.tensor([0.6])

solution = FDEint(fractional_diff_eq, t, y0, alpha)
```

## Paper

The methodology and optimization strategies for the FDEint package are detailed in the paper *"Optimising Neural Fractional Differential Equations for Performance and Efficiency"*. You can access the paper in the following ways:

- View the [official PMLR publication](https://proceedings.mlr.press/v255/zimmering24.html) (once available).
- Download the local version of the paper [here](zimmering24.pdf).

## Citation

If you find FDEint useful for your research, please consider citing the following paper:

```
@InProceedings{zimmering24,
    title     = {Optimising Neural Fractional Differential Equations for Performance and Efficiency},
    author    = {Zimmering, Bernd and Coelho, Cec\'{\i}lia and Niggemann, Oliver},
    booktitle = {1st ECAI Workshop on “Machine Learning Meets Differential Equations: From Theory to Applications”},
    year      = {2024},
    pages     = {1-24},
    volume    = {255},
    series    = {Proceedings of Machine Learning Research},
    publisher = {PMLR},
    url       = {}
}  
```

## License
This project is licensed under the Creative Commons Attribution 4.0 International License. Please see the [LICENSE](LICENSE) file for more details.
