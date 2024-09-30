# FDEint

FDEint is a Python package for solving fractional differential equations using a predictor-corrector method.

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

t = torch.linspace(0., 20., 2001)
y0 = torch.tensor([1., 1.])
alpha = torch.tensor([0.6])

solution = FDEint(fractional_diff_eq, t, y0, alpha)
```
## License
This project is licensed under the Creative Commons Attribution 4.0 International License.
