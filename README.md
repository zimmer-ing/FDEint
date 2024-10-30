# FDEint

[![PyPI version](https://badge.fury.io/py/FDEint.svg)](https://badge.fury.io/py/FDEint)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FDEint** is a Python package designed for solving fractional differential equations through a predictor-corrector method. This package was developed as part of the work presented in the following paper:

> Zimmering, B., Coelho, C., & Niggemann, O. (2024). *Optimising Neural Fractional Differential Equations for Performance and Efficiency*. In *Proceedings of the 1st ECAI Workshop on Machine Learning Meets Differential Equations: From Theory to Applications*. [Available online](https://proceedings.mlr.press/v255/zimmering24a.html).

For a deeper understanding of the mathematical foundations and optimization techniques used in **FDEint**, please refer to the paper linked above. The code linked to the paper is also is available on [GitHub](https://github.com/zimmer-ing/Neural-FDE).
## Installation

You can install the package using python >=3.8 and pip:

```bash
pip install FDEint
```
## Usage

### Example 1: Solving a Single Dimensional Fractional Differential Equation    
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

### Example 2: Using FDEint to learn a Neural Fractional Differential Equation
Please find the code for this example in the [examples](Examples) folder.

## Paper

The methodology and optimization strategies for the FDEint package are detailed in the paper *"Optimising Neural Fractional Differential Equations for Performance and Efficiency"*. You can access the paper in the following ways:

View the [official PMLR publication](https://proceedings.mlr.press/v255/zimmering24.html) (once available) or the version from the github repo [here](https://github.com/zimmer-ing/Neural-FDE).

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
This project is licensed under the MIT License. Please see the [LICENSE](LICENSE.txt) file for more details.
