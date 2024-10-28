from setuptools import setup, find_packages

setup(
    name='FDEint',
    version='0.1.0',
    description='A PyTorch-based solver for Neural Fractional Differential Equations (NFDE) using a predictor-corrector method.',
    long_description="""
    **FDEint** is a specialized Python package for solving Neural Fractional Differential Equations (NFDEs) with high efficiency and accuracy, specifically designed for PyTorch models. The neural network provided to the solver must be a subclass of `torch.nn.Module`. This package enables NFDEs to capture memory effects and long-range dependencies effectively.

    If you find **FDEint** valuable for your research, please consider citing the following work:

    ```
    @InProceedings{pmlr-v255-zimmering24a,
      title = {Optimising Neural Fractional Differential Equations for Performance and Efficiency},
      author = {Zimmering, Bernd and Coelho, Cec\'{i}lia and Niggemann, Oliver},
      booktitle = {Proceedings of the 1st ECAI Workshop on "Machine Learning Meets Differential Equations: From Theory to Applications"},
      pages = {1--22},
      year = {2024},
      editor = {Coelho, Cecı́lia and Zimmering, Bernd and Costa, M. Fernanda P. and Ferrás, Luı́s L. and Niggemann, Oliver},
      volume = {255},
      series = {Proceedings of Machine Learning Research},
      month = {20 Oct},
      publisher = {PMLR},
      pdf = {https://raw.githubusercontent.com/mlresearch/v255/main/assets/zimmering24a/zimmering24a.pdf},
      url = {https://proceedings.mlr.press/v255/zimmering24a.html},
      abstract = {Neural Ordinary Differential Equations (NODEs) are well-established architectures that fit an ODE, modeled by a neural network (NN), to data, effectively capturing complex dynamical systems. Recently, Neural Fractional Differential Equations (NFDEs) were proposed to incorporate non-integer order differential equations, capturing memory effects and long-range dependencies. In this work, we present an optimised implementation of the NFDE solver, achieving up to 570 times faster computations and up to 79 times higher accuracy. Our results demonstrate that for systems with fractional dynamics, NFDEs significantly outperform NODEs, particularly in extrapolation tasks on unseen time horizons. The code is available at https://github.com/zimmer-ing/Neural-FDE.}
    }
    ```

    The solver and pseudocode are thoroughly described in the paper, with pseudocode available in Appendix A. Full documentation and a usage example can be found on [GitHub](https://github.com/zimmer-ing/FDEint).
    """,
    long_description_content_type='text/markdown',
    author='Bernd Zimmering',
    author_email='bernd@zimmer-ing.de',
    url='https://github.com/zimmer-ing/FDEint',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch>=2.2.1',
        'tqdm',
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)