from setuptools import setup, find_packages

setup(
    name='FDEint',
    version='0.1.0',
    description='A package for solving fractional differential equations using a predictor-corrector method.',
    long_description="""
    If you find this solver useful in your research, please consider citing:

    @InProceedings{zimmering24,
        title     = {Optimising Neural Fractional Differential Equations for Performance and Efficiency},
        author    = {Zimmering, Bernd and Coelho, Cec\'{\i}lia and Niggemann, Oliver},
        booktitle = {Proceedings of the 1st ECAI Workshop on “Machine Learning Meets Differential Equations: From Theory to Applications”},
        year      = {2024},
        pages     = {1-24},
        volume    = {255},
        series    = {Proceedings of Machine Learning Research},
        publisher = {PMLR},
        url       = {}
    }
    """,
    long_description_content_type='text/plain',
    author='Bernd Zimmering',
    author_email='bernd@zimmer-ing.de',
    url='https://github.com/zimmer-ing/FDEint',
    license='CC-BY-4.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.2.1',
        'tqdm',
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Creative Commons Attribution 4.0 International License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)