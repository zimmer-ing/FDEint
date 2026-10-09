from pathlib import Path

from setuptools import find_packages, setup


setup(
    name='fdeint',
    version='0.1.2',
    description='A PyTorch-based solver for Neural Fractional Differential Equations (NFDE) using a predictor-corrector method.',
    long_description=Path(__file__).with_name('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='Bernd Zimmering',
    author_email='bernd@zimmer-ing.de',
    url='https://github.com/zimmer-ing/FDEint',
    license='MIT',
    license_files=['LICENSE.txt'],
    packages=find_packages(include=['FDEint', 'FDEint.*']),
    install_requires=[
        'torch>=2.2.1',
        'tqdm',
        'numpy',
    ],
    extras_require={
        'test': ['pytest', 'scipy', 'numpy'],
        'examples': ['matplotlib', 'scipy', 'numpy'],
        'release': ['build', 'twine'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
