[![tests](https://github.com/Ciela-Institute/caustic/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Ciela-Institute/caustic/actions)
[![PyPI version](https://badge.fury.io/py/caustic.svg)](https://pypi.org/project/caustic/)
# caustic

The lensing pipeline of the future: GPU-accelerated, automatically-differentiable,
highly modular. Currently under heavy development: expect interface changes and
some imprecise/untested calculations.

## Installation

Manually install the [torchinterp1d](https://github.com/aliutkus/torchinterp1d)
dependency:
```bash
git clone git@github.com:aliutkus/torchinterp1d.git
cd torchinterp1d
pip install .
```
Then install caustic from PyPI:
```bash
pip install caustic
```

## Contributing

Please reach out to one of us if you're interested in contributing!

To start, follow the installation instructions, replacing the last line with
```bash
pip install -e ".[dev]"
```
This creates an editable install and installs the dev dependencies.

Please use `isort` and `black` to format your code. Open up issues for bugs/missing
features. Use pull requests for additions to the code. Write tests that can be run
by [`pytest`](https://docs.pytest.org/).
