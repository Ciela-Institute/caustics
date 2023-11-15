[![tests](https://github.com/Ciela-Institute/caustics/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Ciela-Institute/caustics/actions)
[![Docs](https://github.com/Ciela-Institute/caustics/actions/workflows/documentation.yaml/badge.svg)](https://github.com/Ciela-Institute/caustics/actions/workflows/documentation.yaml)
[![PyPI version](https://badge.fury.io/py/caustics.svg)](https://pypi.org/project/caustics/)
[![coverage](https://img.shields.io/codecov/c/github/Ciela-Institute/caustics)](https://app.codecov.io/gh/Ciela-Institute/caustics)
# caustics

The lensing pipeline of the future: GPU-accelerated, automatically-differentiable,
highly modular. Currently under heavy development: expect interface changes and
some imprecise/untested calculations.

## Installation 

Simply install caustics from PyPI:
```bash
pip install caustics
```

## Documentation

Please see our [documentation page](Ciela-Institute.github.io/caustics/) for more detailed information.

## Contributing

Please reach out to one of us if you're interested in contributing!

To start, follow the installation instructions, replacing the last line with
```bash
pip install -e ".[dev]"
```
This creates an editable install and installs the dev dependencies.

Some guidelines:
- Please use `isort` and `black` to format your code.
- Use `CamelCase` for class names and `snake_case` for variable and method names.
- Open up issues for bugs/missing features.
- Use pull requests for additions to the code.
- Write tests that can be run by [`pytest`](https://docs.pytest.org/).
