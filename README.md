<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Ciela-Institute/caustics/blob/main/media/caustics_logo.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Ciela-Institute/caustics/blob/main/media/caustics_logo_white.png?raw=true">
  <img alt="caustics logo" src="media/caustics_logo.png" width="70%">
</picture>

[![ssec](https://img.shields.io/badge/SSEC-Project-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAQAAABedl5ZAAAACXBIWXMAAAHKAAABygHMtnUxAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAMNJREFUGBltwcEqwwEcAOAfc1F2sNsOTqSlNUopSv5jW1YzHHYY/6YtLa1Jy4mbl3Bz8QIeyKM4fMaUxr4vZnEpjWnmLMSYCysxTcddhF25+EvJia5hhCudULAePyRalvUteXIfBgYxJufRuaKuprKsbDjVUrUj40FNQ11PTzEmrCmrevPhRcVQai8m1PRVvOPZgX2JttWYsGhD3atbHWcyUqX4oqDtJkJiJHUYv+R1JbaNHJmP/+Q1HLu2GbNoSm3Ft0+Y1YMdPSTSwQAAAABJRU5ErkJggg==&style=plastic)](https://escience.washington.edu/software-engineering/ssec/)
[![CI](https://github.com/Ciela-Institute/caustics/actions/workflows/ci.yml/badge.svg)](https://github.com/Ciela-Institute/caustics/actions/workflows/ci.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Documentation Status](https://readthedocs.org/projects/caustics/badge/?version=latest)](https://caustics.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/caustics.svg)](https://pypi.org/project/caustics/)
[![coverage](https://img.shields.io/codecov/c/github/Ciela-Institute/caustics)](https://app.codecov.io/gh/Ciela-Institute/caustics)

# caustics

The lensing pipeline of the future: GPU-accelerated,
automatically-differentiable, highly modular. Currently under heavy development:
expect interface changes and some imprecise/untested calculations.

## Installation

Simply install caustics from PyPI:

```bash
pip install caustics
```

## Documentation

Please see our [documentation page](https://caustics.readthedocs.io/en/latest/)
for more detailed information.

## Contribution

We welcome contributions from collaborators and researchers interested in our
work. If you have improvements, suggestions, or new findings to share, please
submit an issue or pull request. Your contributions help advance our research
and analysis efforts.

To get started with your development (or fork), click the "Open with GitHub
Codespaces" button below to launch a fully configured development environment
with all the necessary tools and extensions.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/uw-ssec/caustics?quickstart=1)

Instruction on how to contribute to this project can be found in the
CONTRIBUTION.md

Some guidelines:

- Please use `isort` and `black` to format your code.
- Use `CamelCase` for class names and `snake_case` for variable and method
  names.
- Open up issues for bugs/missing features.
- Use pull requests for additions to the code.
- Write tests that can be run by [`pytest`](https://docs.pytest.org/).

Thanks to our contributors so far!

[![Contributors](https://contrib.rocks/image?repo=Ciela-Institute/caustics)](https://github.com/Ciela-Institute/caustics/graphs/contributors)
