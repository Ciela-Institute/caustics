# caustic

![tests](https://github.com/polairis-space/caustic/actions/workflows/python-app.yml/badge.svg?branch=mvp-dev)

The lensing pipeline of the future. This branch (`dev`) is for active developement of caustic. 
This is where feature branches are merged until a stable version is reached, which can then be merged with 
the `main` branch.

## Installation

First install [torchinterp1d](https://github.com/aliutkus/torchinterp1d):
```
git clone git@github.com:aliutkus/torchinterp1d.git
cd torchinterp1d
pip install .
```
Then install caustic:
```
git clone git@github.com:polairis-space/caustic.git
cd caustic
git checkout dev
pip install -e ".[dev]"  # editable install
```
The last line installs the optional dev dependencies.
