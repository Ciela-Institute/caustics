# caustic

The lensing pipeline of the future. This branch (`mvp-dev`) is for developing the
minimum viable version of caustic.

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
git checkout mvp-dev
pip install -e .  # editable install
```
