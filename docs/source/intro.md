# Welcome to Caustics’ documentation!

The lensing pipeline of the future: GPU-accelerated,
automatically-differentiable, highly modular and extensible.

```{note}
Caustics is in its early development phase. This means the API will change with time. These changes are a good thing, but they can be annoying. Watch the version numbers, when we get to 1.0.0 that will be the first stable release!
```

## Installation

The easiest way to install is to make a new virtual environment then run:

```console
pip install caustics
```

this will install all the required libraries and then install caustics and you
are ready to go! You can check out the tutorials afterwards to see some of
caustics' capabilities. If you want to help out with building the caustics code
base check out the developer installation instructions instead.

## Minimal Example

```python
import matplotlib.pyplot as plt
import caustics

C = caustics.cosmology.FlatLambdaCDM()

sie = caustics.lenses.SIE(cosmology=C, z_l=0.5, x0=0., y0=0., q=0.4, phi=1.5708, b=1.)
src = caustics.light.Sersic(x0=-0.2, y0=0.0, q=0.6, phi=-0.785, n=1.5, Re=3.0, Ie=1.0)

sim = caustics.sims.Lens_Source(lens=sie, source=src, pixelscale=0.05, pixels_x=100, z_s=1.5)

plt.imshow(sim(quad_level=3).detach().cpu().numpy(), origin="lower")
plt.axis("off")
plt.show()
```

![Caustics lensed image](../../media/minimal_example.png)

### Contents

```{tableofcontents}

```
