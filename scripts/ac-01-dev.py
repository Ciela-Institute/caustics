from jax.config import config

# Without this the ray-tracing will not be very precise. This must be done before
# any other jax computations.
config.update("jax_enable_x64", True)

from math import pi

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from caustic.lenses.sple import alpha as alpha_sple
from caustic.sources.sersic import brightness as brightness_sersic
from caustic.utils import get_meshgrid

plt.rcParams["image.cmap"] = "inferno"

# For mapping a function over grids
vmap_grid = lambda fn, *args, **kwargs: jax.vmap(
    jax.vmap(fn, *args, **kwargs), *args, **kwargs
)

# Get functions batched over grids for SPLE deflection and Sersic brightness
# TODO: hide this boilerplate in caustic.lenses.sple
in_axes_sple = (0, 0, *(None for _ in range(6)))
in_axes_sersic = (0, 0, *(None for _ in range(7)))
alpha_sple_grid = jax.jit(vmap_grid(alpha_sple, in_axes=in_axes_sple))
brightness_sersic_grid = jax.jit(vmap_grid(brightness_sersic, in_axes=in_axes_sersic))


def simulator(kwargs_sple, kwargs_sersic):
    """
    Simple lensing simulator
    """
    alpha_x, alpha_y = jnp.moveaxis(
        alpha_sple_grid(X, Y, *kwargs_sple.values()), -1, -3
    )
    return brightness_sersic_grid(X - alpha_x, Y - alpha_y, *kwargs_sersic.values())


if __name__ == "__main__":
    # Set up image-plane grid
    n_pix = 128
    X, Y = get_meshgrid(5 / n_pix, n_pix, n_pix)

    # Set some parameters for the SPLE and Sersic
    kwargs_sple = dict(x0=-0.05, y0=0.1, phi=1.0, q=0.75, r_ein=1.5, slope=2.1)
    kwargs_sersic = dict(x0=0.0, y0=0.0, phi=pi / 6, q=0.5, index=2.3, r_e=2.0, I_e=1.0)
    # Run simulator
    image = simulator(kwargs_sple, kwargs_sersic)

    # Save a test plot
    plt.figure(figsize=(8, 8))
    plt.imshow(image, origin="lower")
    plt.gca().axis("off")
    plt.savefig("image.png")
