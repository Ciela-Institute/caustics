from math import pi

import jax
import jax.numpy as jnp
from astropy.constants.codata2018 import G, c

Array = jnp.ndarray


G_OVER_C2 = float((G / c**2).to("Mpc/solMass").value)
C_MPC_S = float(c.to("Mpc/s").value)
KM_TO_MPC = 3.2407792896664e-20


def get_meshgrid(res: float, nx: int, ny: int) -> Array:
    """
    Gets meshgrid of pixel center coordinates.
    """
    dx = res
    dy = res
    x = jnp.linspace(-1, 1, nx) * (nx - 1) * dx / 2
    y = jnp.linspace(-1, 1, ny) * (ny - 1) * dy / 2
    return jnp.meshgrid(x, y)


def downsample(arr: Array, factor: int) -> Array:
    """
    Downsamples an array by averaging over blocks. Batched.

    Notes:
        See https://stackoverflow.com/a/60345995.

    Args:
        arr: the array to downsample
        factor: the factor by which to downsample. The sizes of the last
            two dimensions of ``arr`` must be divisible by ``factor``.

    Returns:
        The downsampled array.
    """
    nr, nc = arr.shape
    assert nr % factor == 0
    assert nc % factor == 0
    return (
        arr.reshape(
            *arr.shape[:-2],
            arr.shape[-2] // factor,
            factor,
            arr.shape[-1] // factor,
            factor
        )
        .mean(-3)
        .mean(-1)
    )


def flip_axis_ratio(q, phi):
    """
    Makes q positive, then swaps x and y axes if it's larger than 1.
    """
    q = jnp.abs(q)
    return jax.lax.cond(q > 1, lambda: (1 / q, phi + pi / 2), lambda: (q, phi))


def translate_rotate(x, y, phi, x_0=0.0, y_0=0.0):
    """
    Translates and applies an ''active'' counterclockwise rotation to the input
    point.
    """
    dx = x - x_0
    dy = y - y_0
    c_phi = jnp.cos(phi)
    s_phi = jnp.sin(phi)
    return jnp.array((dx * c_phi - dy * s_phi, dx * s_phi + dy * c_phi))


def to_elliptical(x, y, q):
    """
    Converts to elliptical Cartesian coordinates.
    """
    return (x * jnp.sqrt(q), y / jnp.sqrt(q))
