import jax.numpy as jnp

from ..utils import flip_axis_ratio, to_elliptical, translate_rotate


def pot(x, y, x0, y0, r_ein, q, phi, x_c=0.0):
    q, phi = flip_axis_ratio(q, phi)
    x, y = translate_rotate(x, y, -phi, x0, y0)
    x, y = to_elliptical(x, y, q)
    X = jnp.sqrt(x**2 + y**2 + x_c**2)
    return r_ein * X
