import jax.numpy as jnp
from ..utils import flip_axis_ratio, translate_rotate, to_elliptical


def brightness(x, y, x0, y0, phi, q, index, r_e, I_e):
    q, phi = flip_axis_ratio(q, phi)
    x, y = translate_rotate(x, y, -phi, x0, y0)
    x, y = to_elliptical(x, y, q)
    X = jnp.sqrt(x**2 + y**2)
    r = X / r_e

    k = 2 * index - 1 / 3 + 4 / 405 / index + 46 / 25515 / index**2
    exponent = -k * (r ** (1 / index) - 1)

    return I_e * jnp.exp(exponent)
