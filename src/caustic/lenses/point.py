import jax.numpy as jnp


def alpha(x, y, x0, y0, r_ein):
    x = x - x0
    y = y - y0
    p = jnp.array((x, y))
    X = jnp.sqrt(x**2 + y**2)
    return r_ein * p / X
