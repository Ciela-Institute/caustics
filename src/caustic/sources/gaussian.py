import jax.numpy as jnp


def brightness(x, y, x0, y0, sigma, norm):
    d2 = (x - x0) ** 2 + (y - y0) ** 2
    return norm * jnp.exp(-d2 / (2 * sigma**2)) / (2 * jnp.pi * sigma**2)
