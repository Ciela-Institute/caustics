import jax
import jax.numpy as jnp

from ..utils import flip_axis_ratio, to_elliptical, translate_rotate


N_ITER = 18  # taken from gigalens


def kappa(x, y, x0, y0, r_ein, q, phi, gamma):
    q, phi = flip_axis_ratio(q, phi)
    x, y = translate_rotate(x, y, -phi, x0, y0)
    x, y = to_elliptical(x, y, q)
    X = jnp.sqrt(x**2 + y**2)
    return (3 - gamma) / 2 * (r_ein / X) ** (gamma - 1)


def alpha(x, y, x0, y0, r_ein, q, phi, gamma):
    """
    Notes:
        Inspired by the `gigalens` EPL class implementation.
    """
    q, phi = flip_axis_ratio(q, phi)
    x, y = translate_rotate(x, y, -phi, x0, y0)
    x, y = to_elliptical(x, y, q)
    X = jnp.sqrt(x**2 + y**2)

    # Trig functions of elliptical polar angle, *not* position angle
    c_ph = x / X
    s_ph = y / X
    c_2ph = (x - y) * (x + y) / X**2
    s_2ph = 2 * x * y / X**2

    f = (1 - q) / (1 + q)
    t = gamma - 1

    # Sec. 4 of Tessore & Metcalf (2015)
    # TODO: not sure fori_loop is required here!
    def body_fun(n, val):
        om_x_prev, om_y_prev, Om_x_prev, Om_y_prev = val
        fact = -f * (2 * n - (2 - t)) / (2 * n + (2 - t))
        Om_x = fact * (Om_x_prev * c_2ph - Om_y_prev * s_2ph)
        Om_y = fact * (Om_x_prev * s_2ph + Om_y_prev * c_2ph)
        om_x = om_x_prev + Om_x
        om_y = om_y_prev + Om_y
        return (om_x, om_y, Om_x, Om_y)

    om_x, om_y, *_ = jax.lax.fori_loop(1, N_ITER, body_fun, (c_ph, s_ph, c_ph, s_ph))

    alpha_fact = r_ein * 2 * jnp.sqrt(q) / (1 + q) * (r_ein / X) ** (gamma - 2)
    return alpha_fact * translate_rotate(om_x, om_y, phi)


def pot(x, y, x0, y0, r_ein, q, phi, gamma):
    ax, ay = alpha(x, y, x0, y0, r_ein, q, phi, gamma)
    x = x - x0
    y = y - y0
    return (x * ax + y * ay) / (3 - gamma)
