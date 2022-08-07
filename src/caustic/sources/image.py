import jax.numpy as jnp
from ..cartesiangrid import CartesianGrid


def brightness(
    x,
    y,
    x_bounds,
    y_bounds,
    values,
    x0=0.0,
    y0=0.0,
    scale=1.0,
    order: int = 1,
    mode: str = "constant",
    cval: float = jnp.nan,
):
    x = (x - x0) / scale
    y = (y - y0) / scale

    # Transpose since rows correspond to x and columns correspond to y
    interp = CartesianGrid(
        (y_bounds, x_bounds),
        values,
        order,
        mode,
        cval,
    )
    return interp(y, x)
