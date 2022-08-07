from typing import Iterable

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from .utils import Array


class CartesianGrid:
    """
    Linear Multivariate Cartesian Grid interpolation in arbitrary dimensions. Based
    on ``map_coordinates``.
    Notes:
        Translated directly from https://github.com/JohannesBuchner/regulargrid/ to jax.
    """

    def __init__(
        self,
        limits: Iterable[Iterable[float]],
        values: Array,
        order: int = 1,
        mode: str = "constant",
        cval: float = jnp.nan,
    ):
        """
        Initializer.

        Args:
            limits: collection of pairs specifying limits of input variables along
                each dimension of ``values``
            values: values to interpolate. These must be defined on a regular grid.
            mode: how to handle out of bounds arguments; see docs for ``map_coordinates``
            cval: constant fill value; see docs for ``map_coordinates``
        """
        super().__init__()
        self.values = values
        self.limits = limits
        self.order = order
        self.mode = mode
        self.cval = cval

    def __call__(self, *coords) -> Array:
        """
        Perform interpolation.

        Args:
            coords: point at which to interpolate. These will be broadcasted if
                they are not the same shape.
        Returns:
            Interpolated values, with extrapolation handled according to ``mode``.
        """
        # transform coords into pixel values
        coords = jnp.broadcast_arrays(*coords)
        # coords = jnp.asarray(coords)
        coords = [
            (c - lo) * (n - 1) / (hi - lo)
            for (lo, hi), c, n in zip(self.limits, coords, self.values.shape)
        ]
        return map_coordinates(
            self.values, coords, mode=self.mode, cval=self.cval, order=self.order
        )
