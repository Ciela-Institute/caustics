from functools import wraps
from math import pi
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def flip_axis_ratio(q, phi):
    """
    Makes q positive, then swaps x and y axes if it's larger than 1.
    """
    q = q.abs()
    return torch.where(q > 1, 1 / q, q), torch.where(q > 1, phi + pi / 2, phi)


def translate_rotate(
    x: Tensor,
    y: Tensor,
    phi: Optional[Tensor] = None,
    x_0: Union[float, Tensor] = 0.0,
    y_0: Union[float, Tensor] = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    Translates and applies an ''active'' counterclockwise rotation to the input
    point.
    """
    x = x - x_0
    y = y - y_0

    if phi is not None:
        c_phi = phi.cos()
        s_phi = phi.sin()
        return x * c_phi - y * s_phi, x * s_phi + y * c_phi
    else:
        return x, y


def transform_scalar_fn(fn, x_name="thx0", y_name="thy0", phi_name="phi"):
    @wraps(fn)
    def wrapped(self, x, y, *args, **kwargs):
        xt = x - getattr(self, x_name)
        yt = y - getattr(self, y_name)
        if hasattr(self, phi_name):
            # Apply R(-phi)
            phi = getattr(self, phi_name)
            c_phi = phi.cos()
            s_phi = phi.sin()
            xt = x * c_phi + y * s_phi
            yt = -x * s_phi + y * c_phi
            # Evaluate function

        return fn(self, xt, yt, *args, **kwargs)

    return wrapped


def transform_vector_fn(fn, x_name="thx0", y_name="thy0", phi_name="phi"):
    @wraps(fn)
    def wrapped(self, x, y, *args, **kwargs):
        xt = x - getattr(self, x_name)
        yt = y - getattr(self, y_name)
        if hasattr(self, phi_name):
            # Apply R(-phi)
            phi = getattr(self, phi_name)
            c_phi = phi.cos()
            s_phi = phi.sin()
            xt = x * c_phi + y * s_phi
            yt = -x * s_phi + y * c_phi
            # Evaluate function
            vx, vy = fn(self, xt, yt, *args, **kwargs)
            # Apply R(phi)
            return vx * c_phi - vy * s_phi, vx * s_phi + vy * c_phi
        else:
            # Function is independent of phi
            return fn(self, xt, yt, *args, **kwargs)

    return wrapped


def to_elliptical(x, y, q):
    """
    Converts to elliptical Cartesian coordinates.
    """
    return x * q, y


def get_meshgrid(resolution, nx, ny, device=None) -> Tuple[Tensor, Tensor]:
    xs = torch.linspace(-1, 1, nx, device=device) * resolution * (nx - 1) / 2
    ys = torch.linspace(-1, 1, ny, device=device) * resolution * (ny - 1) / 2
    return torch.meshgrid([xs, ys], indexing="xy")
