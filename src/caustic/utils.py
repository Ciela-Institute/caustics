from functools import wraps
from math import pi
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from .constants import G_over_c2


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


def transform_scalar_fn(x0, y0, phi=None):
    def decorator(fn):
        @wraps(fn)
        def wrapped(x, y, *args, **kwargs):
            xt = x - x0
            yt = y - y0

            if phi is not None:
                # Apply R(-phi)
                c_phi = phi.cos()
                s_phi = phi.sin()
                # Simultaneous assignment
                xt, yt = xt * c_phi + yt * s_phi, -xt * s_phi + yt * c_phi

            return fn(xt, yt, *args, **kwargs)

        return wrapped

    return decorator


def transform_vector_fn(x0, y0, phi=None):
    def decorator(fn):
        @wraps(fn)
        def wrapped(x, y, *args, **kwargs):
            xt = x - x0
            yt = y - y0

            if phi is not None:
                # Apply R(-phi)
                c_phi = phi.cos()
                s_phi = phi.sin()
                # Simultaneous assignment
                xt, yt = xt * c_phi + yt * s_phi, -xt * s_phi + yt * c_phi
                # Evaluate function
                vx, vy = fn(xt, yt, *args, **kwargs)
                # Apply R(phi) to result
                return vx * c_phi - vy * s_phi, vx * s_phi + vy * c_phi

            # Function is independent of phi
            return fn(xt, yt, *args, **kwargs)

        return wrapped

    return decorator


def to_elliptical(x, y, q):
    """
    Converts to elliptical Cartesian coordinates.
    """
    return x * q.sqrt(), y / q.sqrt()


def get_meshgrid(
    resolution, nx, ny, device=None, dtype=torch.float32
) -> Tuple[Tensor, Tensor]:
    base_grid = torch.linspace(-1, 1, nx, device=device, dtype=dtype)
    xs = base_grid * resolution * (nx - 1) / 2
    ys = base_grid * resolution * (ny - 1) / 2
    return torch.meshgrid([xs, ys], indexing="xy")


def get_Sigma_cr(z_l, z_s, cosmology) -> Tensor:
    """
    Critical lensing density [solMass / Mpc^2]
    """
    d_l = cosmology.angular_diameter_dist(z_l)
    d_s = cosmology.angular_diameter_dist(z_s)
    d_ls = cosmology.angular_diameter_dist_z1z2(z_l, z_s)
    return d_s / d_l / d_ls / (4 * pi * G_over_c2)
