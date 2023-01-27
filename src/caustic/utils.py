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


def translate_rotate(x, y, x0, y0, phi: Optional[Tensor] = None):
    xt = x - x0
    yt = y - y0

    if phi is not None:
        # Apply R(-phi)
        c_phi = phi.cos()
        s_phi = phi.sin()
        # Simultaneous assignment
        xt, yt = xt * c_phi + yt * s_phi, -xt * s_phi + yt * c_phi

    return xt, yt


def derotate(vx, vy, phi: Optional[Tensor] = None):
    if phi is None:
        return vx, vy

    c_phi = phi.cos()
    s_phi = phi.sin()
    return vx * c_phi - vy * s_phi, vx * s_phi + vy * c_phi


def to_elliptical(x, y, q: Tensor):
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


def safe_divide(num, denom):
    """
    Differentiable version of `torch.where(denom != 0, num/denom, 0.0)`.

    Returns:
        `num / denom` where `denom != 0`; zero everywhere else.
    """
    out = torch.zeros_like(num)
    where = denom != 0
    out[where] = num[where] / denom[where]
    return out
