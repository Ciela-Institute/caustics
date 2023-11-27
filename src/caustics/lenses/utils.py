from typing import Tuple

import torch
from torch import Tensor

from ..utils import vmap_n

__all__ = ("get_pix_jacobian", "get_pix_magnification", "get_magnification")


def get_pix_jacobian(
    raytrace, x, y, z_s
) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """Computes the Jacobian matrix of the partial derivatives of the
    image position with respect to the source position
    (:math:`\\partial \beta / \\partial \theta`).  This is done at a
    single point on the lensing plane.

    Args:
        raytrace: A function that maps the lensing plane coordinates to the source plane coordinates.
        x (Tensor): The x-coordinate on the lensing plane.
        y (Tensor): The y-coordinate on the lensing plane.
        z_s (Tensor): The redshift of the source.

    Returns:
        The Jacobian matrix of the image position with respect to the source position at the given point.

    """
    jac = torch.func.jacfwd(raytrace, (0, 1))(x, y, z_s)  # type: ignore
    return jac


def get_pix_magnification(raytrace, x, y, z_s) -> Tensor:
    """
    Computes the magnification at a single point on the lensing plane. The magnification is derived from the determinant
    of the Jacobian matrix of the image position with respect to the source position.

    Args:
        raytrace: A function that maps the lensing plane coordinates to the source plane coordinates.
        x (Tensor): The x-coordinate on the lensing plane.
        y (Tensor): The y-coordinate on the lensing plane.
        z_s (Tensor): The redshift of the source.

    Returns:
        The magnification at the given point on the lensing plane.
    """
    jac = get_pix_jacobian(raytrace, x, y, z_s)
    return 1 / (jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]).abs()


def get_magnification(raytrace, x, y, z_s) -> Tensor:
    """
    Computes the magnification over a grid on the lensing plane. This is done by calling `get_pix_magnification`
    for each point on the grid.

    Args:
        raytrace: A function that maps the lensing plane coordinates to the source plane coordinates.
        x (Tensor): The x-coordinates on the lensing plane.
        y (Tensor): The y-coordinates on the lensing plane.
        z_s (Tensor): The redshift of the source.

    Returns:
        A tensor representing the magnification at each point on the grid.
    """
    return vmap_n(get_pix_magnification, 2, (None, 0, 0, None))(raytrace, x, y, z_s)
