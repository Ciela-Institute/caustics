from typing import Tuple

import torch
from torch import Tensor

from ..utils import vmap_n

__all__ = ("get_pix_jacobian", "get_pix_magnification", "get_magnification")


def get_pix_jacobian(
    raytrace, thx, thy, z_s, x
) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """Computes the Jacobian matrix of the partial derivatives of the
    image position with respect to the source position
    (:math:`\\partial \beta / \\partial \theta`).  This is done at a
    single point on the lensing plane.

    Args:
        raytrace: A function that maps the lensing plane coordinates to the source plane coordinates.
        thx (Tensor): The x-coordinate on the lensing plane.
        thy (Tensor): The y-coordinate on the lensing plane.
        z_s (Tensor): The redshift of the source.
        x: Additional parameters for the raytrace function.

    Returns:
        The Jacobian matrix of the image position with respect to the source position at the given point.

    """
    jac = torch.func.jacfwd(raytrace, (0, 1))(thx, thy, z_s, x)  # type: ignore
    return jac


def get_pix_magnification(raytrace, thx, thy, z_s, x) -> Tensor:
    """
    Computes the magnification at a single point on the lensing plane. The magnification is derived from the determinant
    of the Jacobian matrix of the image position with respect to the source position.

    Args:
        raytrace: A function that maps the lensing plane coordinates to the source plane coordinates.
        thx (Tensor): The x-coordinate on the lensing plane.
        thy (Tensor): The y-coordinate on the lensing plane.
        z_s (Tensor): The redshift of the source.
        x: Additional parameters for the raytrace function.

    Returns:
        The magnification at the given point on the lensing plane.
    """
    jac = get_pix_jacobian(raytrace, thx, thy, z_s, x)
    return 1 / (jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]).abs()


def get_magnification(raytrace, thx, thy, z_s, x) -> Tensor:
    """
    Computes the magnification over a grid on the lensing plane. This is done by calling `get_pix_magnification` 
    for each point on the grid.

    Args:
        raytrace: A function that maps the lensing plane coordinates to the source plane coordinates.
        thx (Tensor): The x-coordinates on the lensing plane.
        thy (Tensor): The y-coordinates on the lensing plane.
        z_s (Tensor): The redshift of the source.
        x: Additional parameters for the raytrace function.

    Returns:
        A tensor representing the magnification at each point on the grid.
    """
    return vmap_n(get_pix_magnification, 2, (None, 0, 0, None, None))(
        raytrace, thx, thy, z_s, x
    )
