from typing import Tuple

import torch
from torch import Tensor

from ..utils import vmap_n

__all__ = ("get_pix_jacobian", "get_pix_magnification", "get_magnification")


def get_pix_jacobian(
    raytrace, thx, thy, z_s
) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """
    Jacobian :math:`\\partial \beta / \\partial \theta` at a single point.

    Note:
        Currently uses `jacrev` due to upcasting bug with `jacfwd` (https://github.com/pytorch/pytorch/issues/90065).
        Should probably switch to `jacfwd` in the future.
    """
    jac = torch.func.jacfwd(raytrace, (0, 1))(thx, thy, z_s)  # type: ignore
    return jac


def get_pix_magnification(raytrace, thx, thy, z_s) -> Tensor:
    """
    Magnification at a single point.
    """
    jac = get_pix_jacobian(raytrace, thx, thy, z_s)
    return 1 / (jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]).abs()


def get_magnification(raytrace, thx, thy, z_s) -> Tensor:
    """
    Magnification over a grid.
    """
    return vmap_n(get_pix_magnification, 2, (None, 0, 0, None))(
        raytrace, thx, thy, z_s
    )
