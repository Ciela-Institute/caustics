from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

__all__ = ("Point",)

class Point(ThinLens):
    """
    Class representing a point mass lens in strong gravitational lensing.

    Attributes:
        name (str): The name of the point lens.
        cosmology (Cosmology): The cosmology used for calculations.
        z_l (Optional[Tensor]): Redshift of the lens.
        x0 (Optional[Tensor]): x-coordinate of the center of the lens.
        y0 (Optional[Tensor]): y-coordinate of the center of the lens.
        th_ein (Optional[Tensor]): Einstein radius of the lens.
        s (float): Softening parameter to prevent numerical instabilities.
    """
    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        th_ein: Optional[Tensor] = None,
        s: float = 0.0,
    ):
        """
        Initialize the Point class.

        Args:
            name (str): The name of the point lens.
            cosmology (Cosmology): The cosmology used for calculations.
            z_l (Optional[Tensor]): Redshift of the lens.
            x0 (Optional[Tensor]): x-coordinate of the center of the lens.
            y0 (Optional[Tensor]): y-coordinate of the center of the lens.
            th_ein (Optional[Tensor]): Einstein radius of the lens.
            s (float): Softening parameter to prevent numerical instabilities.
        """
        super().__init__(name, cosmology, z_l)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("th_ein", th_ein)
        self.s = s

    def reduced_deflection_angle(
        self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the deflection angles.

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            tuple[Tensor, Tensor]: The deflection angles in the x and y directions.
        """
        z_l, x0, y0, th_ein = self.unpack(params)

        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        ax = x / th**2 * th_ein**2
        ay = y / th**2 * th_ein**2
        return ax, ay

    def potential(
        self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the lensing potential.

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The lensing potential.
        """
        z_l, x0, y0, th_ein = self.unpack(params)

        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        return th_ein**2 * th.log()

    def convergence(
        self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the convergence (dimensionless surface mass density).

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The convergence (dimensionless surface mass density).
        """
        z_l, x0, y0, th_ein = self.unpack(params)

        x, y = translate_rotate(x, y, x0, y0)
        return torch.where((x == 0) & (y == 0), torch.inf, 0.0)
