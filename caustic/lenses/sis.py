from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

__all__ = ("SIS",)


class SIS(ThinLens):
    """
    A class representing the Singular Isothermal Sphere (SIS) model. 
    This model inherits from the base `ThinLens` class.

    Attributes:
        name (str): The name of the SIS lens.
        cosmology (Cosmology): An instance of the Cosmology class.
        z_l (Optional[Tensor]): The lens redshift.
        x0 (Optional[Tensor]): The x-coordinate of the lens center.
        y0 (Optional[Tensor]): The y-coordinate of the lens center.
        th_ein (Optional[Tensor]): The Einstein radius of the lens.
        s (float): A smoothing factor, default is 0.0.
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
        Initialize the SIS lens model.
        """
        super().__init__(name, cosmology, z_l)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("th_ein", th_ein)
        self.s = s

    def deflection_angle(
        self, x: Tensor, y: Tensor, z_s: Tensor, P: "Packed" = None
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the deflection angle of the SIS lens.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            x (Packed): Additional parameters.

        Returns:
            Tuple[Tensor, Tensor]: The deflection angle in the x and y directions.
        """
        z_l, x0, y0, th_ein = self.unpack(P)

        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        ax = th_ein * x / th
        ay = th_ein * y / th
        return ax, ay

    def potential(
        self, x: Tensor, y: Tensor, z_s: Tensor, P: "Packed" = None
    ) -> Tensor:
        """
        Compute the lensing potential of the SIS lens.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            x (Packed): Additional parameters.

        Returns:
            Tensor: The lensing potential.
        """
        z_l, x0, y0, th_ein = self.unpack(P)

        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        return th_ein * th

    def convergence(
        self, x: Tensor, y: Tensor, z_s: Tensor, P: "Packed" = None
    ) -> Tensor:
        """
        Calculate the projected mass density of the SIS lens.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            P (Packed): Additional parameters.

        Returns:
            Tensor: The projected mass density.
        """
        z_l, x0, y0, th_ein = self.unpack(P)

        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        return 0.5 * th_ein / th
