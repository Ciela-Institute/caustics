from typing import Any, Optional, Union

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens
from ..parametrized import unpack

__all__ = ("SIS",)


class SIS(ThinLens):
    """
    A class representing the Singular Isothermal Sphere (SIS) model. 
    This model inherits from the base `ThinLens` class.

    Attributes:
        name (str): The name of the SIS lens.
        cosmology (Cosmology): An instance of the Cosmology class.
        z_l (Optional[Union[Tensor, float]]): The lens redshift.
        x0 (Optional[Union[Tensor, float]]): The x-coordinate of the lens center.
        y0 (Optional[Union[Tensor, float]]): The y-coordinate of the lens center.
        th_ein (Optional[Union[Tensor, float]]): The Einstein radius of the lens.
        s (float): A smoothing factor, default is 0.0.
    """
    def __init__(
        self,
        cosmology: Cosmology,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        th_ein: Optional[Union[Tensor, float]] = None,
        s: float = 0.0,
        name: str = None
    ):
        """
        Initialize the SIS lens model.
        """
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("th_ein", th_ein)
        self.s = s

    @unpack(3)
    def reduced_deflection_angle(
            self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, th_ein, *args, params: Optional["Packed"] = None, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the deflection angle of the SIS lens.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tuple[Tensor, Tensor]: The deflection angle in the x and y directions.
        """
        x, y = translate_rotate(x, y, x0, y0)
        R = (x**2 + y**2).sqrt() + self.s
        ax = th_ein * x / R
        ay = th_ein * y / R
        return ax, ay

    @unpack(3)
    def potential(
        self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, th_ein, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the lensing potential of the SIS lens.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The lensing potential.
        """
        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        return th_ein * th

    @unpack(3)
    def convergence(
        self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, th_ein, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Calculate the projected mass density of the SIS lens.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The projected mass density.
        """
        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        return 0.5 * th_ein / th
