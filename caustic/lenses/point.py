from typing import Any, Optional, Union

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens
from ..parametrized import unpack

__all__ = ("Point",)

class Point(ThinLens):
    """
    Class representing a point mass lens in strong gravitational lensing.

    Attributes:
        name (str): The name of the point lens.
        cosmology (Cosmology): The cosmology used for calculations.
        z_l (Optional[Union[Tensor, float]]): Redshift of the lens.
        x0 (Optional[Union[Tensor, float]]): x-coordinate of the center of the lens.
        y0 (Optional[Union[Tensor, float]]): y-coordinate of the center of the lens.
        th_ein (Optional[Union[Tensor, float]]): Einstein radius of the lens.
        s (float): Softening parameter to prevent numerical instabilities.
    """
    def __init__(
        self,
        cosmology: Cosmology,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        th_ein: Optional[Union[Tensor, float]] = None,
        s: float = 0.0,
        name: str = None,
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
        Compute the deflection angles.

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            tuple[Tensor, Tensor]: The deflection angles in the x and y directions.
        """
        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        ax = x / th**2 * th_ein**2
        ay = y / th**2 * th_ein**2
        return ax, ay

    @unpack(3)
    def potential(
        self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, th_ein, *args, params: Optional["Packed"] = None, **kwargs
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
        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        return th_ein**2 * th.log()

    @unpack(3)
    def convergence(
        self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, th_ein, *args, params: Optional["Packed"] = None, **kwargs
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
        x, y = translate_rotate(x, y, x0, y0)
        return torch.where((x == 0) & (y == 0), torch.inf, 0.0)
