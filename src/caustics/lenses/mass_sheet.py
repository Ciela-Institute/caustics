from typing import Optional, Union

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens
from ..parametrized import unpack

__all__ = ("MassSheet",)


class MassSheet(ThinLens):
    """
    Represents an external shear effect in a gravitational lensing system.

    Attributes:
        name (str): Identifier for the lens instance.
        cosmology (Cosmology): The cosmological model used for lensing calculations.
        z_l (Optional[Union[Tensor, float]]): The redshift of the lens.
        x0, y0 (Optional[Union[Tensor, float]]): Coordinates of the shear center in the lens plane.
        gamma_1, gamma_2 (Optional[Union[Tensor, float]]): Shear components.

    Note: The shear components gamma_1 and gamma_2 represent an external shear, a gravitational
    distortion that can be caused by nearby structures outside of the main lens galaxy.
    """

    def __init__(
        self,
        cosmology: Cosmology,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        surface_density: Optional[Union[Tensor, float]] = None,
        name: str = None,
    ):
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("surface_density", surface_density)

    @unpack(3)
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        z_l,
        x0,
        y0,
        surface_density,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculates the reduced deflection angle.

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            tuple[Tensor, Tensor]: The reduced deflection angles in the x and y directions.
        """
        x, y = translate_rotate(x, y, x0, y0)
        # Meneghetti eq 3.84
        ax = x * surface_density
        ay = y * surface_density
        return ax, ay

    @unpack(3)
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        z_l,
        x0,
        y0,
        surface_density,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        # Meneghetti eq 3.81
        return surface_density * 0.5 * (x**2 + y**2)

    @unpack(3)
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        z_l,
        x0,
        y0,
        surface_density,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        # Essentially by definition
        return surface_density * torch.ones_like(x)
