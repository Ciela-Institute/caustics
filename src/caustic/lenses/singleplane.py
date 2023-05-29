from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from .base import ThinLens

__all__ = ("SinglePlane",)


class SinglePlane(ThinLens):
    """
    A class for combining multiple thin lenses into a single lensing plane. 
    This model inherits from the base `ThinLens` class.
    
    Attributes:
        name (str): The name of the single plane lens.
        cosmology (Cosmology): An instance of the Cosmology class.
        lenses (List[ThinLens]): A list of ThinLens objects that are being combined into a single lensing plane.
    """

    def __init__(self, name: str, cosmology: Cosmology, lenses: list[ThinLens]):
        """
        Initialize the SinglePlane lens model.
        """
        super().__init__(name, cosmology)
        self.lenses = lenses
        for lens in lenses:
            self.add_parametrized(lens)

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the total deflection angle by summing the deflection angles of all individual lenses.

        Args:
            thx (Tensor): The x-coordinate of the lens.
            thy (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tuple[Tensor, Tensor]: The total deflection angle in the x and y directions.
        """
        ax = torch.zeros_like(thx)
        ay = torch.zeros_like(thx)
        for lens in self.lenses:
            ax_cur, ay_cur = lens.alpha(thx, thy, z_s, x)
            ax = ax + ax_cur
            ay = ay + ay_cur
        return ax, ay

    def kappa(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Calculate the total projected mass density by summing the mass densities of all individual lenses.

        Args:
            thx (Tensor): The x-coordinate of the lens.
            thy (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tensor: The total projected mass density.
        """
        kappa = torch.zeros_like(thx)
        for lens in self.lenses:
            kappa_cur = lens.kappa(thx, thy, z_s, x)
            kappa = kappa + kappa_cur
        return kappa

    def Psi(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Compute the total lensing potential by summing the lensing potentials of all individual lenses.

        Args:
            thx (Tensor): The x-coordinate of the lens.
            thy (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tensor: The total lensing potential.
        """
        Psi = torch.zeros_like(thx)
        for lens in self.lenses:
            Psi_cur = lens.Psi(thx, thy, z_s, x)
            Psi = Psi + Psi_cur
        return Psi
