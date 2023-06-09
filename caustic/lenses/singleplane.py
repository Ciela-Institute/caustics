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
        # TODO: assert all z_l are the same?

    def reduced_deflection_angle(
        self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the total deflection angle by summing the deflection angles of all individual lenses.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tuple[Tensor, Tensor]: The total deflection angle in the x and y directions.
        """
        ax = torch.zeros_like(x)
        ay = torch.zeros_like(x)
        for lens in self.lenses:
            ax_cur, ay_cur = lens.reduced_deflection_angle(x, y, z_s, params)
            ax = ax + ax_cur
            ay = ay + ay_cur
        return ax, ay

    def convergence(
        self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Calculate the total projected mass density by summing the mass densities of all individual lenses.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The total projected mass density.
        """
        convergence = torch.zeros_like(x)
        for lens in self.lenses:
            convergence_cur = lens.convergence(x, y, z_s, params)
            convergence = convergence + convergence_cur
        return convergence

    def potential(
        self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the total lensing potential by summing the lensing potentials of all individual lenses.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The total lensing potential.
        """
        potential = torch.zeros_like(x)
        for lens in self.lenses:
            potential_cur = lens.potential(x, y, z_s, params)
            potential = potential + potential_cur
        return potential
