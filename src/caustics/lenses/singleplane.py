from warnings import warn

import torch
from torch import Tensor
from caskade import forward

from .base import ThinLens, CosmologyType, NameType, LensesType, ZType

__all__ = ("SinglePlane",)


class SinglePlane(ThinLens):
    """
    A class for combining multiple thin lenses into a single lensing plane.
    This model inherits from the base `ThinLens` class.

    Attributes
    ----------
    name: str
        The name of the single plane lens.

    cosmology: Cosmology
        An instance of the Cosmology class.

    lenses: List[ThinLens]
        A list of ThinLens objects that are being combined into a single lensing plane.

    """

    def __init__(
        self,
        cosmology: CosmologyType,
        lenses: LensesType,
        name: NameType = None,
        z_l: ZType = None,
        z_s: ZType = None,
    ):
        """
        Initialize the SinglePlane lens model.
        """
        super().__init__(cosmology, z_l=z_l, name=name, z_s=z_s)
        self.lenses = lenses

        for lens in self.lenses:
            if lens.z_l.static:
                warn(
                    f"Lens model {lens.name} has a static lens redshift. This is now overwritten by the SinglePlane ({self.name}) lens redshift. To prevent this warning, set the lens redshift of the lens model to be dynamic before adding to the system."
                )
            lens.z_l = self.z_l

            if lens.z_s.static:
                warn(
                    f"Lens model {lens.name} has a static source redshift. This is now overwritten by the SinglePlane ({self.name}) source redshift. To prevent this warning, set the source redshift of the lens model to be dynamic before adding to the system."
                )
            lens.z_s = self.z_s

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the total deflection angle by summing
        the deflection angles of all individual lenses.

        Parameters
        ----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        x_component: Tensor
            The x-component of the deflection angle.

            *Unit: arcsec*

        y_component: Tensor
            The y-component of the deflection angle.

            *Unit: arcsec*

        """
        ax = torch.zeros_like(x)
        ay = torch.zeros_like(x)
        for lens in self.lenses:
            ax_cur, ay_cur = lens.reduced_deflection_angle(x, y)
            ax = ax + ax_cur
            ay = ay + ay_cur
        return ax, ay

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Calculate the total projected mass density by
        summing the mass densities of all individual lenses.

        Parameters
        ----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The total projected mass density.

            *Unit: unitless*

        """
        convergence = torch.zeros_like(x)
        for lens in self.lenses:
            convergence_cur = lens.convergence(x, y)
            convergence = convergence + convergence_cur
        return convergence

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Compute the total lensing potential by summing
        the lensing potentials of all individual lenses.

        Parameters
        -----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The total lensing potential.

            *Unit: arcsec^2*

        """
        potential = torch.zeros_like(x)
        for lens in self.lenses:
            potential_cur = lens.potential(x, y)
            potential = potential + potential_cur
        return potential
