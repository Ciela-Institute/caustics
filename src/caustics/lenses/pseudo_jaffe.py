# mypy: disable-error-code="operator,dict-item"
from math import pi
from typing import Optional, Union, Annotated

import torch
from torch import Tensor
from caskade import forward, Param

from ..constants import arcsec_to_rad
from .base import ThinLens, CosmologyType, NameType, ZLType
from . import func

__all__ = ("PseudoJaffe",)


class PseudoJaffe(ThinLens):
    """
    Class representing a Pseudo Jaffe lens in strong gravitational lensing,
    based on `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_ and
    the `lenstronomy` source code.

    Attributes
    ----------
    name: string
        The name of the Pseudo Jaffe lens.
    cosmology: Cosmology
        The cosmology used for calculations.
    z_l: Optional[Union[Tensor, float]]
        Redshift of the lens.

        *Unit: unitless*

    x0: Optional[Union[Tensor, float]]
        x-coordinate of the center of the lens (arcsec).

        *Unit: arcsec*

    y0: Optional[Union[Tensor, float]]
        y-coordinate of the center of the lens (arcsec).

        *Unit: arcsec*

    mass: Optional[Union[Tensor, float]]
        Total mass of the lens (Msun).

        *Unit: Msun*

    Rc: Optional[Union[Tensor, float]]
        Core radius of the lens (arcsec).

        *Unit: arcsec*

    Rs: Optional[Union[Tensor, float]]
        Scaling radius of the lens (arcsec).

        *Unit: arcsec*

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*

    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "mass": 1e12,
        "Rc": 0.1,
        "Rs": 1.0,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZLType = None,
        x0: Annotated[
            Optional[Union[Tensor, float]],
            "X coordinate of the center of the lens",
            True,
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]],
            "Y coordinate of the center of the lens",
            True,
        ] = None,
        mass: Annotated[
            Optional[Union[Tensor, float]], "Total mass of the lens", True, "Msol"
        ] = None,
        Rc: Annotated[
            Optional[Union[Tensor, float]], "Core radius of the lens", True, "arcsec"
        ] = None,
        Rs: Annotated[
            Optional[Union[Tensor, float]],
            "Scaling radius of the lens",
            True,
            "arcsec",
        ] = None,
        s: Annotated[
            float, "Softening parameter to prevent numerical instabilities"
        ] = 0.0,
        name: NameType = None,
    ):
        """
        Initialize the PseudoJaffe class.

        Parameters
        ----------
        name: string
            The name of the Pseudo Jaffe lens.

        cosmology: Cosmology
            The cosmology used for calculations.

        z_l: Optional[Tensor]
            Redshift of the lens.

            *Unit: unitless*

        x0: Optional[Tensor]
            x-coordinate of the center of the lens.

            *Unit: arcsec*

        y0: Optional[Tensor]
            y-coordinate of the center of the lens.

            *Unit: arcsec*

        mass: Optional[Tensor]
            Total mass of the lens (Msun).

            *Unit: Msun*

        Rc: Optional[Tensor]
            Core radius of the lens.

            *Unit: arcsec*

        Rs: Optional[Tensor]
            Scaling radius of the lens.

            *Unit: arcsec*

        s: float
            Softening parameter to prevent numerical instabilities.

            *Unit: arcsec*

        """
        super().__init__(cosmology, z_l, name=name)

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.mass = Param("mass", mass, units="Msun", valid=(0, None))
        self.Rc = Param("Rc", Rc, units="arcsec", valid=(0, None))
        self.Rs = Param("Rs", Rs, units="arcsec", valid=(0, None))
        self.s = s

    @forward
    def get_convergence_0(
        self,
        z_s,
        z_l: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        Rc: Annotated[Tensor, "Param"],
        Rs: Annotated[Tensor, "Param"],
    ):
        d_l = self.cosmology.angular_diameter_distance(z_l)
        sigma_crit = self.cosmology.critical_surface_density(z_l, z_s)
        return mass / (2 * torch.pi * sigma_crit * Rc * Rs * (d_l * arcsec_to_rad) ** 2)  # fmt: skip

    @forward
    def mass_enclosed_2d(
        self,
        theta,
        z_s,
        mass: Annotated[Tensor, "Param"],
        Rc: Annotated[Tensor, "Param"],
        Rs: Annotated[Tensor, "Param"],
    ):
        """
        Calculate the mass enclosed within a two-dimensional radius. Using equation A10 from `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_.

        Parameters
        ----------
        theta: Tensor
            Radius at which to calculate enclosed mass (arcsec).

            *Unit: arcsec*

        z_s: Tensor
            Source redshift.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The mass enclosed within the given radius.

            *Unit: Msun*

        """
        return func.mass_enclosed_2d_pseudo_jaffe(theta, mass, Rc, Rs)

    @staticmethod
    def central_convergence(
        z_l,
        z_s,
        rho_0,
        Rc,
        Rs,
        critical_surface_density,
    ):
        """
        Compute the central convergence.

        Parameters
        -----------
        z_l: Tensor
            Lens redshift.

            *Unit: unitless*

        z_s: Tensor
            Source redshift.

            *Unit: unitless*

        rho_0: Tensor
            Central mass density.

            *Unit: Msun/Mpc^3*

        Rc: Tensor
            Core radius of the lens (must be in Mpc).

            *Unit: Mpc*

        Rs: Tensor
            Scaling radius of the lens (must be in Mpc).

            *Unit: Mpc*

        cosmology: Cosmology
            The cosmology used for calculations.

        Returns
        --------
        Tensor
            The central convergence.

            *Unit: unitless*

        """
        return pi * rho_0 * Rc * Rs / ((Rc + Rs) * critical_surface_density)  # fmt: skip

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        z_l: Annotated[Tensor, "Param"],
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        Rc: Annotated[Tensor, "Param"],
        Rs: Annotated[Tensor, "Param"],
    ) -> tuple[Tensor, Tensor]:
        """Calculate the deflection angle.

        Parameters
        ----------
        x: Tensor
            x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            Source redshift.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        --------
        x_component: Tensor
            x-component of the deflection angle.

            *Unit: arcsec*

        y_component: Tensor
            y-component of the deflection angle.

            *Unit: arcsec*

        """
        d_l = self.cosmology.angular_diameter_distance(z_l)
        critical_surface_density = self.cosmology.critical_surface_density(z_l, z_s)
        return func.reduced_deflection_angle_pseudo_jaffe(x0, y0, mass, Rc, Rs, x, y, d_l, critical_surface_density)  # fmt: skip

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        z_l: Annotated[Tensor, "Param"],
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        Rc: Annotated[Tensor, "Param"],
        Rs: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Compute the lensing potential. This calculation is based on equation A18 from `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_.

        Parameters
        --------
        x: Tensor
            x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            Source redshift.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        --------
        Tensor
            The lensing potential (arcsec^2).

            *Unit: arcsec^2*

        """

        d_l = self.cosmology.angular_diameter_distance(z_l)  # Mpc
        d_s = self.cosmology.angular_diameter_distance(z_s)  # Mpc
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)  # Mpc

        return func.potential_pseudo_jaffe(x0, y0, mass, Rc, Rs, x, y, d_l, d_s, d_ls)  # fmt: skip

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        z_l: Annotated[Tensor, "Param"],
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        Rc: Annotated[Tensor, "Param"],
        Rs: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Calculate the projected mass density, based on equation A6.

        Parameters
        -----------
        x: Tensor
            x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            Source redshift.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The projected mass density.

            *Unit: unitless*

        """
        d_l = self.cosmology.angular_diameter_distance(z_l)
        critical_surface_density = self.cosmology.critical_surface_density(z_l, z_s)
        return func.convergence_pseudo_jaffe(
            x0, y0, mass, Rc, Rs, x, y, d_l, critical_surface_density
        )
