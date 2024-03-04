# mypy: disable-error-code="operator,dict-item"
from math import pi
from typing import Optional, Union, Annotated

import torch
from torch import Tensor

from ..constants import arcsec_to_rad, G_over_c2
from ..utils import translate_rotate
from .base import ThinLens, CosmologyType, NameType, ZLType
from ..parametrized import unpack
from ..packed import Packed

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
    x0: Optional[Union[Tensor, float]]
        x-coordinate of the center of the lens (arcsec).
    y0: Optional[Union[Tensor, float]]
        y-coordinate of the center of the lens (arcsec).
    mass: Optional[Union[Tensor, float]]
        Total mass of the lens (Msol).
    core_radius: Optional[Union[Tensor, float]]
        Core radius of the lens (arcsec).
    scale_radius: Optional[Union[Tensor, float]]
        Scaling radius of the lens (arcsec).
    s: float
        Softening parameter to prevent numerical instabilities.
    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "mass": 1e12,
        "core_radius": 0.1,
        "scale_radius": 1.0,
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
        core_radius: Annotated[
            Optional[Union[Tensor, float]], "Core radius of the lens", True, "arcsec"
        ] = None,
        scale_radius: Annotated[
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
        x0: Optional[Tensor]
            x-coordinate of the center of the lens.
        y0: Optional[Tensor]
            y-coordinate of the center of the lens.
        mass: Optional[Tensor]
            Total mass of the lens (Msol).
        core_radius: Optional[Tensor]
            Core radius of the lens.
        scale_radius: Optional[Tensor]
            Scaling radius of the lens.
        s: float
            Softening parameter to prevent numerical instabilities.
        """
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("mass", mass)
        self.add_param("core_radius", core_radius)
        self.add_param("scale_radius", scale_radius)
        self.s = s

    @unpack
    def get_convergence_0(
        self,
        z_s,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        core_radius: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        **kwargs,
    ):
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        sigma_crit = self.cosmology.critical_surface_density(z_l, z_s, params)
        return mass / (2 * torch.pi * sigma_crit * core_radius * scale_radius * (d_l * arcsec_to_rad) ** 2)  # fmt: skip

    @unpack
    def mass_enclosed_2d(
        self,
        theta,
        z_s,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        core_radius: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Calculate the mass enclosed within a two-dimensional radius. Using equation A10 from `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_.

        Parameters
        ----------
        theta: Tensor
            Radius at which to calculate enclosed mass (arcsec).
        z_s: Tensor
            Source redshift.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The mass enclosed within the given radius.
        """
        # fmt: off
        theta = theta + self.s
        d_l = self.cosmology.angular_diameter_distance(z_l, params) # Mpc
        sigma_crit = self.cosmology.critical_surface_density(z_l, z_s, params) # Msun / Mpc^2
        surface_density_0 = self.get_convergence_0(z_s, params) * sigma_crit # Msun / Mpc^2
        total_mass = 2 * pi * surface_density_0 * core_radius * scale_radius * (d_l * arcsec_to_rad) ** 2 # Msun
        frac_enclosed_num = ((core_radius**2 + theta**2).sqrt() - core_radius - (scale_radius**2 + theta**2).sqrt() + scale_radius) # arcsec
        frac_enclosed_denom = (scale_radius - core_radius) # arcsec
        return total_mass * frac_enclosed_num / frac_enclosed_denom

    # fmt: on

    @staticmethod
    def central_convergence(
        z_l,
        z_s,
        rho_0,
        core_radius,
        scale_radius,
        critical_surface_density,
    ):
        """
        Compute the central convergence.

        Parameters
        -----------
        z_l: Tensor
            Lens redshift.
        z_s: Tensor
            Source redshift.
        rho_0: Tensor
            Central mass density.
        core_radius: Tensor
            Core radius of the lens (must be in Mpc).
        scale_radius: Tensor
            Scaling radius of the lens (must be in Mpc).
        cosmology: Cosmology
            The cosmology used for calculations.

        Returns
        --------
        Tensor
            The central convergence.
        """
        return pi * rho_0 * core_radius * scale_radius / ((core_radius + scale_radius) * critical_surface_density)  # fmt: skip

    @unpack
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        core_radius: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Calculate the deflection angle.

        Parameters
        ----------
        x: Tensor
            x-coordinate of the lens.
        y: Tensor
            y-coordinate of the lens.
        z_s: Tensor
            Source redshift.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        --------
        Tuple[Tensor, Tensor]
            The deflection angle in the x and y directions.
        """
        x, y = translate_rotate(x, y, x0, y0)
        R = (x**2 + y**2).sqrt() + self.s
        f = R / core_radius / (1 + (1 + (R / core_radius) ** 2).sqrt()) - R / (scale_radius * (1 + (1 + (R / scale_radius) ** 2).sqrt()))  # fmt: skip
        alpha = 2 * self.get_convergence_0(z_s, params) * core_radius * scale_radius / (scale_radius - core_radius) * f  # fmt: skip
        ax = alpha * x / R
        ay = alpha * y / R
        return ax, ay

    @unpack
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        core_radius: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the lensing potential. This calculation is based on equation A18 from `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_.

        Parameters
        --------
        x: Tensor
            x-coordinate of the lens.
        y: Tensor
            y-coordinate of the lens.
        z_s: Tensor
            Source redshift.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        --------
        Tensor
            The lensing potential (arcsec^2).
        """

        # fmt: off
        x, y = translate_rotate(x, y, x0, y0)
        d_l = self.cosmology.angular_diameter_distance(z_l, params) # Mpc
        d_s = self.cosmology.angular_diameter_distance(z_s, params) # Mpc
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s, params) # Mpc

        R_squared = x**2 + y**2 + self.s # arcsec^2
        sigma_crit = self.cosmology.critical_surface_density(z_l, z_s, params) # Msun / Mpc^2
        surface_density_0 = self.get_convergence_0(z_s, params) * sigma_crit # Msun / Mpc^2

        coeff = -8 * pi * G_over_c2 * surface_density_0 * (d_l * d_ls / d_s) * core_radius * scale_radius / (scale_radius - core_radius) # arcsec

        scale_a = (scale_radius**2 + R_squared).sqrt() # arcsec
        scale_b = (core_radius**2 + R_squared).sqrt() # arcsec
        scale_c = core_radius * (core_radius + (core_radius**2 + R_squared).sqrt()).log() # arcsec
        scale_d = scale_radius * (scale_radius + (scale_radius**2 + R_squared).sqrt()).log() # arcsec
        scale_factor = (scale_a - scale_b + scale_c - scale_d) # arcsec
        return coeff * scale_factor

    # fmt: on

    @unpack
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        core_radius: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the projected mass density, based on equation A6.

        Parameters
        -----------
        x: Tensor
            x-coordinate of the lens.
        y: Tensor
            y-coordinate of the lens.
        z_s: Tensor
            Source redshift.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The projected mass density.
        """
        x, y = translate_rotate(x, y, x0, y0)
        R_squared = x**2 + y**2 + self.s
        coeff = self.get_convergence_0(z_s, params) * core_radius * scale_radius / (scale_radius - core_radius)  # fmt: skip
        return coeff * (1 / (core_radius**2 + R_squared).sqrt() - 1 / (scale_radius**2 + R_squared).sqrt())  # fmt: skip
