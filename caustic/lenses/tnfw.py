from math import pi
from typing import Any, Optional

import torch
from torch import Tensor

from ..constants import G_over_c2, arcsec_to_rad, rad_to_arcsec
from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

DELTA = 200.0

__all__ = ("TNFW",)


class TNFW(ThinLens):
    """
    Truncated NFW lens class. This class models a lens using the Truncated Navarro-Frenk-White (TNFW) profile,
    with a truncation function (r_trunc^2)/(r^2+r_trunc^2), as described in https://arxiv.org/abs/1101.0650
    for n=1. 
    
    Attributes:
        z_l (Optional[Tensor]): Redshift of the lens. Default is None.
        x0 (Optional[Tensor]): x-coordinate of the lens center in the lens plane. 
            Default is None.
        y0 (Optional[Tensor]): y-coordinate of the lens center in the lens plane. 
            Default is None.
        m (Optional[Tensor]): Mass of the lens. Default is None.
        c (Optional[Tensor]): Concentration parameter of the lens. Default is None.
        r_trunc (Optional[Tensor]): Truncation radius of the the lens. Default is None.
        s (float): Softening parameter to avoid singularities at the center of the lens. 
            Default is 0.0.
    Methods:
        get_scale_radius: Returns the scale radius of the lens.
        get_scale_density: Returns the scale density of the lens.
        get_convergence_s: Returns the dimensionless surface mass density of the lens.
        _f: Helper method for computing deflection angles.
        _g: Helper method for computing lensing potential.
        _h: Helper method for computing reduced deflection angles.
        deflection_angle_hat: Computes the reduced deflection angle.
        deflection_angle: Computes the deflection angle.
        convergence: Computes the convergence (dimensionless surface mass density).
        potential: Computes the lensing potential.
    """
    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        m: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        r_trunc: Optional[Tensor] = None,
        s: float = 0.001,
    ):
        """
        Initialize an instance of the TNFW lens class.
        """
        super().__init__(name, cosmology, z_l)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("m", m)
        self.add_param("c", c)
        self.add_param("r_trunc", r_trunc)
        self.s = torch.tensor(s)

    def get_scale_radius(self, z_l: Tensor, m: Tensor, c: Tensor, params: Optional["Packed"] = None) -> Tensor:
        """
        Calculate the scale radius of the lens.

        Args:
            z_l (Tensor): Redshift of the lens.
            m (Tensor): Mass of the lens.
            c (Tensor): Concentration parameter of the lens.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The scale radius of the lens in Mpc.
        """
        critical_density = self.cosmology.critical_density(z_l, params)
        r_delta = (3 * m / (4 * pi * DELTA * critical_density)) ** (1 / 3)
        return 1 / c * r_delta

    def get_scale_density(self, z_l: Tensor, c: Tensor, params: Optional["Packed"] = None) -> Tensor:
        """
        Calculate the scale density of the lens.

        Args:
            z_l (Tensor): Redshift of the lens.
            c (Tensor): Concentration parameter of the lens.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The scale density of the lens in solar masses per Mpc cubed.
        """
        return (
            DELTA
            / 3
            * self.cosmology.critical_density(z_l, params)
            * c**3
            / (torch.log(1 + c) - c / (1 + c))
        )

    def get_convergence_s(self: Tensor, z_l: Tensor, z_s: Tensor, m: Tensor, c: Tensor, params: Optional["Packed"] = None) -> Tensor:
        """
        Calculate the dimensionless surface mass density of the lens at the scale radius.

        Args:
            z_l (Tensor): Redshift of the lens.
            z_s (Tensor): Redshift of the source.
            m (Tensor): Mass of the lens.
            c (Tensor): Concentration parameter of the lens.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The dimensionless surface mass density of the lens.
        """
        critical_surface_density = self.cosmology.critical_surface_density(z_l, z_s, params)
        return self.get_scale_density(z_l, c, params) * self.get_scale_radius(z_l, m, c, params) / critical_surface_density

    @staticmethod
    def _F(r: Tensor) -> Tensor:
        """
        Helper function F for the TNFW profile.

        Args:
            r (Tensor): The scaled radius (xi / scale_radius)

        Returns:
            Tensor: Result of the F(X) function
        """

        return torch.where(
            r < 1,
            (1 - r ** 2).arctanh() / (1 - r ** 2).sqrt(),
            torch.where(
                r > 1,
                (r ** 2 - 1).arctan() / (r ** 2 - 1).sqrt(),
                0.0
                )
            )

    # def _L(r: Tensor, z_l: Tensor, m: Tensor, c: Tensor, r_trunc: Tensor, params: Optional["Packed"]) -> Tensor:
    @staticmethod
    def _L(r: Tensor, tau: Tensor) -> Tensor:
        """
        Helper function L for the TNFW profile.

        Args:
            r (Tensor): The scaled radius (xi / scale_radius)
            z_l (Tensor): Redshift of the lens.
            m (Tensor): Mass of the lens.
            c (Tensor): Concentration parameter of the lens.
            r_trunc (Tensor): Truncation radius of the the lens.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: Result of the L(X) function
        """
        # scale_radius = self.get_scale_radius(z_l, m, c, params)
        # tau = r_trunc / scale_radius
        return (r / ((tau ** 2 + r ** 2).sqrt() + tau)).log()

    def reduced_deflection_angle(self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None) -> tuple[Tensor, Tensor]:
        """
        Compute the reduced deflection angle.

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            tuple[Tensor, Tensor]: The reduced deflection angles in the x and y directions.
        """
        z_l, x0, y0, m, c, r_trunc = self.unpack(params)
        
        x, y = translate_rotate(x, y, x0, y0)
        th = (x ** 2 + y ** 2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        scale_radius = self.get_scale_radius(z_l, m, c, params)
        xi = d_l * th * arcsec_to_rad
        r = xi / scale_radius
        r = torch.maximum(r, self.s)
        convergence_s = self.get_convergence_s(z_l, z_s, m, c, params)
        tau = r_trunc / scale_radius
        F = self._F(r)
        L = self._L(r, tau)

        # TODO: compare first line between TNFW and TNFW_Baltz papers
        average_convergence = 4 * convergence_s * tau ** 2 / (r ** 2 * (tau ** 2  + 1) ** 2) * (
            (tau ** 2 + 2 * r ** 2 + 1) * F
            + tau * pi
            + (tau ** 2 - 1) * (tau).log()
            + (tau ** 2 + r ** 2).sqrt() * (-pi + (tau ** 2 - 1) / tau * L)
            )

        deflection_angle = th * average_convergence     # TODO: make sure this form is valid, has right units
        ax = deflection_angle * x / th
        ay = deflection_angle * y / th
        d_s = self.cosmology.angular_diameter_distance(z_s, params)
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s, params)
        return ax * d_ls / d_s, ay * d_ls / d_s

    def convergence(self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None) -> Tensor:
        """
        Compute the convergence (dimensionless surface density).

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.
        
        Returns:
            Tensor: The convergence
        """
        z_l, x0, y0, m, c, r_trunc = self.unpack(params)

        x, y = translate_rotate(x, y, x0, y0)
        th = (x ** 2 + y ** 2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        scale_radius = self.get_scale_radius(z_l, m, c, params)
        xi = d_l * th * arcsec_to_rad
        r = xi / scale_radius
        r = torch.maximum(r, self.s)
        convergence_s = self.get_convergence_s(z_l, z_s, m, c, params)
        tau = r_trunc / scale_radius
        F = self._F(r)
        L = self._L(r, tau)

        return 4 * convergence_s * tau ** 2 / (2 * (tau ** 2 + 1) ** 2) * (
            (tau ** 2 + 1) / (r ** 2 - 1) * (1 - F)
            + 2 * F
            - pi / (tau ** 2 + r ** 2).sqrt()
            + (tau ** 2 - 1) / (tau * (tau ** 2 + r ** 2).sqrt()) * L
            )

    def potential(self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None) -> Tensor:
        """
        Compute the lensing potential

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The lensing potential.
        """
        z_l, x0, y0, m, c, r_trunc = self.unpack(params)

        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        scale_radius = self.get_scale_radius(z_l, m, c, params)
        scale_density = self.get_scale_density(z_l, c, params)
        xi = d_l * th * arcsec_to_rad
        r = xi / scale_radius  # xi / xi_0
        r = torch.maximum(r, self.s)
        tau = r_trunc / scale_radius
        u = r ** 2
        M_0 = 4 * pi * scale_radius ** 3 * scale_density
        F = self._F(r)
        L = self._L(r, tau)
        
        return 2 * G_over_c2 * M_0 / (tau ** 2 + 1) ** 2 * (
            2 * tau ** 2 * pi * (
                tau - (tau ** 2 + u).sqrt() + tau * (tau + (tau ** 2 + u).sqrt()).log()
                )
            + 2 * (tau ** 2 - 1) * tau * (tau ** 2 + u).sqrt() * L
            + tau ** 2 * (tau ** 2 - 1) * L ** 2
            + 4 * tau ** 2 * (u - 1) * F 
            + tau ** 2 * (tau ** 2 - 1) * torch.acos(1/r) ** 2
            + tau ** 2 * ((tau ** 2 - 1) * torch.log(tau) - tau ** 2 - 1) * torch.log(u)
            - tau ** 2 * ((tau ** 2 - 1) * torch.log(tau) * torch.log(4 * tau)
                + 2 * torch.log(tau / 2) 
                - 2 * tau * (tau - pi) * torch.log(2 * tau)
                )
            )

