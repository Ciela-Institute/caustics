from math import pi
from typing import Any, Optional, Union

import torch
from torch import Tensor

from ..constants import G_over_c2, arcsec_to_rad, rad_to_arcsec
from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens
from ..parametrized import unpack

DELTA = 200.0

__all__ = ("TNFW",)

class TNFW(ThinLens):
    """fixme
    
    TNFW lens class. This class models a lens using the truncated
    Navarro-Frenk-White (NFW) profile.  The NFW profile is a spatial
    density profile of dark matter halo that arises in cosmological
    simulations. It is truncated with an extra scaling term which
    smoothly reduces the density such that it does not diverge to
    infinity. This is based off the paper by Baltz et al. 2008:
    https://arxiv.org/abs/0705.0682

    Attributes:
        z_l (Optional[Tensor]): Redshift of the lens. Default is None.
        x0 (Optional[Tensor]): x-coordinate of the lens center in the lens plane. 
            Default is None.
        y0 (Optional[Tensor]): y-coordinate of the lens center in the lens plane. 
            Default is None.
        m (Optional[Tensor]): Mass of the lens. Default is None.
        c (Optional[Tensor]): Concentration parameter of the lens. Default is None.
        t (Optional[Tensor]): Truncation scale (t = truncation radius / scale radius).
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
        cosmology: Cosmology,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        m: Optional[Union[Tensor, float]] = None,
        c: Optional[Union[Tensor, float]] = None,
        t: Optional[Union[Tensor, float]] = None,
        s: float = 0.0,
        name: str = None,
    ):
        """
        Initialize an instance of the NFW lens class.

        Args:
            name (str): Name of the lens instance.
            cosmology (Cosmology): An instance of the Cosmology class which contains 
                information about the cosmological model and parameters.
            z_l (Optional[Union[Tensor, float]]): Redshift of the lens. Default is None.
            x0 (Optional[Union[Tensor, float]]): x-coordinate of the lens center in the lens plane. 
                Default is None.
            y0 (Optional[Union[Tensor, float]]): y-coordinate of the lens center in the lens plane. 
                Default is None.
            m (Optional[Union[Tensor, float]]): Mass of the lens. Default is None.
            c (Optional[Union[Tensor, float]]): Concentration parameter of the lens. Default is None.
            s (float): Softening parameter to avoid singularities at the center of the lens. 
                Default is 0.0.
        """
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("m", m)
        self.add_param("c", c)
        self.add_param("t", t)
        self.s = s

    @staticmethod
    def _F(x):
        return torch.where(x == 1, torch.ones_like(x), ((1 / x.to(dtype=torch.cdouble)).arccos() / (x.to(dtype=torch.cdouble)**2 - 1).sqrt()).abs())

    @staticmethod
    def _L(x, t):
        return (x / (t + (t**2 + x**2).sqrt())).log()

    @unpack(0)
    def get_scale_radius(self, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs) -> Tensor:
        """
        Calculate the scale radius of the lens.

        Args:
            z_l (Tensor): Redshift of the lens.
            m (Tensor): Mass of the lens.
            c (Tensor): Concentration parameter of the lens.
            x (dict): Dynamic parameter container.

        Returns:
            Tensor: The scale radius of the lens in Mpc.
        """
        critical_density = self.cosmology.critical_density(z_l, params)
        r_delta = (3 * m / (4 * pi * DELTA * critical_density)) ** (1 / 3)
        return 1 / c * r_delta

    @unpack(0)
    def get_truncation_radius(self, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs) -> Tensor:
        """
        Calculate the truncation radius of the TNFW lens.

        Args:
            z_l (Tensor): Redshift of the lens.
            m (Tensor): Mass of the lens.
            c (Tensor): Concentration parameter of the lens.
            x (dict): Dynamic parameter container.

        Returns:
            Tensor: The truncation radius of the lens in Mpc.
        """
        return t * self.get_scale_radius(params)


    @unpack(0)
    def get_scale_density(self, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs) -> Tensor:
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
            / ((1 + c).log() - c / (1 + c))
        )

    @unpack(0)
    def get_M0(self, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs) -> Tensor:
        """
        Calculate the reference mass.

        Args:
            z_l (Tensor): Redshift of the lens.
            c (Tensor): Concentration parameter of the lens.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The scale density of the lens in solar masses per Mpc cubed.
        """
        return m * (t**2 + 1) / (t**2 * ((t**2 - 1) * t.log() + torch.pi * t - (t**2 + 1)))

    @unpack(1)
    def get_convergence_s(self, z_s, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs) -> Tensor:
        """
        Calculate the dimensionless surface mass density of the lens.

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
        return self.get_scale_density(params) * self.get_scale_radius(params) / critical_surface_density
    

    @unpack(3)
    def convergence(
            self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        x, y = translate_rotate(x, y, x0, y0)
        r = (x**2 + y**2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        rs = self.get_scale_radius(params)
        x = r * (d_l * arcsec_to_rad / rs)
        F = self._F(x)
        L = self._L(x, t)
        critical_density = self.cosmology.critical_surface_density(z_l, z_s, params)

        S = self.get_M0(params) / (2 * torch.pi * rs**2)
        a1 = t**2 / (t**2 + 1)**2
        a2 = (t**2 + 1) * (1 - F) / (x**2 - 1)
        a3 = 2 * F
        a4 = - torch.pi / (t**2 + x**2).sqrt()
        a5 = (t**2 - 1) * L / (t * (t**2 + x**2).sqrt())
        return a1 * (a2 + a3 + a4 + a5) * S / critical_density

    @unpack(0)
    def get_scale_density(self, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs) -> Tensor:
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
            / ((1 + c).log() - c / (1 + c))
        )

    @unpack(2)
    def projected_mass(
            self, r: Tensor, z_s: Tensor, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        rs = self.get_scale_radius(params)
        x = r / rs
        F = self._F(x)
        L = self._L(x, t)
        a1 = t**2 / (t**2 + 1)**2
        a2 = (t**2 + 1 + 2*(x**2 - 1)) * F
        a3 = t * torch.pi
        a4 = (t**2 - 1) * t.log()
        a5 = (t**2 + x**2).sqrt() * (-torch.pi + (t**2 - 1) * L / t)
        S = self.get_M0(params)
        return S * a1 * (a2 + a3 + a4 + a5)
        
    @unpack(3)
    def reduced_deflection_angle(
            self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the reduced deflection angle.

        Args:
            x (Tensor): x-coordinates in the lens plane (arcsec).
            y (Tensor): y-coordinates in the lens plane (arcsec).
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            tuple[Tensor, Tensor]: The reduced deflection angles in the x and y directions (arcsec).
        """
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        rs = self.get_scale_radius(params)
        x, y = translate_rotate(x, y, x0, y0)
        r = ((x**2 + y**2).sqrt() + self.s) * d_l * arcsec_to_rad
        theta = torch.arctan2(y,x)
        x = r / rs

        dr = 2 * self.projected_mass(r, z_s, params) / x # note dpsi(u)/du = 2x*dpsi(x)/dx when u = x^2
        S = 2 * G_over_c2 * rad_to_arcsec * d_l
        return S * dr * theta.cos(), S * dr * theta.sin()

    @unpack(3)
    def potential(
        self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, c, t, *args, params: Optional["Packed"] = None, **kwargs
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
        r = (x**2 + y**2).sqrt() + self.s
        rs = self.get_scale_radius(params)
        x = r / rs
        u = x**2
        F = self._F(x)
        L = self._L(x, t)

        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        S = 2 * self.get_M0(params) * G_over_c2 * rad_to_arcsec * d_l**2
        a1 = 1 / (t**2 + 1)**2
        a2 = 2 * torch.pi * t**2 * (t - (t**2 + u).sqrt() + t * (t + (t**2 + u).sqrt()).log())
        a3 = 2 * (t**2 - 1) * t * (t**2 + u).sqrt() * L
        a4 = t**2 * (t**2 - 1) * L**2
        a5 = 4 * t**2 * (u - 1) * F
        a6 = t**2 * (t**2 - 1) * (1 / x).arccos()**2
        a7 = t**2 * ((t**2 - 1) * t.log() - t**2 - 1) * u.log()
        a8 = t**2 * ((t**2 - 1) * t.log() * (4*t).log() + 2 * (t/2).log() - 2 * t * (t - torch.pi) * (2*t).log())

        return S * a1 * (a2 + a3 + a4 + a5 + a6 + a7 + a8)
