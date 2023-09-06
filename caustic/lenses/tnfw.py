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
    """Truncated Navaro-Frenk-White profile
    
    TNFW lens class. This class models a lens using the truncated
    Navarro-Frenk-White (NFW) profile.  The NFW profile is a spatial
    density profile of dark matter halo that arises in cosmological
    simulations. It is truncated with an extra scaling term which
    smoothly reduces the density such that it does not diverge to
    infinity. This is based off the paper by Baltz et al. 2009:

    https://arxiv.org/abs/0705.0682
    
    https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract

    Args:
        name (str): Name of the lens instance.
        cosmology (Cosmology): An instance of the Cosmology class which contains 
            information about the cosmological model and parameters.
        z_l (Optional[Tensor]): Redshift of the lens.
        x0 (Optional[Tensor]): Center of lens position on x-axis (arcsec). 
        y0 (Optional[Tensor]): Center of lens position on y-axis (arcsec). 
        m (Optional[Tensor]): Mass of the lens (Msol).
        c (Optional[Tensor]): Concentration parameter of the lens (r200/rs for a classic NFW).
        t (Optional[Tensor]): Truncation scale. Ratio of truncation radius to scale radius (rt/rs).
        s (float): Softening parameter to avoid singularities at the center of the lens. 
            Default is 0.0.
        interpret_m_total_mass (bool): Indicates how to interpret the mass variable "m". If true
            the mass is intepreted as the total mass of the halo (good because it makes sense). If
            false it is intepreted as what the mass would have been within R200 of a an NFW that
            isn't truncated (good because it is easily compared with an NFW).
    
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
        interpret_m_total_mass: bool = True,
        name: str = None,
    ):
        """
        Initialize an instance of the TNFW lens class.

        """
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("m", m)
        self.add_param("c", c)
        self.add_param("t", t)
        self.s = s
        self.interpret_m_total_mass = interpret_m_total_mass

    @staticmethod
    def _F(x):
        """
        Helper method from Baltz et al. 2009 equation A.5
        """
        return torch.where(x == 1, torch.ones_like(x), ((1 / x.to(dtype=torch.cdouble)).arccos() / (x.to(dtype=torch.cdouble)**2 - 1).sqrt()).abs())

    @staticmethod
    def _L(x, t):
        """
        Helper method from Baltz et al. 2009 equation A.6
        """
        return (x / (t + (t**2 + x**2).sqrt())).log()

    @unpack(0)
    def get_scale_radius(self, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs) -> Tensor:
        """
        Calculate the scale radius of the lens. This is the same formula used for the classic NFW profile.

        Args:
            z_l (Tensor): Redshift of the lens.
            x0 (Tensor): Center of lens position on x-axis (arcsec). 
            y0 (Tensor): Center of lens position on y-axis (arcsec). 
            m (Tensor): Mass of the lens (Msol).
            c (Tensor): Concentration parameter of the lens (r200/rs for a classic NFW).
            t (Tensor): Truncation scale. Ratio of truncation radius to scale radius (rt/rs).
            params (dict): Dynamic parameter container.

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
            x0 (Tensor): Center of lens position on x-axis (arcsec). 
            y0 (Tensor): Center of lens position on y-axis (arcsec). 
            m (Tensor): Mass of the lens (Msol).
            c (Tensor): Concentration parameter of the lens (r200/rs for a classic NFW).
            t (Tensor): Truncation scale. Ratio of truncation radius to scale radius (rt/rs).
            params (dict): Dynamic parameter container.

        Returns:
            Tensor: The truncation radius of the lens in Mpc.
        """
        return t * self.get_scale_radius(params)

    @unpack(0)
    def get_M0(self, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs) -> Tensor:
        """
        Calculate the reference mass. This is an abstract reference mass used internally in the equations from Baltz et al. 2009.

        Args:
            z_l (Tensor): Redshift of the lens.
            x0 (Tensor): Center of lens position on x-axis (arcsec). 
            y0 (Tensor): Center of lens position on y-axis (arcsec). 
            m (Tensor): Mass of the lens (Msol).
            c (Tensor): Concentration parameter of the lens (r200/rs for a classic NFW).
            t (Tensor): Truncation scale. Ratio of truncation radius to scale radius (rt/rs).
            params (dict): Dynamic parameter container.

        Returns:
            Tensor: The reference mass of the lens in Msol.
        """
        if self.interpret_m_total_mass:
            return m * (t**2 + 1) / (t**2 * ((t**2 - 1) * t.log() + torch.pi * t - (t**2 + 1)))
        else:
            return 4 * torch.pi * self.get_scale_radius(params)**3 * self.get_scale_density(params)


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

    @unpack(3)
    def convergence(
            self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        TNFW convergence as given in Baltz et al. 2009. This is unitless since it is Sigma(x) / Sigma_crit.

        Args:
            z_l (Tensor): Redshift of the lens.
            x0 (Tensor): Center of lens position on x-axis (arcsec). 
            y0 (Tensor): Center of lens position on y-axis (arcsec). 
            m (Tensor): Mass of the lens (Msol).
            c (Tensor): Concentration parameter of the lens (r200/rs for a classic NFW).
            t (Tensor): Truncation scale. Ratio of truncation radius to scale radius (rt/rs).
            params (dict): Dynamic parameter container.

        Returns:
            Tensor: unitless convergence at requested position
        
        """
        x, y = translate_rotate(x, y, x0, y0)
        r = (x**2 + y**2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        rs = self.get_scale_radius(params)
        g = r * (d_l * arcsec_to_rad / rs)
        F = self._F(g)
        L = self._L(g, t)
        critical_density = self.cosmology.critical_surface_density(z_l, z_s, params)

        S = self.get_M0(params) / (2 * torch.pi * rs**2)
            
        t2 = t**2
        a1 = t2 / (t2 + 1)**2
        a2 = torch.where(g == 1, (t2 + 1) / 3., (t2 + 1) * (1 - F) / (g**2 - 1))
        a3 = 2 * F
        a4 = - torch.pi / (t2 + g**2).sqrt()
        a5 = (t2 - 1) * L / (t * (t2 + g**2).sqrt())
        return a1 * (a2 + a3 + a4 + a5) * S / critical_density

    @unpack(2)
    def projected_mass(
            self, r: Tensor, z_s: Tensor, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Total projected mass (Msol) within a radius r (Mpc).

        Args:
            z_l (Tensor): Redshift of the lens.
            x0 (Tensor): Center of lens position on x-axis (arcsec). 
            y0 (Tensor): Center of lens position on y-axis (arcsec). 
            m (Tensor): Mass of the lens (Msol).
            c (Tensor): Concentration parameter of the lens (r200/rs for a classic NFW).
            t (Tensor): Truncation scale. Ratio of truncation radius to scale radius (rt/rs).
            params (dict): Dynamic parameter container.

        Returns:
            Tensor: Integrated mass projected in infinite cylinder within radius r.
        """
        rs = self.get_scale_radius(params)
        g = r / rs
        t2 = t**2
        F = self._F(g)
        L = self._L(g, t)
        a1 = t2 / (t2 + 1)**2
        a2 = (t2 + 1 + 2*(g**2 - 1)) * F
        a3 = t * torch.pi
        a4 = (t2 - 1) * t.log()
        a5 = (t2 + g**2).sqrt() * (-torch.pi + (t2 - 1) * L / t)
        S = self.get_M0(params)
        return S * a1 * (a2 + a3 + a4 + a5)
        
    @unpack(3)
    def physical_deflection_angle(
            self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, m, c, t, *args, params: Optional["Packed"] = None, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """Compute the physical deflection angle (arcsec) for this lens at
        the requested position. Note that the NFW/TNFW profile is more
        naturally represented as a physical deflection angle, this is
        easily internally converted to a reduced deflection angle.

        Args:
            z_l (Tensor): Redshift of the lens.
            x0 (Tensor): Center of lens position on x-axis (arcsec). 
            y0 (Tensor): Center of lens position on y-axis (arcsec). 
            m (Tensor): Mass of the lens (Msol).
            c (Tensor): Concentration parameter of the lens (r200/rs for a classic NFW).
            t (Tensor): Truncation scale. Ratio of truncation radius to scale radius (rt/rs).
            params (dict): Dynamic parameter container.

        Returns:
            tuple[Tensor, Tensor]: The physical deflection angles in the x and y directions (arcsec).

        """
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        rs = self.get_scale_radius(params)
        x, y = translate_rotate(x, y, x0, y0)
        r = ((x**2 + y**2).sqrt() + self.s) * d_l * arcsec_to_rad
        theta = torch.arctan2(y,x)

        # The below actually equally comes from eq 2.13 in Meneghetti notes
        dr = self.projected_mass(r, z_s, params) / r # note dpsi(u)/du = 2x*dpsi(x)/dx when u = x^2
        S = 4 * G_over_c2 * rad_to_arcsec
        return S * dr * theta.cos(), S * dr * theta.sin()

    @unpack(3)
    def potential(
        self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, c, t, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the lensing potential. Note that this is not a unitless potential! This is the potential as given in Baltz et al. 2009.

        TODO: convert to dimensionless potential.

        Args:
            z_l (Tensor): Redshift of the lens.
            x0 (Tensor): Center of lens position on x-axis (arcsec). 
            y0 (Tensor): Center of lens position on y-axis (arcsec). 
            m (Tensor): Mass of the lens (Msol).
            c (Tensor): Concentration parameter of the lens (r200/rs for a classic NFW).
            t (Tensor): Truncation scale. Ratio of truncation radius to scale radius (rt/rs).
            params (dict): Dynamic parameter container.

        Returns:
            Tensor: The lensing potential.
        """
        x, y = translate_rotate(x, y, x0, y0)
        r = (x**2 + y**2).sqrt() + self.s
        rs = self.get_scale_radius(params)
        g = r / rs
        t2 = t**2
        u = g**2
        F = self._F(g)
        L = self._L(g, t)

        #d_l = self.cosmology.angular_diameter_distance(z_l, params)
        S = 2 * self.get_M0(params) * G_over_c2 # * rad_to_arcsec * d_l**2
        a1 = 1 / (t2 + 1)**2
        a2 = 2 * torch.pi * t2 * (t - (t2 + u).sqrt() + t * (t + (t2 + u).sqrt()).log())
        a3 = 2 * (t2 - 1) * t * (t2 + u).sqrt() * L
        a4 = t2 * (t2 - 1) * L**2
        a5 = 4 * t2 * (u - 1) * F
        a6 = t2 * (t2 - 1) * (1 / g.to(dtype=torch.cdouble)).arccos().abs()**2
        a7 = t2 * ((t2 - 1) * t.log() - t2 - 1) * u.log()
        a8 = t2 * ((t2 - 1) * t.log() * (4*t).log() + 2 * (t/2).log() - 2 * t * (t - torch.pi) * (2*t).log())

        return S * a1 * (a2 + a3 + a4 + a5 + a6 + a7 - a8)
