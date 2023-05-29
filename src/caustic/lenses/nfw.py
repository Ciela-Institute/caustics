from math import pi
from typing import Any, Optional

import torch
from torch import Tensor

from ..constants import G_over_c2, arcsec_to_rad, rad_to_arcsec
from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

DELTA = 200.0

__all__ = ("NFW",)


class NFW(ThinLens):
    """
    NFW lens class. This class models a lens using the Navarro-Frenk-White (NFW) profile.
    The NFW profile is a spatial density profile of dark matter halo that arises in 
    cosmological simulations.

    Attributes:
        z_l (Optional[Tensor]): Redshift of the lens. Default is None.
        thx0 (Optional[Tensor]): x-coordinate of the lens center in the lens plane. 
            Default is None.
        thy0 (Optional[Tensor]): y-coordinate of the lens center in the lens plane. 
            Default is None.
        m (Optional[Tensor]): Mass of the lens. Default is None.
        c (Optional[Tensor]): Concentration parameter of the lens. Default is None.
        s (float): Softening parameter to avoid singularities at the center of the lens. 
            Default is 0.0.

    Methods:
        get_r_s: Returns the scale radius of the lens.
        get_rho_s: Returns the scale density of the lens.
        get_kappa_s: Returns the dimensionless surface mass density of the lens.
        _f: Helper method for computing deflection angles.
        _g: Helper method for computing lensing potential.
        _h: Helper method for computing reduced deflection angles.
        alpha_hat: Computes the reduced deflection angle.
        alpha: Computes the deflection angle.
        kappa: Computes the convergence (dimensionless surface mass density).
        Psi: Computes the lensing potential.
    """
    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        m: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        s: float = 0.0,
    ):
        """
        Initialize an instance of the NFW lens class.

        Args:
            name (str): Name of the lens instance.
            cosmology (Cosmology): An instance of the Cosmology class which contains 
                information about the cosmological model and parameters.
            z_l (Optional[Tensor]): Redshift of the lens. Default is None.
            thx0 (Optional[Tensor]): x-coordinate of the lens center in the lens plane. 
                Default is None.
            thy0 (Optional[Tensor]): y-coordinate of the lens center in the lens plane. 
                Default is None.
            m (Optional[Tensor]): Mass of the lens. Default is None.
            c (Optional[Tensor]): Concentration parameter of the lens. Default is None.
            s (float): Softening parameter to avoid singularities at the center of the lens. 
                Default is 0.0.
        """
        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("m", m)
        self.add_param("c", c)
        self.s = s

    def get_r_s(self, z_l, m, c, x) -> Tensor:
        """
        Calculate the scale radius of the lens.

        Args:
            z_l (Tensor): Redshift of the lens.
            m (Tensor): Mass of the lens.
            c (Tensor): Concentration parameter of the lens.
            x (dict): Additional parameters.

        Returns:
            Tensor: The scale radius of the lens in Mpc.
        """
        rho_cr = self.cosmology.rho_cr(z_l, x)
        r_delta = (3 * m / (4 * pi * DELTA * rho_cr)) ** (1 / 3)
        return 1 / c * r_delta

    def get_rho_s(self, z_l, c, x) -> Tensor:
        """
        Calculate the scale density of the lens.

        Args:
            z_l (Tensor): Redshift of the lens.
            c (Tensor): Concentration parameter of the lens.
            x (dict): Additional parameters.

        Returns:
            Tensor: The scale density of the lens in solar masses per Mpc cubed.
        """
        return (
            DELTA
            / 3
            * self.cosmology.rho_cr(z_l, x)
            * c**3
            / ((1 + c).log() - c / (1 + c))
        )

    def get_kappa_s(self, z_l, z_s, m, c, x) -> Tensor:
        """
        Calculate the dimensionless surface mass density of the lens.

        Args:
            z_l (Tensor): Redshift of the lens.
            z_s (Tensor): Redshift of the source.
            m (Tensor): Mass of the lens.
            c (Tensor): Concentration parameter of the lens.
            x (dict): Additional parameters.

        Returns:
            Tensor: The dimensionless surface mass density of the lens.
        """
        Sigma_cr = self.cosmology.Sigma_cr(z_l, z_s)
        return self.get_rho_s(z_l, c, x) * self.get_r_s(z_l, m, c, x) / Sigma_cr

    @classmethod
    def _f(cls, x: Tensor) -> Tensor:
        """
        Helper method for computing deflection angles.

        Args:
            x (Tensor): The scaled radius (xi / xi_0).

        Returns:
            Tensor: Result of the deflection angle computation.
        """
        # TODO: generalize beyond torch, or patch Tensor
        return torch.where(
            x > 1,
            1 - 2 / (x**2 - 1).sqrt() * ((x - 1) / (x + 1)).sqrt().arctan(),
            torch.where(
                x < 1,
                1 - 2 / (1 - x**2).sqrt() * ((1 - x) / (1 + x)).sqrt().arctanh(),
                0.0,
            ),
        )

    @classmethod
    def _g(cls, x: Tensor) -> Tensor:
        """
        Helper method for computing lensing potential.

        Args:
            x (Tensor): The scaled radius (xi / xi_0).

        Returns:
            Tensor: Result of the lensing potential computation.
        """
        # TODO: generalize beyond torch, or patch Tensor
        term_1 = (x / 2).log() ** 2
        term_2 = torch.where(
            x > 1,
            (1 / x).arccos() ** 2,
            torch.where(x < 1, -(1 / x).arccosh() ** 2, 0.0),
        )
        return term_1 + term_2

    @classmethod
    def _h(cls, x: Tensor) -> Tensor:
        """
        Helper method for computing reduced deflection angles.

        Args:
            x (Tensor): The scaled radius (xi / xi_0).

        Returns:
            Tensor: Result of the reduced deflection angle computation.
        """
        term_1 = (x / 2).log()
        term_2 = torch.where(
            x > 1,
            term_1 + (1 / x).arccos() * 1 / (x**2 - 1).sqrt(),
            torch.where(
                x < 1,
                term_1 + (1 / x).arccosh() * 1 / (1 - x**2).sqrt(),
                1.0 + torch.tensor(1 / 2).log(),
            ),
        )
        return term_2

    def alpha_hat(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the reduced deflection angle.

        Args:
            thx (Tensor): x-coordinates in the lens plane.
            thy (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            tuple[Tensor, Tensor]: The reduced deflection angles in the x and y directions.
        """
        z_l, thx0, thy0, m, c = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_dist(z_l)
        r_s = self.get_r_s(z_l, m, c, x)
        xi = d_l * th * arcsec_to_rad
        r = xi / r_s

        alpha = (
            16
            * pi
            * G_over_c2
            * self.get_rho_s(z_l, c, x)
            * r_s**3
            * self._h(r)
            * rad_to_arcsec
            / xi
        )

        ax = alpha * thx / th
        ay = alpha * thy / th
        return ax, ay

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the deflection angle.

        Args:
            thx (Tensor): x-coordinates in the lens plane.
            thy (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            tuple[Tensor, Tensor]: The deflection angles in the x and y directions.
        """
        z_l = self.unpack(x)[0]

        d_s = self.cosmology.angular_diameter_dist(z_s, x)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s, x)
        ahx, ahy = self.alpha_hat(thx, thy, z_s, x)
        return d_ls / d_s * ahx, d_ls / d_s * ahy

    def kappa(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Compute the convergence (dimensionless surface mass density).

        Args:
            thx (Tensor): x-coordinates in the lens plane.
            thy (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tensor: The convergence (dimensionless surface mass density).
        """
        z_l, thx0, thy0, m, c = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_dist(z_l, x)
        r_s = self.get_r_s(z_l, m, c, x)
        xi = d_l * th * arcsec_to_rad
        r = xi / r_s  # xi / xi_0
        kappa_s = self.get_kappa_s(z_l, z_s, m, c, x)
        return 2 * kappa_s * self._f(r) / (r**2 - 1)

    def Psi(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Compute the lensing potential.

        Args:
            thx (Tensor): x-coordinates in the lens plane.
            thy (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tensor: The lensing potential.
        """
        z_l, thx0, thy0, m, c = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_dist(z_l, x)
        r_s = self.get_r_s(z_l, m, c, x)
        xi = d_l * th * arcsec_to_rad
        r = xi / r_s  # xi / xi_0
        kappa_s = self.get_kappa_s(z_l, z_s, m, c, x)
        return 2 * kappa_s * self._g(r) * r_s**2 / (d_l**2 * arcsec_to_rad**2)
