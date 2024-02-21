# mypy: disable-error-code="operator,union-attr"
from math import pi
from typing import Optional, Union

import torch
from torch import Tensor

from ..constants import G_over_c2, arcsec_to_rad, rad_to_arcsec
from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens
from ..parametrized import unpack
from ..packed import Packed

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

    Notes
    ------
        The mass `m` in the TNFW profile corresponds to the total mass
        of the lens. This is different from the NFW profile where the
        mass `m` parameter corresponds to the mass within R200. If you
        prefer the "mass within R200" version you can set:
        `interpret_m_total_mass = False` on initialization of the
        object. However, the mass within R200 will be computed for an
        NFW profile, not a TNFW profile. This is in line with how
        lenstronomy interprets the mass parameter.

    Parameters
    -----
    name: string
        Name of the lens instance.

    cosmology: Cosmology
        An instance of the Cosmology class which contains
        information about the cosmological model and parameters.

    z_l: Optional[Tensor]
        Redshift of the lens.

        *Unit: unitless*

    x0: Optional[Tensor]
        Center of lens position on x-axis.

        *Unit: arcsec*

    y0: Optional[Tensor]
        Center of lens position on y-axis.

        *Unit: arcsec*

    mass: Optional[Tensor]
        Mass of the lens.

        *Unit: Msun*

    scale_radius: Optional[Tensor]
        Scale radius of the TNFW lens.

        *Unit: arcsec*

    tau: Optional[Tensor]
        Truncation scale. Ratio of truncation radius to scale radius.

        *Unit: unitless*

    s: float
        Softening parameter to avoid singularities at the center of the lens.
        Default is 0.0.

        *Unit: arcsec*

    interpret_m_total_mass: boolean
        Indicates how to interpret the mass variable "m". If true
        the mass is interpreted as the total mass of the halo (good because it makes sense). If
        false it is interpreted as what the mass would have been within R200 of a an NFW that
        isn't truncated (good because it is easily compared with an NFW).


    use_case: str
        Due to an idyosyncratic behaviour of PyTorch, the NFW/TNFW profile
        specifically can't be both batchable and differentiable. You may select which version
        you wish to use by setting this parameter to one of: batchable, differentiable.

    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "mass": 1e13,
        "scale_radius": 1.0,
        "tau": 3.0,
    }

    def __init__(
        self,
        cosmology: Cosmology,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        mass: Optional[Union[Tensor, float]] = None,
        scale_radius: Optional[Union[Tensor, float]] = None,
        tau: Optional[Union[Tensor, float]] = None,
        s: float = 0.0,
        interpret_m_total_mass: bool = True,
        use_case="batchable",
        name: Optional[str] = None,
    ):
        """
        Initialize an instance of the TNFW lens class.

        """
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("mass", mass)
        self.add_param("scale_radius", scale_radius)
        self.add_param("tau", tau)
        self.s = s
        self.interpret_m_total_mass = interpret_m_total_mass
        if use_case == "batchable":
            self._F = self._F_batchable
        elif use_case == "differentiable":
            self._F = self._F_differentiable
        else:
            raise ValueError("use case should be one of: batchable, differentiable")

    @staticmethod
    def _F_batchable(x):
        """
        Helper method from Baltz et al. 2009 equation A.5
        """
        return torch.where(
            x == 1,
            torch.ones_like(x),
            (
                (1 / x.to(dtype=torch.cdouble)).arccos()
                / (x.to(dtype=torch.cdouble) ** 2 - 1).sqrt()
            ).abs(),
        )

    @staticmethod
    def _F_differentiable(x):
        f = torch.ones_like(x)
        f[x < 1] = torch.arctanh((1.0 - x[x < 1] ** 2).sqrt()) / (1.0 - x[x < 1] ** 2).sqrt()  # fmt: skip
        f[x > 1] = torch.arctan( (x[x > 1] ** 2 - 1.0).sqrt()) / (x[x > 1] ** 2 - 1.0).sqrt()  # fmt: skip
        return f

    @staticmethod
    def _L(x, tau):
        """
        Helper method from Baltz et al. 2009 equation A.6
        """
        return (x / (tau + (tau**2 + x**2).sqrt())).log()  # fmt: skip

    @unpack
    def get_concentration(
        self,
        *args,
        params: Optional[Packed] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the scale radius of the lens.
        This is the same formula used for the classic NFW profile.

        Parameters
        ----------
        z_l: Tensor
            Redshift of the lens.

            *Unit: unitless*

        x0: Tensor
            Center of lens position on x-axis.

            *Unit: arcsec*

        y0: Tensor
            Center of lens position on y-axis.

            *Unit: arcsec*

        mass: Optional[Tensor]
            Mass of the lens.

            *Unit: Msun*

        scale_radius: Optional[Tensor]
            Scale radius of the TNFW lens.

            *Unit: arcsec*

        tau: Optional[Tensor]
            Truncation scale. Ratio of truncation radius to scale radius.

            *Unit: unitless*

        params: dict
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The scale radius of the lens.

            *Unit: Mpc*

        """
        critical_density = self.cosmology.critical_density(z_l, params)
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        r_delta = (3 * mass / (4 * pi * DELTA * critical_density)) ** (1 / 3)  # fmt: skip
        return r_delta / (scale_radius * d_l * arcsec_to_rad)  # fmt: skip

    @unpack
    def get_truncation_radius(
        self,
        *args,
        params: Optional[Packed] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the truncation radius of the TNFW lens.

        Parameters
        ----------
        z_l: Tensor
            Redshift of the lens.

            *Unit: unitless*

        x0: Tensor
            Center of lens position on x-axis.

            *Unit: arcsec*

        y0: Tensor
            Center of lens position on y-axis.

            *Unit: arcsec*

        mass: Optional[Tensor]
            Mass of the lens.

            *Unit: Msun*

        scale_radius: Optional[Tensor]
            Scale radius of the TNFW lens.

            *Unit: arcsec*

        tau: Optional[Tensor]
            Truncation scale. Ratio of truncation radius to scale radius.

            *Unit: unitless*

        params: dictionary
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The truncation radius of the lens.

            *Unit: arcsec*

        """
        return tau * scale_radius

    @unpack
    def get_M0(
        self,
        *args,
        params: Optional[Packed] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the reference mass.
        This is an abstract reference mass used internally
        in the equations from Baltz et al. 2009.

        Parameters
        ----------
        z_l: Tensor
            Redshift of the lens.

            *Unit: unitless*

        x0: Tensor
            Center of lens position on x-axis.

            *Unit: arcsec*

        y0: Tensor
            Center of lens position on y-axis.

            *Unit: arcsec*

        mass: Optional[Tensor]
            Mass of the lens.

            *Unit: Msun*

        scale_radius: Optional[Tensor]
            Scale radius of the TNFW lens.

            *Unit: arcsec*

        tau: Optional[Tensor]
            Truncation scale. Ratio of truncation radius to scale radius.

            *Unit: unitless*

        params: dictionary
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The reference mass of the lens in Msun.

            *Unit: Msun*

        """
        if self.interpret_m_total_mass:
            return mass * (tau**2 + 1) ** 2 / (tau**2 * ((tau**2 - 1) * tau.log() + torch.pi * tau - (tau**2 + 1)))  # fmt: skip
        else:
            d_l = self.cosmology.angular_diameter_distance(z_l, params)
            return 4 * torch.pi * (scale_radius * d_l * arcsec_to_rad) ** 3 * self.get_scale_density(params)  # fmt: skip

    @unpack
    def get_scale_density(
        self,
        *args,
        params: Optional[Packed] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the scale density of the lens.

        Parameters
        ----------
        z_l: Tensor
            Redshift of the lens.

            *Unit: unitless*

        x0: Tensor
            Center of lens position on x-axis.

            *Unit: arcsec*

        y0: Tensor
            Center of lens position on y-axis.

            *Unit: arcsec*

        mass: Optional[Tensor]
            Mass of the lens.

            *Unit: Msun*

        scale_radius: Optional[Tensor]
            Scale radius of the TNFW lens.

            *Unit: arcsec*

        tau: Optional[Tensor]
            Truncation scale. Ratio of truncation radius to scale radius.

            *Unit: unitless*

        params: dict
            Dynamic parameter container.

        Returns
        --------
        Tensor
            The scale density of the lens.

            *Unit: Msun/Mpc^3*

        """
        c = self.get_concentration(params)
        return DELTA / 3 * self.cosmology.critical_density(z_l, params) * c**3 / ((1 + c).log() - c / (1 + c))  # fmt: skip

    @unpack
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional[Packed] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        TNFW convergence as given in Baltz et al. 2009.
        This is unitless since it is Sigma(x) / Sigma_crit.

        Parameters
        ----------
        z_l: Tensor
            Redshift of the lens.

            *Unit: unitless*

        x0: Tensor
            Center of lens position on x-axis.

            *Unit: arcsec*

        y0: Tensor
            Center of lens position on y-axis.

            *Unit: arcsec*

        mass: Optional[Tensor]
            Mass of the lens.

            *Unit: Msun*

        scale_radius: Optional[Tensor]
            Scale radius of the TNFW lens.

            *Unit: arcsec*

        tau: Optional[Tensor]
            Truncation scale. Ratio of truncation radius to scale radius.

            *Unit: unitless*

        params: dict
            Dynamic parameter container.

        Returns
        ---------
        Tensor
            Convergence at requested position.

            *Unit: unitless*

        """
        x, y = translate_rotate(x, y, x0, y0)
        r = (x**2 + y**2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        g = r / scale_radius
        F = self._F(g)
        L = self._L(g, tau)
        critical_density = self.cosmology.critical_surface_density(z_l, z_s, params)

        S = self.get_M0(params) / (2 * torch.pi * (scale_radius * d_l * arcsec_to_rad) ** 2)  # fmt: skip

        t2 = tau**2
        a1 = t2 / (t2 + 1) ** 2
        a2 = torch.where(g == 1, (t2 + 1) / 3.0, (t2 + 1) * (1 - F) / (g**2 - 1))  # fmt: skip
        a3 = 2 * F
        a4 = -torch.pi / (t2 + g**2).sqrt()
        a5 = (t2 - 1) * L / (tau * (t2 + g**2).sqrt())
        return a1 * (a2 + a3 + a4 + a5) * S / critical_density  # fmt: skip

    @unpack
    def mass_enclosed_2d(
        self,
        r: Tensor,
        z_s: Tensor,
        *args,
        params: Optional[Packed] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Total projected mass (Msun) within a radius r (arcsec).

        Parameters
        -----------
        z_l: Tensor
            Redshift of the lens.

            *Unit: unitless*

        x0: Tensor
            Center of lens position on x-axis.

            *Unit: arcsec*

        y0: Tensor
            Center of lens position on y-axis.

            *Unit: arcsec*

        mass: Optional[Tensor]
            Mass of the lens.

            *Unit: Msun*

        scale_radius: Optional[Tensor]
            Scale radius of the TNFW lens.

            *Unit: arcsec*

        tau: Optional[Tensor]
            Truncation scale. Ratio of truncation radius to scale radius.

            *Unit: unitless*

        params: dict
            Dynamic parameter container.

        Returns
        -------
        Tensor
            Integrated mass projected in infinite cylinder within radius r.

            *Unit: Msun*

        """
        g = r / scale_radius
        t2 = tau**2
        F = self._F(g)
        L = self._L(g, tau)
        a1 = t2 / (t2 + 1) ** 2
        a2 = (t2 + 1 + 2 * (g**2 - 1)) * F
        a3 = tau * torch.pi
        a4 = (t2 - 1) * tau.log()
        a5 = (t2 + g**2).sqrt() * (-torch.pi + (t2 - 1) * L / tau)  # fmt: skip
        S = self.get_M0(params)
        return S * a1 * (a2 + a3 + a4 + a5)

    @unpack
    def physical_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional[Packed] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Compute the physical deflection angle (arcsec) for this lens at
        the requested position. Note that the NFW/TNFW profile is more
        naturally represented as a physical deflection angle, this is
        easily internally converted to a reduced deflection angle.

        Parameters
        ----------
        z_l: Tensor
            Redshift of the lens.

            *Unit: unitless*

        x0: Tensor
            Center of lens position on x-axis.

            *Unit: arcsec*

        y0: Tensor
            Center of lens position on y-axis.

            *Unit: arcsec*

        mass: Optional[Tensor]
            Mass of the lens (Msun).

            *Unit: Msun*

        scale_radius: Optional[Tensor]
            Scale radius of the TNFW lens.

            *Unit: arcsec*

        tau: Optional[Tensor]
            Truncation scale. Ratio of truncation radius to scale radius.

            *Unit: unitless*

        params: dict
            Dynamic parameter container.

        Returns
        --------
        x_component: Tensor
            Deflection Angle in x-direction.

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle in y-direction.

            *Unit: arcsec*

        """
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        x, y = translate_rotate(x, y, x0, y0)
        r = (x**2 + y**2).sqrt() + self.s
        theta = torch.arctan2(y, x)

        # The below actually equally comes from eq 2.13 in Meneghetti notes
        dr = self.mass_enclosed_2d(r, z_s, params) / (
            r * d_l * arcsec_to_rad
        )  # note dpsi(u)/du = 2x*dpsi(x)/dx when u = x^2  # fmt: skip
        S = 4 * G_over_c2 * rad_to_arcsec
        return S * dr * theta.cos(), S * dr * theta.sin()

    @unpack
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional[Packed] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        mass: Optional[Tensor] = None,
        scale_radius: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the lensing potential.
        Note that this is not a unitless potential!
        This is the potential as given in Baltz et al. 2009.

        TODO: convert to dimensionless potential.

        Parameters
        -----------
        z_l: Tensor
            Redshift of the lens.

            *Unit: unitless*

        x0: Tensor
            Center of lens position on x-axis.

            *Unit: arcsec*

        y0: Tensor
            Center of lens position on y-axis.

            *Unit: arcsec*

        mass: Optional[Tensor]
            Mass of the lens.

            *Unit: Msun*

        scale_radius: Optional[Tensor]
            Scale radius of the TNFW lens.

            *Unit: arcsec*

        tau: Optional[Tensor]
            Truncation scale. Ratio of truncation radius to scale radius.

            *Unit: unitless*

        params: dict
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """
        x, y = translate_rotate(x, y, x0, y0)
        r = (x**2 + y**2).sqrt() + self.s
        g = r / scale_radius
        t2 = tau**2
        u = g**2
        F = self._F(g)
        L = self._L(g, tau)
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        d_s = self.cosmology.angular_diameter_distance(z_s, params)
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s, params)

        # fmt: off
        S = 2 * self.get_M0(params) * G_over_c2 * (d_ls / d_s) / (d_l * arcsec_to_rad**2)
        a1 = 1 / (t2 + 1) ** 2
        a2 = 2 * torch.pi * t2 * (tau - (t2 + u).sqrt() + tau * (tau + (t2 + u).sqrt()).log())
        a3 = 2 * (t2 - 1) * tau * (t2 + u).sqrt() * L
        a4 = t2 * (t2 - 1) * L**2
        a5 = 4 * t2 * (u - 1) * F
        a6 = t2 * (t2 - 1) * (1 / g.to(dtype=torch.cdouble)).arccos().abs() ** 2
        a7 = t2 * ((t2 - 1) * tau.log() - t2 - 1) * u.log()
        a8 = t2 * ((t2 - 1) * tau.log() * (4 * tau).log() + 2 * (tau / 2).log() - 2 * tau * (tau - torch.pi) * (2 * tau).log())

        return S * a1 * (a2 + a3 + a4 + a5 + a6 + a7 - a8)

    # fmt: on
