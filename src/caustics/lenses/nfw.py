# mypy: disable-error-code="operator,union-attr,dict-item"
from math import pi
from typing import Optional, Union, Annotated, Literal

import torch
from torch import Tensor

from ..constants import G_over_c2, arcsec_to_rad, rad_to_arcsec
from ..utils import translate_rotate
from .base import ThinLens, NameType, CosmologyType, ZLType
from ..parametrized import unpack
from ..packed import Packed

DELTA = 200.0

__all__ = ("NFW",)


class NFW(ThinLens):
    """
    NFW lens class. This class models a lens using the Navarro-Frenk-White (NFW) profile.
    The NFW profile is a spatial density profile of dark matter halo that arises in
    cosmological simulations.

    Attributes
    -----------
    z_l: Optional[Tensor]
        Redshift of the lens. Default is None.
    x0: Optional[Tensor]
        x-coordinate of the lens center in the lens plane. Default is None.
    y0: Optional[Tensor]
        y-coordinate of the lens center in the lens plane. Default is None.
    m: Optional[Tensor]
        Mass of the lens. Default is None.
    c: Optional[Tensor]
        Concentration parameter of the lens. Default is None.
    s: float
        Softening parameter to avoid singularities at the center of the lens. Default is 0.0.
    use_case: str
        Due to an idyosyncratic behaviour of PyTorch, the NFW/TNFW profile
        specifically can't be both batchable and differentiable. You may select which version
        you wish to use by setting this parameter to one of: batchable, differentiable.

    Methods
    -------
    get_scale_radius
        Returns the scale radius of the lens.
    get_scale_density
        Returns the scale density of the lens.
    get_convergence_s
        Returns the dimensionless surface mass density of the lens.
    _f
        Helper method for computing deflection angles.
    _g
        Helper method for computing lensing potential.
    _h
        Helper method for computing reduced deflection angles.
    deflection_angle_hat
        Computes the reduced deflection angle.
    deflection_angle
        Computes the deflection angle.
    convergence
        Computes the convergence (dimensionless surface mass density).
    potential
        Computes the lensing potential.
    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "m": 1e13,
        "c": 5.0,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZLType = None,
        x0: Annotated[
            Optional[Union[Tensor, float]], "X coordinate of the lens center", True
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]], "Y coordinate of the lens center", True
        ] = None,
        m: Annotated[Optional[Union[Tensor, float]], "Mass of the lens", True] = None,
        c: Annotated[
            Optional[Union[Tensor, float]], "Concentration parameter of the lens", True
        ] = None,
        s: Annotated[
            float,
            "Softening parameter to avoid singularities at the center of the lens",
        ] = 0.0,
        use_case: Annotated[
            Literal["batchable", "differentiable"], "the NFW/TNFW profile"
        ] = "batchable",
        name: NameType = None,
    ):
        """
        Initialize an instance of the NFW lens class.

        Parameters
        ----------
        name: str
            Name of the lens instance.
        cosmology: Cosmology
            An instance of the Cosmology class which contains
            information about the cosmological model and parameters.
        z_l: Optional[Union[Tensor, float]]
            Redshift of the lens. Default is None.
        x0: Optional[Union[Tensor, float]]
            x-coordinate of the lens center in the lens plane.
                Default is None.
        y0: Optional[Union[Tensor, float]]
            y-coordinate of the lens center in the lens plane.
                Default is None.
        m: Optional[Union[Tensor, float]]
            Mass of the lens. Default is None.
        c: Optional[Union[Tensor, float]]
            Concentration parameter of the lens. Default is None.
        s: float
            Softening parameter to avoid singularities at the center of the lens.
            Default is 0.0.
        """
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("m", m)
        self.add_param("c", c)
        self.s = s
        if use_case == "batchable":
            self._f = self._f_batchable
            self._h = self._h_batchable
            self._g = self._g_batchable
        elif use_case == "differentiable":
            self._f = self._f_differentiable
            self._h = self._h_differentiable
            self._g = self._g_differentiable
        else:
            raise ValueError("use case should be one of: batchable, differentiable")

    @unpack
    def get_scale_radius(
        self,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        m: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the scale radius of the lens.

        Parameters
        ----------
        z_l: Tensor
            Redshift of the lens.
        m: Tensor
            Mass of the lens.
        c: Tensor
            Concentration parameter of the lens.
        x: dict
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The scale radius of the lens in Mpc.
        """
        critical_density = self.cosmology.critical_density(z_l, params)
        r_delta = (3 * m / (4 * pi * DELTA * critical_density)) ** (1 / 3)  # fmt: skip
        return 1 / c * r_delta

    @unpack
    def get_scale_density(
        self,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        m: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the scale density of the lens.

        Parameters
        ----------
        z_l: Tensor
            Redshift of the lens.
        c: Tensor
            Concentration parameter of the lens.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The scale density of the lens in solar masses per Mpc cubed.
        """
        sigma_crit = self.cosmology.critical_density(z_l, params)
        return DELTA / 3 * sigma_crit * c**3 / ((1 + c).log() - c / (1 + c))  # fmt: skip

    @unpack
    def get_convergence_s(
        self,
        z_s,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        m: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the dimensionless surface mass density of the lens.

        Parameters
        ----------
        z_l: Tensor
            Redshift of the lens.
        z_s: Tensor
            Redshift of the source.
        m: Tensor
            Mass of the lens.
        c: Tensor
            Concentration parameter of the lens.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The dimensionless surface mass density of the lens.
        """
        critical_surface_density = self.cosmology.critical_surface_density(
            z_l, z_s, params
        )
        return  self.get_scale_density(params) * self.get_scale_radius(params) / critical_surface_density  # fmt: skip

    @staticmethod
    def _f_differentiable(x: Tensor) -> Tensor:
        """
        Helper method for computing deflection angles.

        Parameters
        ----------
        x: Tensor
            The scaled radius (xi / xi_0).

        Returns
        -------
        Tensor
            Result of the deflection angle computation.
        """
        # TODO: generalize beyond torch, or patch Tensor
        f = torch.zeros_like(x)
        f[x > 1] = 1 - 2 / (x[x > 1] ** 2 - 1).sqrt() * ((x[x > 1] - 1) / (x[x > 1] + 1)).sqrt().arctan()  # fmt: skip
        f[x < 1] = 1 - 2 / (1 - x[x < 1] ** 2).sqrt() * ((1 - x[x < 1]) / (1 + x[x < 1])).sqrt().arctanh()  # fmt: skip
        return f

    @staticmethod
    def _f_batchable(x: Tensor) -> Tensor:
        """
        Helper method for computing deflection angles.

        Parameters
        ----------
        x: Tensor
            The scaled radius (xi / xi_0).

        Returns
        -------
        Tensor
            Result of the deflection angle computation.
        """
        # TODO: generalize beyond torch, or patch Tensor
        # fmt: off
        return torch.where(
            x > 1,
            1 - 2 / (x**2 - 1).sqrt() * ((x - 1) / (x + 1)).sqrt().arctan(),
            torch.where(
                x < 1,
                1 - 2 / (1 - x**2).sqrt() * ((1 - x) / (1 + x)).sqrt().arctanh(),
                torch.zeros_like(x),  # where: x == 1
            ),
        )

    # fmt: on

    @staticmethod
    def _g_differentiable(x: Tensor) -> Tensor:
        """
        Helper method for computing lensing potential.

        Parameters
        ----------
        x: Tensor
            The scaled radius (xi / xi_0).

        Returns
        -------
        Tensor
            Result of the lensing potential computation.
        """
        # TODO: generalize beyond torch, or patch Tensor
        term_1 = (x / 2).log() ** 2
        term_2 = torch.zeros_like(x)
        term_2[x > 1] = (1 / x[x > 1]).arccos() ** 2  # fmt: skip
        term_2[x < 1] = -(1 / x[x < 1]).arccosh() ** 2  # fmt: skip
        return term_1 + term_2

    @staticmethod
    def _g_batchable(x: Tensor) -> Tensor:
        """
        Helper method for computing lensing potential.

        Parameters
        ----------
        x: Tensor
            The scaled radius (xi / xi_0).

        Returns
        -------
        Tensor
            Result of the lensing potential computation.
        """
        # TODO: generalize beyond torch, or patch Tensor
        term_1 = (x / 2).log() ** 2
        term_2 = torch.where(
            x > 1,
            (1 / x).arccos() ** 2,
            torch.where(
                x < 1,
                -(1 / x).arccosh() ** 2,
                torch.zeros_like(x),  # where: x == 1
            ),
        )
        return term_1 + term_2

    @staticmethod
    def _h_differentiable(x: Tensor) -> Tensor:
        """
        Helper method for computing reduced deflection angles.

        Parameters
        ----------
        x: Tensor
            The scaled radius (xi / xi_0).

        Returns
        -------
        Tensor
            Result of the reduced deflection angle computation.
        """
        term_1 = (x / 2).log()
        term_2 = torch.ones_like(x)
        term_2[x > 1] = (1 / x[x > 1]).arccos() * 1 / (x[x > 1] ** 2 - 1).sqrt()  # fmt: skip
        term_2[x < 1] = (1 / x[x < 1]).arccosh() * 1 / (1 - x[x < 1] ** 2).sqrt()  # fmt: skip
        return term_1 + term_2

    @staticmethod
    def _h_batchable(x: Tensor) -> Tensor:
        """
        Helper method for computing reduced deflection angles.

        Parameters
        ----------
        x: Tensor
            The scaled radius (xi / xi_0).

        Returns
        -------
        Tensor
            Result of the reduced deflection angle computation.
        """
        term_1 = (x / 2).log()
        term_2 = torch.where(
            x > 1,
            (1 / x).arccos() * 1 / (x**2 - 1).sqrt(),  # fmt: skip
            torch.where(
                x < 1, (1 / x).arccosh() * 1 / (1 - x**2).sqrt(), torch.ones_like(x)  # fmt: skip
            ),
        )
        return term_1 + term_2

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
        m: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the reduced deflection angle.

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.
        y: Tensor
            y-coordinates in the lens plane.
        z_s: Tensor
            Redshifts of the sources.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        tuple[Tensor, Tensor]
            The reduced deflection angles in the x and y directions.
        """
        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        scale_radius = self.get_scale_radius(params)
        xi = d_l * th * arcsec_to_rad
        r = xi / scale_radius

        deflection_angle = 16 * pi * G_over_c2 * self.get_scale_density(params) * scale_radius**3 * self._h(r) * rad_to_arcsec / xi  # fmt: skip

        ax = deflection_angle * x / th
        ay = deflection_angle * y / th
        d_s = self.cosmology.angular_diameter_distance(z_s, params)
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s, params)
        return ax * d_ls / d_s, ay * d_ls / d_s  # fmt: skip

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
        m: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the convergence (dimensionless surface mass density).

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.
        y: Tensor
            y-coordinates in the lens plane.
        z_s: Tensor
            Redshifts of the sources.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The convergence (dimensionless surface mass density).
        """
        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        scale_radius = self.get_scale_radius(params)
        xi = d_l * th * arcsec_to_rad
        r = xi / scale_radius  # xi / xi_0
        convergence_s = self.get_convergence_s(z_s, params)
        return 2 * convergence_s * self._f(r) / (r**2 - 1)  # fmt: skip

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
        m: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the lensing potential.

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.
        y: Tensor
            y-coordinates in the lens plane.
        z_s: Tensor
            Redshifts of the sources.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The lensing potential.
        """
        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        scale_radius = self.get_scale_radius(params)
        xi = d_l * th * arcsec_to_rad
        r = xi / scale_radius  # xi / xi_0
        convergence_s = self.get_convergence_s(z_s, params)
        return 2 * convergence_s * self._g(r) * scale_radius**2 / (d_l**2 * arcsec_to_rad**2)  # fmt: skip
