# mypy: disable-error-code="operator,union-attr,dict-item"
from typing import Optional, Union, Annotated, Literal

from torch import Tensor

from .base import ThinLens, NameType, CosmologyType, ZLType
from ..parametrized import unpack
from ..packed import Packed
from . import func

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

        *Unit: unitless*

    x0: Optional[Tensor]
        x-coordinate of the lens center in the lens plane. Default is None.

        *Unit: arcsec*

    y0: Optional[Tensor]
        y-coordinate of the lens center in the lens plane. Default is None.

        *Unit: arcsec*

    m: Optional[Tensor]
        Mass of the lens. Default is None.

        *Unit: Msun*

    c: Optional[Tensor]
        Concentration parameter of the lens. Default is None.

        *Unit: unitless*

    s: float
        Softening parameter to avoid singularities at the center of the lens. Default is 0.0.

        *Unit: arcsec*

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

            *Unit: unitless*

        x0: Optional[Union[Tensor, float]]
            x-coordinate of the lens center in the lens plane.
                Default is None.

            *Unit: arcsec*

        y0: Optional[Union[Tensor, float]]
            y-coordinate of the lens center in the lens plane.
                Default is None.

            *Unit: arcsec*

        m: Optional[Union[Tensor, float]]
            Mass of the lens. Default is None.

            *Unit: Msun*

        c: Optional[Union[Tensor, float]]
            Concentration parameter of the lens. Default is None.

            *Unit: unitless*

        s: float
            Softening parameter to avoid singularities at the center of the lens.
            Default is 0.0.

            *Unit: arcsec*

        """
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("m", m)
        self.add_param("c", c)
        self.s = s
        if use_case == "batchable":
            self._f = func._f_batchable_nfw
            self._h = func._h_batchable_nfw
            self._g = func._g_batchable_nfw
        elif use_case == "differentiable":
            self._f = func._f_differentiable_nfw
            self._h = func._h_differentiable_nfw
            self._g = func._g_differentiable_nfw
        else:
            raise ValueError("use case should be one of: batchable, differentiable")

    @unpack
    def get_scale_radius(
        self,
        *args,
        params: Optional[Packed] = None,
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

            *Unit: unitless*

        m: Tensor
            Mass of the lens.

            *Unit: Msun*

        c: Tensor
            Concentration parameter of the lens.

            *Unit: unitless*

        x: dict
            Dynamic parameter container.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The scale radius of the lens in Mpc.

            *Unit: Mpc*

        """
        critical_density = self.cosmology.critical_density(z_l, params)
        return func.scale_radius_nfw(critical_density, m, c, DELTA)

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

            *Unit: unitless*

        c: Tensor
            Concentration parameter of the lens.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The scale density of the lens in solar masses per Mpc cubed.

            *Unit: Msun/Mpc^3*

        """
        critical_density = self.cosmology.critical_density(z_l, params)
        return func.scale_density_nfw(critical_density, c, DELTA)

    @unpack
    def physical_deflection_angle(
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
        Compute the physical deflection angle.

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.

            *Unit: arcsec*

        y: Tensor
            y-coordinates in the lens plane.

            *Unit: arcsec*

        z_s: Tensor
            Redshifts of the sources.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        x_component: Tensor
            The x-component of the reduced deflection angle.

            *Unit: arcsec*

        y_component: Tensor
            The y-component of the reduced deflection angle.

            *Unit: arcsec*

        """
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        critical_density = self.cosmology.critical_density(z_l, params)
        return func.physical_deflection_angle_nfw(
            x0, y0, m, c, critical_density, d_l, x, y, _h=self._h, DELTA=DELTA, s=self.s
        )

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

            *Unit: arcsec*

        y: Tensor
            y-coordinates in the lens plane.

            *Unit: arcsec*

        z_s: Tensor
            Redshifts of the sources.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The convergence (dimensionless surface mass density).

            *Unit: unitless*

        """
        critical_surface_density = self.cosmology.critical_surface_density(
            z_l, z_s, params
        )
        critical_density = self.cosmology.critical_density(z_l, params)
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        return func.convergence_nfw(
            critical_surface_density,
            critical_density,
            x0,
            y0,
            m,
            c,
            x,
            y,
            d_l,
            _f=self._f,
            DELTA=DELTA,
            s=self.s,
        )

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

            *Unit: arcsec*

        y: Tensor
            y-coordinates in the lens plane.

            *Unit: arcsec*

        z_s: Tensor
            Redshifts of the sources.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """
        critical_surface_density = self.cosmology.critical_surface_density(
            z_l, z_s, params
        )
        critical_density = self.cosmology.critical_density(z_l, params)
        d_l = self.cosmology.angular_diameter_distance(z_l, params)
        return func.potential_nfw(
            critical_surface_density,
            critical_density,
            x0,
            y0,
            m,
            c,
            d_l,
            x,
            y,
            _g=self._g,
            DELTA=DELTA,
            s=self.s,
        )
