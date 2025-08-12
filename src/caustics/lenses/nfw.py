# mypy: disable-error-code="operator,union-attr,dict-item"
from typing import Optional, Union, Annotated

from torch import Tensor
from caskade import forward, Param

from .base import ThinLens, NameType, CosmologyType, ZType
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

    z_s: Optional[Tensor]
        Redshift of the source. Default is None.

        *Unit: unitless*

    x0: Optional[Tensor]
        x-coordinate of the lens center in the lens plane. Default is None.

        *Unit: arcsec*

    y0: Optional[Tensor]
        y-coordinate of the lens center in the lens plane. Default is None.

        *Unit: arcsec*

    mass: Optional[Tensor]
        Mass of the lens. Default is None.

        *Unit: Msun*

    c: Optional[Tensor]
        Concentration parameter of the lens. Default is None.

        *Unit: unitless*

    s: float
        Softening parameter to avoid singularities at the center of the lens. Default is 0.0.

        *Unit: arcsec*

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
        "mass": 1e13,
        "c": 5.0,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZType = None,
        z_s: ZType = None,
        x0: Annotated[
            Optional[Union[Tensor, float]], "X coordinate of the lens center", True
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]], "Y coordinate of the lens center", True
        ] = None,
        mass: Annotated[
            Optional[Union[Tensor, float]], "Mass of the lens", True
        ] = None,
        c: Annotated[
            Optional[Union[Tensor, float]], "Concentration parameter of the lens", True
        ] = None,
        s: Annotated[
            float,
            "Softening parameter to avoid singularities at the center of the lens",
        ] = 0.0,
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

        mass: Optional[Union[Tensor, float]]
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
        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.mass = Param("mass", mass, units="Msun")
        self.c = Param("c", c, units="unitless")
        self.s = s

    @forward
    def get_scale_radius(
        self,
        z_l: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        c: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Calculate the scale radius of the lens.

        Parameters
        ----------
        z_l: Tensor
            Redshift of the lens.

            *Unit: unitless*

        mass: Tensor
            Mass of the lens.

            *Unit: Msun*

        c: Tensor
            Concentration parameter of the lens.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The scale radius of the lens in Mpc.

            *Unit: Mpc*

        """
        critical_density = self.cosmology.critical_density(z_l)
        return func.scale_radius_nfw(critical_density, mass, c, DELTA)

    @forward
    def get_scale_density(
        self,
        z_l: Annotated[Tensor, "Param"],
        c: Annotated[Tensor, "Param"],
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

        Returns
        -------
        Tensor
            The scale density of the lens in solar masses per Mpc cubed.

            *Unit: Msun/Mpc^3*

        """
        critical_density = self.cosmology.critical_density(z_l)
        return func.scale_density_nfw(critical_density, c, DELTA)

    @forward
    def physical_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_l: Annotated[Tensor, "Param"],
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        c: Annotated[Tensor, "Param"],
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

        Returns
        -------
        x_component: Tensor
            The x-component of the reduced deflection angle.

            *Unit: arcsec*

        y_component: Tensor
            The y-component of the reduced deflection angle.

            *Unit: arcsec*

        """
        d_l = self.cosmology.angular_diameter_distance(z_l)
        critical_density = self.cosmology.critical_density(z_l)
        return func.physical_deflection_angle_nfw(
            x0, y0, mass, c, critical_density, d_l, x, y, DELTA=DELTA, s=self.s
        )

    @forward
    def reduced_deflection_angle(self, x, y, z_s, z_l):
        d_s = self.cosmology.angular_diameter_distance(z_s)
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)
        deflection_angle_x, deflection_angle_y = self.physical_deflection_angle(x, y)
        return func.reduced_from_physical_deflection_angle(
            deflection_angle_x, deflection_angle_y, d_s, d_ls
        )

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Annotated[Tensor, "Param"],
        z_l: Annotated[Tensor, "Param"],
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        c: Annotated[Tensor, "Param"],
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

        Returns
        -------
        Tensor
            The convergence (dimensionless surface mass density).

            *Unit: unitless*

        """
        critical_surface_density = self.cosmology.critical_surface_density(z_l, z_s)
        critical_density = self.cosmology.critical_density(z_l)
        d_l = self.cosmology.angular_diameter_distance(z_l)
        return func.convergence_nfw(
            critical_surface_density,
            critical_density,
            x0,
            y0,
            mass,
            c,
            x,
            y,
            d_l,
            DELTA=DELTA,
            s=self.s,
        )

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Annotated[Tensor, "Param"],
        z_l: Annotated[Tensor, "Param"],
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        c: Annotated[Tensor, "Param"],
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

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """
        critical_surface_density = self.cosmology.critical_surface_density(z_l, z_s)
        critical_density = self.cosmology.critical_density(z_l)
        d_l = self.cosmology.angular_diameter_distance(z_l)
        return func.potential_nfw(
            critical_surface_density,
            critical_density,
            x0,
            y0,
            mass,
            c,
            d_l,
            x,
            y,
            DELTA=DELTA,
            s=self.s,
        )
