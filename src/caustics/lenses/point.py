# mypy: disable-error-code="operator,dict-item"
from typing import Optional, Union, Annotated

from torch import Tensor
from caskade import forward, Param

from .base import ThinLens, CosmologyType, NameType, ZLType
from . import func

__all__ = ("Point",)


class Point(ThinLens):
    """
    Class representing a point mass lens in strong gravitational lensing.

    Attributes
    ----------
    name: str
        The name of the point lens.

    cosmology: Cosmology
        The cosmology used for calculations.

    z_l: Optional[Union[Tensor, float]]
        Redshift of the lens.

        *Unit: unitless*

    x0: Optional[Union[Tensor, float]]
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Optional[Union[Tensor, float]]
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    th_ein: Optional[Union[Tensor, float]]
        Einstein radius of the lens.

        *Unit: arcsec*

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*

    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "th_ein": 1.0,
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
        th_ein: Annotated[
            Optional[Union[Tensor, float]], "Einstein radius of the lens", True
        ] = None,
        s: Annotated[
            float, "Softening parameter to prevent numerical instabilities"
        ] = 0.0,
        name: NameType = None,
    ):
        """
        Initialize the Point class.

        Parameters
        ----------
        name: string
            The name of the point lens.

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

        th_ein: Optional[Tensor]
            Einstein radius of the lens.

            *Unit: arcsec*

        s: float
            Softening parameter to prevent numerical instabilities.

            *Unit: arcsec*

        """
        super().__init__(cosmology, z_l, name=name)

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.th_ein = Param("th_ein", th_ein, units="arcsec", valid=(0, None))
        self.s = s

    @forward
    def mass_to_rein(
        self, mass: Tensor, z_s: Tensor, z_l: Annotated[Tensor, "Param"]
    ) -> Tensor:
        """
        Convert mass to the Einstein radius.

        Parameters
        ----------
        mass: Tensor
            The mass of the lens

            *Unit: solar mass*

        Returns
        -------
        Tensor
            The Einstein radius.

            *Unit: arcsec*

        """

        Dls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)
        Dl = self.cosmology.angular_diameter_distance(z_l)
        Ds = self.cosmology.angular_diameter_distance(z_s)
        return func.mass_to_rein_point(mass, Dls, Dl, Ds)

    @forward
    def rein_to_mass(
        self, r: Tensor, z_s: Tensor, z_l: Annotated[Tensor, "Param"]
    ) -> Tensor:
        """
        Convert Einstein radius to mass.

        Parameters
        ----------
        r: Tensor
            The Einstein radius.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The mass of the lens

            *Unit: solar mass*

        """

        Dls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)
        Dl = self.cosmology.angular_diameter_distance(z_l)
        Ds = self.cosmology.angular_diameter_distance(z_s)
        return func.rein_to_mass_point(r, Dls, Dl, Ds)

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        th_ein: Annotated[Tensor, "Param"],
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the deflection angles.

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
            Deflection Angle in the x-direction.

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle in the y-direction.

            *Unit: arcsec*

        """
        return func.reduced_deflection_angle_point(x0, y0, th_ein, x, y, self.s)

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        th_ein: Annotated[Tensor, "Param"],
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
        return func.potential_point(x0, y0, th_ein, x, y, self.s)

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
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
        --------
        Tensor
            The convergence (dimensionless surface mass density).

            *Unit: unitless*

        """
        return func.convergence_point(x0, y0, x, y)
