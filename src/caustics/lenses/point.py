# mypy: disable-error-code="operator,dict-item"
from typing import Optional, Union, Annotated

from torch import Tensor

from .base import ThinLens, CosmologyType, NameType, ZLType
from ..parametrized import unpack
from ..packed import Packed
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

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("th_ein", th_ein)
        self.s = s

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
        th_ein: Optional[Tensor] = None,
        **kwargs,
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
        th_ein: Optional[Tensor] = None,
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
        return func.potential_point(x0, y0, th_ein, x, y, self.s)

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
        th_ein: Optional[Tensor] = None,
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
        --------
        Tensor
            The convergence (dimensionless surface mass density).

            *Unit: unitless*

        """
        return func.convergence_point(x0, y0, x, y)
