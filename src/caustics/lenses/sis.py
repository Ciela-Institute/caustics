# mypy: disable-error-code="operator,dict-item"
from typing import Optional, Union, Annotated

from torch import Tensor

from .base import ThinLens, CosmologyType, NameType, ZLType
from ..parametrized import unpack
from ..packed import Packed
from . import func

__all__ = ("SIS",)


class SIS(ThinLens):
    """
    A class representing the Singular Isothermal Sphere (SIS) model.
    This model inherits from the base `ThinLens` class.

    Attributes
    ----------
    name: str
        The name of the SIS lens.

    cosmology: Cosmology
        An instance of the Cosmology class.

    z_l: Optional[Union[Tensor, float]]
        The lens redshift.

        *Unit: unitless*

    x0: Optional[Union[Tensor, float]]
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Optional[Union[Tensor, float]]
        The y-coordinate of the lens center.

    th_ein: Optional[Union[Tensor, float]]
        The Einstein radius of the lens.

        *Unit: arcsec*

    s: float
        A smoothing factor, default is 0.0.

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
            Optional[Union[Tensor, float]], "The x-coordinate of the lens center", True
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]], "The y-coordinate of the lens center", True
        ] = None,
        th_ein: Annotated[
            Optional[Union[Tensor, float]], "The Einstein radius of the lens", True
        ] = None,
        s: Annotated[float, "A smoothing factor"] = 0.0,
        name: NameType = None,
    ):
        """
        Initialize the SIS lens model.
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
        Calculate the deflection angle of the SIS lens.

        Parameters
        ----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        x_component: Tensor
            Deflection Angle

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle

            *Unit: arcsec*

        """
        return func.reduced_deflection_angle_sis(x0, y0, th_ein, x, y, self.s)

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
        Compute the lensing potential of the SIS lens.

        Parameters
        ----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """
        return func.potential_sis(x0, y0, th_ein, x, y, self.s)

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
        Calculate the projected mass density of the SIS lens.

        Parameters
        ----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The projected mass density.

            *Unit: unitless*

        """
        return func.convergence_sis(x0, y0, th_ein, x, y, self.s)
