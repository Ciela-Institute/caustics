# mypy: disable-error-code="operator,dict-item"
from typing import Optional, Union, Annotated

from caskade import forward, Param

from .base import ThinLens, CosmologyType, NameType, ZType
from . import func
from ..backend_obj import ArrayLike

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

    z_l: Optional[Union[ArrayLike, float]]
        The lens redshift.

        *Unit: unitless*

    z_s: Optional[Union[ArrayLike, float]]
        The source redshift.

        *Unit: unitless*

    x0: Optional[Union[ArrayLike, float]]
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Optional[Union[ArrayLike, float]]
        The y-coordinate of the lens center.

    Rein: Optional[Union[ArrayLike, float]]
        The Einstein radius of the lens.

        *Unit: arcsec*

    s: float
        A smoothing factor, default is 0.0.

        *Unit: arcsec*

    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "Rein": 1.0,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZType = None,
        z_s: ZType = None,
        x0: Annotated[
            Optional[Union[ArrayLike, float]],
            "The x-coordinate of the lens center",
            True,
        ] = None,
        y0: Annotated[
            Optional[Union[ArrayLike, float]],
            "The y-coordinate of the lens center",
            True,
        ] = None,
        Rein: Annotated[
            Optional[Union[ArrayLike, float]], "The Einstein radius of the lens", True
        ] = None,
        s: Annotated[float, "A smoothing factor"] = 0.0,
        name: NameType = None,
    ):
        """
        Initialize the SIS lens model.
        """
        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        self.x0 = Param("x0", x0, shape=(), units="arcsec")
        self.y0 = Param("y0", y0, shape=(), units="arcsec")
        self.Rein = Param("Rein", Rein, shape=(), units="arcsec", valid=(0, None))
        self.s = s

    @forward
    def reduced_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        Rein: Annotated[ArrayLike, "Param"],
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculate the deflection angle of the SIS lens.

        Parameters
        ----------
        x: ArrayLike
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        x_component: ArrayLike
            Deflection Angle

            *Unit: arcsec*

        y_component: ArrayLike
            Deflection Angle

            *Unit: arcsec*

        """
        return func.reduced_deflection_angle_sis(x0, y0, Rein, x, y, self.s)

    @forward
    def potential(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        Rein: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Compute the lensing potential of the SIS lens.

        Parameters
        ----------
        x: ArrayLike
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            The lensing potential.

            *Unit: arcsec^2*

        """
        return func.potential_sis(x0, y0, Rein, x, y, self.s)

    @forward
    def convergence(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        Rein: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Calculate the projected mass density of the SIS lens.

        Parameters
        ----------
        x: ArrayLike
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            The projected mass density.

            *Unit: unitless*

        """
        return func.convergence_sis(x0, y0, Rein, x, y, self.s)
