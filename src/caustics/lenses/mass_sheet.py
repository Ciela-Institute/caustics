# mypy: disable-error-code="operator,dict-item"
from typing import Optional, Union, Annotated

from caskade import forward, Param

from ..backend_obj import ArrayLike
from .base import ThinLens, CosmologyType, NameType, ZType
from . import func

__all__ = ("MassSheet",)


class MassSheet(ThinLens):
    """
    Represents an external shear effect in a gravitational lensing system.

    Attributes
    ----------
    name: string
        Identifier for the lens instance.

    cosmology: Cosmology
        The cosmological model used for lensing calculations.

    z_l: Optional[Union[ArrayLike, float]]
        The redshift of the lens.

        *Unit: unitless*

    z_s : Optional[Union[ArrayLike, float]]
        The redshift of the source.

        *Unit: unitless*

    x0: Optional[Union[ArrayLike, float]]
        x-coordinate of the shear center in the lens plane.

        *Unit: arcsec*

    y0: Optional[Union[ArrayLike, float]]
        y-coordinate of the shear center in the lens plane.

        *Unit: arcsec*

    kappa: Optional[Union[ArrayLike, float]]
        Convergence. Surface density normalized by the critical surface density.

        *Unit: unitless*
    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "kappa": 0.1,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZType = None,
        z_s: ZType = None,
        x0: Annotated[
            Optional[Union[ArrayLike, float]],
            "x-coordinate of the shear center in the lens plane",
            True,
        ] = None,
        y0: Annotated[
            Optional[Union[ArrayLike, float]],
            "y-coordinate of the shear center in the lens plane",
            True,
        ] = None,
        kappa: Annotated[
            Optional[Union[ArrayLike, float]], "Surface density", True
        ] = None,
        name: NameType = None,
    ):
        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        self.x0 = Param("x0", x0, shape=(), units="arcsec")
        self.y0 = Param("y0", y0, shape=(), units="arcsec")
        self.kappa = Param("kappa", kappa, shape=(), units="unitless")

    @forward
    def reduced_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        kappa: Annotated[ArrayLike, "Param"],
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculates the reduced deflection angle.

        Parameters
        ----------
        x: ArrayLike
            x-coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            y-coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        x_component: ArrayLike
            Deflection Angle in x-direction.

            *Unit: arcsec*

        y_component: ArrayLike
            Deflection Angle in y-direction.

            *Unit: arcsec*

        """
        return func.reduced_deflection_angle_mass_sheet(x0, y0, kappa, x, y)

    @forward
    def potential(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        kappa: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        # Meneghetti eq 3.81
        return func.potential_mass_sheet(x0, y0, kappa, x, y)

    @forward
    def convergence(
        self,
        x: ArrayLike,
        y: ArrayLike,
        kappa: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        # Essentially by definition
        return func.convergence_mass_sheet(kappa, x)
