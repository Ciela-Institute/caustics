# mypy: disable-error-code="operator,dict-item"
from typing import Optional, Union, Annotated

from torch import Tensor

from .base import ThinLens, CosmologyType, NameType, ZLType
from ..parametrized import unpack
from ..packed import Packed
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

    z_l: Optional[Union[Tensor, float]]
        The redshift of the lens.

        *Unit: unitless*

    x0: Optional[Union[Tensor, float]]
        x-coordinate of the shear center in the lens plane.

        *Unit: arcsec*

    y0: Optional[Union[Tensor, float]]
        y-coordinate of the shear center in the lens plane.

        *Unit: arcsec*

    surface_density: Optional[Union[Tensor, float]]
        Surface density normalized by the critical surface density.

        *Unit: unitless*
    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "surface_density": 0.1,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZLType = None,
        x0: Annotated[
            Optional[Union[Tensor, float]],
            "x-coordinate of the shear center in the lens plane",
            True,
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]],
            "y-coordinate of the shear center in the lens plane",
            True,
        ] = None,
        surface_density: Annotated[
            Optional[Union[Tensor, float]], "Surface density", True
        ] = None,
        name: NameType = None,
    ):
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("surface_density", surface_density)

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
        surface_density: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculates the reduced deflection angle.

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

        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        x_component: Tensor
            Deflection Angle in x-direction.

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle in y-direction.

            *Unit: arcsec*

        """
        return func.reduced_deflection_angle_mass_sheet(x0, y0, surface_density, x, y)

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
        surface_density: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # Meneghetti eq 3.81
        return func.potential_mass_sheet(x0, y0, surface_density, x, y)

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
        surface_density: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # Essentially by definition
        return func.convergence_mass_sheet(surface_density, x)
