# mypy: disable-error-code="dict-item"
from typing import Optional, Union, Annotated

from torch import Tensor
import torch

from .base import ThinLens, CosmologyType, NameType, ZLType
from ..parametrized import unpack
from ..packed import Packed
from . import func

__all__ = ("ExternalShear",)


class ExternalShear(ThinLens):
    """
    Represents an external shear effect in a gravitational lensing system.

    Attributes
    ----------
    name: str
        Identifier for the lens instance.

    cosmology: Cosmology
        The cosmological model used for lensing calculations.

    z_l: Optional[Union[Tensor, float]]
        The redshift of the lens.

        *Unit: unitless*

    x0, y0: Optional[Union[Tensor, float]]
        Coordinates of the shear center in the lens plane.

        *Unit: arcsec*

    gamma_1, gamma_2: Optional[Union[Tensor, float]]
        Shear components.

        *Unit: unitless*

    Notes
    ------
    The shear components gamma_1 and gamma_2 represent an external shear, a gravitational
    distortion that can be caused by nearby structures outside of the main lens galaxy.
    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "gamma_1": 0.1,
        "gamma_2": 0.1,
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
        gamma_1: Annotated[
            Optional[Union[Tensor, float]], "Shear component in the x-direction", True
        ] = None,
        gamma_2: Annotated[
            Optional[Union[Tensor, float]], "Shear component in the y-direction", True
        ] = None,
        s: Annotated[
            float, "Softening length for the elliptical power-law profile"
        ] = 0.0,
        name: NameType = None,
    ):
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("gamma_1", gamma_1)
        self.add_param("gamma_2", gamma_2)
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
        gamma_1: Optional[Tensor] = None,
        gamma_2: Optional[Tensor] = None,
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
        return func.reduced_deflection_angle_external_shear(
            x0, y0, gamma_1, gamma_2, x, y
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
        gamma_1: Optional[Tensor] = None,
        gamma_2: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculates the lensing potential.

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
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """
        return func.potential_external_shear(x0, y0, gamma_1, gamma_2, x, y)

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
        gamma_1: Optional[Tensor] = None,
        gamma_2: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        The convergence is undefined for an external shear.

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
        Tensor
            Convergence for an external shear.

            *Unit: unitless*

        Raises
        ------
        NotImplementedError
            This method is not implemented as the convergence is not defined
            for an external shear.
        """
        return torch.zeros_like(x)
