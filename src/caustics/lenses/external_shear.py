# mypy: disable-error-code="dict-item"
from typing import Optional, Union, Annotated

from torch import Tensor
import torch
from ..utils import translate_rotate
from .base import ThinLens, CosmologyType, NameType, ZLType
from ..parametrized import unpack
from ..packed import Packed

__all__ = ("ExternalShear", "ExternalShear_angle")


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
            Optional[Union[Tensor, float]],
            "Shear component in the vertical direction",
            True,
        ] = None,
        gamma_2: Annotated[
            Optional[Union[Tensor, float]],
            "Shear component in the y=-x direction",
            True,
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
        x, y = translate_rotate(x, y, x0, y0)
        # Meneghetti eq 3.83
        # TODO, why is it not:
        # th = (x**2 + y**2).sqrt() + self.s
        # a1 = x/th + x * gamma_1 + y * gamma_2
        # a2 = y/th + x * gamma_2 - y * gamma_1
        a1 = x * gamma_1 + y * gamma_2
        a2 = x * gamma_2 - y * gamma_1

        return a1, a2  # I'm not sure but I think no derotation necessary

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
        ax, ay = self.reduced_deflection_angle(x, y, z_s, params)
        x, y = translate_rotate(x, y, x0, y0)
        return 0.5 * (x * ax + y * ay)

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
        raise NotImplementedError("convergence undefined for external shear")


class ExternalShear_angle(ThinLens):
    """
    Represents an external shear effect in a gravitational lensing system. Parametrized using a magnitude and rotation angle.

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

    gamma, theta: Optional[Union[Tensor, float]]
        Shear magnitude and angle.

        *Unit: unitless*

    Notes
    ------
    The shear components gamma_1 and gamma_2 represent an external shear, a gravitational
    distortion that can be caused by nearby structures outside of the main lens galaxy.
    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "gamma": 0.0,
        "theta": 0.0,
        "gamma_1": 0.0,
        "gamma_2": 0.0,
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
        gamma: Annotated[
            Optional[Union[Tensor, float]], "Shear magnitude", True
        ] = None,
        theta: Annotated[
            Optional[Union[Tensor, float]], "Shear rotation angle", True
        ] = None,
        s: Annotated[
            float, "Softening length for the elliptical power-law profile"
        ] = 0.0,
        name: NameType = None,
    ):
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("gamma", gamma)
        self.add_param("theta", theta)
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
        gamma: Optional[Tensor] = None,
        theta: Optional[Tensor] = None,
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
        gamma_1 = gamma * torch.cos(2 * theta)
        gamma_2 = gamma * torch.sin(2 * theta)
        x, y = translate_rotate(x, y, x0, y0)
        # Meneghetti eq 3.83
        # TODO, why is it not:
        # th = (x**2 + y**2).sqrt() + self.s
        # a1 = x/th + x * gamma_1 + y * gamma_2
        # a2 = y/th + x * gamma_2 - y * gamma_1
        a1 = x * gamma_1 + y * gamma_2
        a2 = x * gamma_2 - y * gamma_1

        return a1, a2  # I'm not sure but I think no derotation necessary

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
        gamma: Optional[Tensor] = None,
        theta: Optional[Tensor] = None,
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
        gamma_1 = gamma * torch.cos(2 * theta)
        gamma_2 = gamma * torch.sin(2 * theta)
        ax, ay = self.reduced_deflection_angle(x, y, z_s, params)
        x, y = translate_rotate(x, y, x0, y0)
        return 0.5 * (x * ax + y * ay)

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
        gamma: Optional[Tensor] = None,
        theta: Optional[Tensor] = None,
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
        raise NotImplementedError("convergence undefined for external shear")
