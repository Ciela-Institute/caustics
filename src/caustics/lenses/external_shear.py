# mypy: disable-error-code="dict-item"
from typing import Optional, Union, Annotated, Literal
from warnings import warn

from torch import Tensor
import torch
from caskade import forward, Param

from .base import ThinLens, CosmologyType, NameType, ZLType
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
        parametrization: Literal["cartesian", "angular"] = "cartesian",
        s: Annotated[
            float, "Softening length for the elliptical power-law profile"
        ] = 0.0,
        name: NameType = None,
    ):
        super().__init__(cosmology, z_l, name=name)

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.gamma_1 = Param("gamma_1", gamma_1, units="unitless")
        self.gamma_2 = Param("gamma_2", gamma_2, units="unitless")
        self._parametrization = "cartesian"
        self.parametrization = parametrization
        self.s = s

    @property
    def parametrization(self) -> str:
        return self._parametrization

    @parametrization.setter
    def parametrization(self, value: str):
        if value not in ["cartesian", "angular"]:
            raise ValueError(
                f"Invalid parametrization: {value}. Must be 'cartesian' or 'angular'."
            )
        if value == "angular" and self._parametrization != "angular":
            self.gamma = Param("gamma", shape=self.gamma_1.shape, units="unitless")
            self.theta = Param("theta", shape=self.gamma_1.shape, units="radians")
            if self.gamma_1.static:
                warn(
                    f"Parameter {self.gamma_1.name} was static, value now overridden by new {value} parametrization. To remove this warning, have {self.gamma_1.name} be dynamic when changing parametrizations.",
                )
            self.gamma_1.value = lambda p: func.gamma_theta_to_gamma1(
                p["gamma"].value, p["theta"].value
            )
            if self.gamma_1.static:
                warn(
                    f"Parameter {self.gamma_2.name} was static, value now overridden by new {value} parametrization. To remove this warning, have {self.gamma_2.name} be dynamic when changing parametrizations.",
                )
            self.gamma_2.value = lambda p: func.gamma_theta_to_gamma2(
                p["gamma"].value, p["theta"].value
            )
            self.gamma_1.link(self.gamma)
            self.gamma_1.link(self.theta)
            self.gamma_2.link(self.gamma)
            self.gamma_2.link(self.theta)
        if value == "cartesian" and self._parametrization != "cartesian":
            try:
                if self.gamma.static or self.theta.static:
                    warn(
                        f"Parameter {self.gamma.name} and/or {self.theta.name} was static, value now overridden by new {value} parametrization. To remove this warning, have {self.gamma.name} and {self.theta.name} be dynamic when changing parametrizations.",
                    )
                self.gamma_1 = None
                self.gamma_2 = None
                del self.gamma
                del self.theta
            except AttributeError:
                pass

        self._parametrization = value

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        gamma_1: Annotated[Tensor, "Param"],
        gamma_2: Annotated[Tensor, "Param"],
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

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        gamma_1: Annotated[Tensor, "Param"],
        gamma_2: Annotated[Tensor, "Param"],
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

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
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
