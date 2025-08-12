# mypy: disable-error-code="dict-item"
from typing import Optional, Union, Annotated, Literal
from warnings import warn

from torch import Tensor
import torch
from caskade import forward, Param

from .base import ThinLens, CosmologyType, NameType, ZType
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

    z_s: Optional[Union[Tensor, float]]
        The redshift of the source.

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
        z_l: ZType = None,
        z_s: ZType = None,
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
        **kwargs,
    ):
        super().__init__(cosmology=cosmology, z_l=z_l, name=name, z_s=z_s)

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.gamma_1 = Param("gamma_1", gamma_1, units="unitless")
        self.gamma_2 = Param("gamma_2", gamma_2, units="unitless")
        self._parametrization = "cartesian"
        self.parametrization = parametrization
        if self.parametrization == "angular":
            self.gamma.value = kwargs.get("gamma", None)
            self.phi.value = kwargs.get("phi", None)
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
            self.gamma = Param(
                "gamma",
                shape=self.gamma_1.shape if self.gamma_1.static else (),
                units="unitless",
            )
            self.phi = Param(
                "phi",
                shape=self.gamma_1.shape if self.gamma_1.static else (),
                units="radians",
            )
            if self.gamma_1.static:
                warn(
                    f"Parameter {self.gamma_1.name} was static, value now overridden by new {value} parametrization. To remove this warning, have {self.gamma_1.name} be dynamic when changing parametrizations.",
                )
            self.gamma_1.value = lambda p: func.gamma_phi_to_gamma1(
                p["gamma"].value, p["phi"].value
            )
            if self.gamma_2.static:
                warn(
                    f"Parameter {self.gamma_2.name} was static, value now overridden by new {value} parametrization. To remove this warning, have {self.gamma_2.name} be dynamic when changing parametrizations.",
                )
            self.gamma_2.value = lambda p: func.gamma_phi_to_gamma2(
                p["gamma"].value, p["phi"].value
            )
            self.gamma_1.link(self.gamma)
            self.gamma_1.link(self.phi)
            self.gamma_2.link(self.gamma)
            self.gamma_2.link(self.phi)
        if value == "cartesian" and self._parametrization != "cartesian":
            self.gamma_1.value = None
            self.gamma_2.value = None
            try:
                if self.gamma.static:
                    warn(
                        f"Parameter {self.gamma.name} was static, value now overridden by new {value} parametrization. To remove this warning, have {self.gamma.name} be dynamic when changing parametrizations.",
                    )
                del self.gamma
            except AttributeError:
                pass
            try:
                if self.phi.static:
                    warn(
                        f"Parameter {self.phi.name} was static, value now overridden by new {value} parametrization. To remove this warning, have {self.phi.name} be dynamic when changing parametrizations.",
                    )
                del self.phi
            except AttributeError:
                pass

        self._parametrization = value

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
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
