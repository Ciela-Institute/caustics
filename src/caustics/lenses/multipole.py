# mypy: disable-error-code="operator,dict-item"
from typing import Optional, Union, Annotated

import torch
from torch import Tensor, pi
from caskade import forward, Param

from .base import ThinLens, CosmologyType, NameType, ZType
from . import func

__all__ = ("Multipole",)


class Multipole(ThinLens):
    """
    Represents a multipole effect in a gravitational lensing system.

    Attributes
    ----------
    name: str
        Identifier for the lens instance.
    cosmology: Cosmology
        The cosmological model used for lensing calculations.
    m: Union[Tensor, int, tuple[int]]
        Order of multipole(s).
    z_l: Optional[Union[Tensor, float]]
        The redshift of the lens.
    x0, y0: Optional[Union[Tensor, float]]
        Coordinates of the shear center in the lens plane.
    a_m: Optional[Union[Tensor, float]]
        Strength of multipole.
    phi_m: Optional[Union[Tensor, float]]
        Orientation of multiple.

    """

    _null_params = {"x0": 0.0, "y0": 0.0, "a_m": 1.0, "phi_m": 0.0, "m": 2}

    def __init__(
        self,
        cosmology: CosmologyType,
        m: Annotated[Union[Tensor, int, tuple[int]], "The Multipole moment(s) m"],
        z_l: ZType = None,
        z_s: ZType = None,
        x0: Annotated[
            Optional[Union[Tensor, float]], "The x-coordinate of the lens center", True
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]], "The y-coordinate of the lens center", True
        ] = None,
        a_m: Annotated[
            Optional[Union[Tensor, float]], "The amplitude of the multipole", True
        ] = None,  # TODO change to true axis ratio
        phi_m: Annotated[
            Optional[Union[Tensor, float]],
            "The orientation angle of the multipole",
            True,
        ] = None,
        name: NameType = None,
    ):
        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.m = torch.as_tensor(m, dtype=torch.int32)
        assert torch.all(self.m >= 2).item(), "Multipole order must be >= 2"
        self.a_m = Param("a_m", a_m, self.m.shape, units="unitless")
        self.phi_m = Param(
            "phi_m",
            phi_m,
            self.m.shape,
            units="radians",
            valid=(0, 2 * pi),
            cyclic=True,
        )

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        """
        Move the lens to the specified device.

        Parameters
        ----------
        device: torch.device
            The device to move the lens to.

        dtype: torch.dtype
            The data type to cast the lens to.

        Returns
        -------
        Multipole
            The lens object.

        """
        super().to(device, dtype)
        self.m = self.m.to(device, torch.int32)
        return self

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        a_m: Annotated[Tensor, "Param"],
        phi_m: Annotated[Tensor, "Param"],
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the deflection angle of the multipole.

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

        Equation (B11) and (B12) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

        """
        return func.reduced_deflection_angle_multipole(x0, y0, self.m, a_m, phi_m, x, y)

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        a_m: Annotated[Tensor, "Param"],
        phi_m: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Compute the lensing potential of the multiplane.

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

        Equation (B11) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

        """
        return func.potential_multipole(x0, y0, self.m, a_m, phi_m, x, y)

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        a_m: Annotated[Tensor, "Param"],
        phi_m: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Calculate the projected mass density of the multipole.

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

        Equation (B10) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

        """
        return func.convergence_multipole(x0, y0, self.m, a_m, phi_m, x, y)
