# mypy: disable-error-code="operator,dict-item"
from typing import Optional, Union, Annotated

from caskade import forward, Param

from ..backend_obj import backend, ArrayLike, dtypeLike, deviceLike
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

    m: Union[ArrayLike, int, tuple[int]]
        Order of multipole(s).

    z_l: Optional[Union[ArrayLike, float]]
        The redshift of the lens.

    x0, y0: Optional[Union[ArrayLike, float]]
        Coordinates of the shear center in the lens plane.

    a_m: Optional[Union[ArrayLike, float]]
        Strength of multipole.

    phi_m: Optional[Union[ArrayLike, float]]
        Orientation of multipole.

    """

    _null_params = {"x0": 0.0, "y0": 0.0, "a_m": 1.0, "phi_m": 0.0, "m": 2}

    def __init__(
        self,
        cosmology: CosmologyType,
        m: Annotated[Union[ArrayLike, int, tuple[int]], "The Multipole moment(s) m"],
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
        a_m: Annotated[
            Optional[Union[ArrayLike, float]], "The amplitude of the multipole", True
        ] = None,
        phi_m: Annotated[
            Optional[Union[ArrayLike, float]],
            "The orientation angle of the multipole",
            True,
        ] = None,
        name: NameType = None,
    ):
        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.m = backend.as_array(m, dtype=backend.int32)
        assert backend.all(self.m >= 2).item(), "Multipole order must be >= 2"
        self.a_m = Param("a_m", a_m, self.m.shape, units="unitless")
        self.phi_m = Param(
            "phi_m",
            phi_m,
            self.m.shape,
            units="radians",
            valid=(0, 2 * backend.pi),
            cyclic=True,
        )

    def to(self, device: deviceLike = None, dtype: dtypeLike = None):
        """
        Move the lens to the specified device.

        Parameters
        ----------
        device: deviceLike
            The device to move the lens to.

        dtype: dtypeLike
            The data type to cast the lens to.

        Returns
        -------
        Multipole
            The lens object.

        """
        super().to(device, dtype)
        self.m = backend.to(self.m, device=device, dtype=backend.int32)
        return self

    @forward
    def reduced_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        a_m: Annotated[ArrayLike, "Param"],
        phi_m: Annotated[ArrayLike, "Param"],
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculate the deflection angle of the multipole.

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

        Equation (B11) and (B12) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

        """
        return func.reduced_deflection_angle_multipole(x0, y0, self.m, a_m, phi_m, x, y)

    @forward
    def potential(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        a_m: Annotated[ArrayLike, "Param"],
        phi_m: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Compute the lensing potential of the multiplane.

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

        Equation (B11) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

        """
        return func.potential_multipole(x0, y0, self.m, a_m, phi_m, x, y)

    @forward
    def convergence(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        a_m: Annotated[ArrayLike, "Param"],
        phi_m: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Calculate the projected mass density of the multipole.

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

        Equation (B10) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

        """
        return func.convergence_multipole(x0, y0, self.m, a_m, phi_m, x, y)
