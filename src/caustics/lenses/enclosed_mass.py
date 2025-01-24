# mypy: disable-error-code="operator,union-attr,dict-item"
from typing import Optional, Union, Annotated, Callable

from torch import Tensor, pi
from caskade import forward, Param

from .base import ThinLens, CosmologyType, NameType, ZType
from .func import physical_deflection_angle_enclosed_mass, convergence_enclosed_mass

__all__ = ("EnclosedMass",)


class EnclosedMass(ThinLens):
    """
    A class for representing a lens with an enclosed mass profile. This generic
    lens profile can represent any lens with a mass distribution that can be
    described by a function that returns the enclosed mass as a function of
    radius.
    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "q": 1.0,
        "phi": 0.0,
        "p": 1.0,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        enclosed_mass: Callable,
        z_l: ZType = None,
        z_s: ZType = None,
        x0: Annotated[
            Optional[Union[Tensor, float]], "The x-coordinate of the lens center", True
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]], "The y-coordinate of the lens center", True
        ] = None,
        q: Annotated[
            Optional[Union[Tensor, float]], "The axis ratio of the lens", True
        ] = None,
        phi: Annotated[
            Optional[Union[Tensor, float]], "The position angle of the lens", True
        ] = None,
        p: Annotated[
            Optional[Union[Tensor, list[float]]],
            "parameters for the enclosed mass function",
            True,
        ] = None,
        s: Annotated[
            float, "Softening parameter to prevent numerical instabilities"
        ] = 0.0,
        name: NameType = None,
        **kwargs,
    ):
        """
        Initialize an enclosed mass lens.

        Parameters
        ----------
        name : str
            The name of the lens.

        cosmology : Cosmology
            The cosmology object that describes the Universe.

        enclosed_mass : callable
            A function that takes a radius and a set of parameters and returns the enclosed mass. Should be of the form ``enclosed_mass(r, p) -> M`` and returns units of Msun.

            *Unit: Msun*

        z_l : float
            The redshift of the lens.

            *Unit: unitless*

        z_s : float
            The redshift of the source.

            *Unit: unitless*

        x0 : float or Tensor, optional
            The x-coordinate of the lens center.

            *Unit: arcsec*

        y0 : float or Tensor, optional
            The y-coordinate of the lens center.

            *Unit: arcsec*

        q : float or Tensor, optional
            The axis ratio of the lens. ratio of semi-minor to semi-major axis (b/a).

            *Unit: unitless*

        phi : float or Tensor, optional
            The position angle of the lens.

            *Unit: radians*

        p : list[float] or Tensor, optional
            The parameters for the enclosed mass function.

            *Unit: user-defined*

        s : float, optional
            Softening parameter to prevent numerical instabilities.

            *Unit: arcsec*
        """
        super().__init__(cosmology, z_l, name=name, z_s=z_s, **kwargs)
        self.enclosed_mass = enclosed_mass

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.q = Param("q", q, units="unitless", valid=(0, 1))
        self.phi = Param("phi", phi, units="radians", valid=(0, pi), cyclic=True)
        self.p = Param("p", p, units="user-defined")

        self.s = s

    @forward
    def physical_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        q: Annotated[Tensor, "Param"],
        phi: Annotated[Tensor, "Param"],
        p: Annotated[Tensor, "Param"],
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the physical deflection angle of the lens at a given position.

        Parameters
        ----------
        x: Tensor
            The x-coordinate on the lens plane.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate on the lens plane.

            *Unit: arcsec*

        Returns
        -------
            The physical deflection angle at the given position. [Tensor, Tensor]

            *Unit: arcsec*
        """
        return physical_deflection_angle_enclosed_mass(
            x0, y0, q, phi, lambda r: self.enclosed_mass(r, p), x, y, self.s
        )

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError(
            "Potential is not implemented for enclosed mass profiles."
        )

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Annotated[Tensor, "Param"],
        z_l: Annotated[Tensor, "Param"],
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        q: Annotated[Tensor, "Param"],
        phi: Annotated[Tensor, "Param"],
        p: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Calculate the dimensionless convergence of the lens at a given position.

        Parameters
        ----------
        x: Tensor
            The x-coordinate on the lens plane.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate on the lens plane.

            *Unit: arcsec*

        Returns
        -------
            The dimensionless convergence at the given position. [Tensor]

            *Unit: unitless*
        """

        csd = self.cosmology.critical_surface_density(z_l, z_s)
        return convergence_enclosed_mass(
            x0,
            y0,
            q,
            phi,
            lambda r: self.enclosed_mass(r, p),
            x,
            y,
            csd,
            self.s,
        )
