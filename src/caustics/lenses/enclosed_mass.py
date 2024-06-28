# mypy: disable-error-code="operator,union-attr,dict-item"
from typing import Optional, Union, Annotated, Callable

from torch import Tensor

from .base import ThinLens, CosmologyType, NameType, ZLType
from ..parametrized import unpack
from ..packed import Packed
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
        z_l: ZLType = None,
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
        super().__init__(cosmology, z_l, name=name, **kwargs)
        self.enclosed_mass = enclosed_mass

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("p", p)

        self.s = s

    @unpack
    def physical_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional[Packed] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        p: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the physical deflection angle of the lens at a given position.

        Parameters
        ----------
        name : str
            The name of the lens.

        cosmology : Cosmology
            The cosmology object that describes the Universe.

        z_l : float
            The redshift of the lens.

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

        Returns
        -------
            The physical deflection angle at the given position. [Tensor, Tensor]

            *Unit: arcsec*
        """
        return physical_deflection_angle_enclosed_mass(
            x0, y0, q, phi, lambda r: self.enclosed_mass(r, p), x, y, self.s
        )

    @unpack
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional[Packed] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        p: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError(
            "Potential is not implemented for enclosed mass profiles."
        )

    @unpack
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional[Packed] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        p: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the dimensionless convergence of the lens at a given position.

        Parameters
        ----------
        name : str
            The name of the lens.

        cosmology : Cosmology
            The cosmology object that describes the Universe.

        z_l : float
            The redshift of the lens.

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

        Returns
        -------
            The dimensionless convergence at the given position. [Tensor]

            *Unit: unitless*
        """

        csd = self.cosmology.critical_surface_density(z_l, z_s, params)
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
