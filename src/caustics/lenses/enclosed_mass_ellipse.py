# mypy: disable-error-code="operator,union-attr,dict-item"
from typing import Optional, Union, Annotated, Callable

from torch import Tensor
import torch

from ..utils import translate_rotate, derotate
from .base import ThinLens, CosmologyType, NameType, ZLType
from ..packed import Packed
from ..constants import G_over_c2

__all__ = ("Enclosed_Mass_Ellipse",)


class Enclosed_Mass_Ellipse(ThinLens):
    """
    A class for representing a lens with an enclosed mass profile. This generic
    lens profile can represent any lens with a mass distribution that can be
    described by a function that returns the enclosed mass as a function of
    radius. Note that this will be an axisymmetric lens.
    """

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

        # TODO: merge with enclosed mass in func version
        x, y = translate_rotate(x, y, x0, y0, phi)
        r = (x**2 / q + q * y**2).sqrt() + self.s
        alpha = 4 * G_over_c2 * self.enclosed_mass(r, p) / r
        theta = torch.atan2(y, x)
        ax = alpha * torch.cos(theta) * x / (q * r)
        ay = alpha * torch.sin(theta) * y * q / r
        return derotate(ax, ay)

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
        x, y = translate_rotate(x, y, x0, y0, phi)
        r = (x**2 / q + q * y**2).sqrt() + self.s
        critical_surface_density = self.cosmology.critical_surface_density(
            z_l, z_s, params
        )
        return (
            0.5
            * torch.func.grad(self.enclosed_mass)(r, p)
            / (r * torch.pi * critical_surface_density)
        )
