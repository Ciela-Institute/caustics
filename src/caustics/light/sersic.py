# mypy: disable-error-code="operator,union-attr"
from typing import Optional, Union, Annotated

from torch import Tensor, pi
from caskade import forward, Param

from .base import Source, NameType
from . import func
from ..angle_mixin import Angle_Mixin

__all__ = ("Sersic",)


class Sersic(Angle_Mixin, Source):
    """
    `Sersic` is a subclass of the abstract class `Source`.
    It represents a source in a strong gravitational lensing system
    that follows a Sersic profile, a mathematical function that describes
    how the intensity I of a galaxy varies with distance r from its center.

    The Sersic profile is often used to describe
    elliptical galaxies and spiral galaxies' bulges.

    Attributes
    -----------
    x0: Optional[Tensor]
        The x-coordinate of the Sersic source's center.

        *Unit: arcsec*

    y0: Optional[Tensor]
        The y-coordinate of the Sersic source's center.

        *Unit: arcsec*

    q: Optional[Tensor]
        The axis ratio of the Sersic source.

        *Unit: unitless*

    phi: Optional[Tensor]
        The orientation of the Sersic source (position angle).

        *Unit: radians*

    n: Optional[Tensor]
        The Sersic index, which describes the degree of concentration of the source.

        *Unit: unitless*

    Re: Optional[Tensor]
        The scale length of the Sersic source.

        *Unit: arcsec*

    Ie: Optional[Tensor]
        The intensity at the effective radius.

        *Unit: flux*

    s: float
        A small constant for numerical stability.

        *Unit: arcsec*

    lenstronomy_k_mode: bool
        A flag indicating whether to use lenstronomy to compute the value of k.


    """

    def __init__(
        self,
        x0: Annotated[
            Optional[Union[Tensor, float]],
            "The x-coordinate of the Sersic source's center",
            True,
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]],
            "The y-coordinate of the Sersic source's center",
            True,
        ] = None,
        q: Annotated[
            Optional[Union[Tensor, float]], "The axis ratio of the Sersic source", True
        ] = None,
        phi: Annotated[
            Optional[Union[Tensor, float]],
            "The orientation of the Sersic source (position angle)",
            True,
        ] = None,
        n: Annotated[
            Optional[Union[Tensor, float]],
            "The Sersic index, which describes the degree of concentration of the source",
            True,
        ] = None,
        Re: Annotated[
            Optional[Union[Tensor, float]],
            "The scale length of the Sersic source",
            True,
        ] = None,
        Ie: Annotated[
            Optional[Union[Tensor, float]],
            "The intensity at the effective radius",
            True,
        ] = None,
        s: Annotated[float, "A small constant for numerical stability"] = 0.0,
        use_lenstronomy_k: Annotated[
            bool,
            "A flag indicating whether to use lenstronomy to compute the value of k.",
        ] = False,
        angle_system: str = "q_phi",
        e1: Optional[Union[Tensor, float]] = None,
        e2: Optional[Union[Tensor, float]] = None,
        c1: Optional[Union[Tensor, float]] = None,
        c2: Optional[Union[Tensor, float]] = None,
        name: NameType = None,
    ):
        """
        Constructs the `Sersic` object with the given parameters.

        Parameters
        ----------
        name: str
            The name of the source.

        x0: Optional[Tensor]
            The x-coordinate of the Sersic source's center.

            *Unit: arcsec*

        y0: Optional[Tensor]
            The y-coordinate of the Sersic source's center.

            *Unit: arcsec*

        q: Optional[Tensor]
            The axis ratio of the Sersic source.

            *Unit: unitless*

        phi: Optional[Tensor]
            The orientation of the Sersic source.

            *Unit: radians*

        n: Optional[Tensor]
            The Sersic index, which describes the degree of concentration of the source.

            *Unit: unitless*

        Re: Optional[Tensor]
            The scale length of the Sersic source.

            *Unit: arcsec*

        Ie: Optional[Tensor]
            The intensity at the effective radius.

            *Unit: flux*

        s: float
            A small constant for numerical stability.

            *Unit: arcsec*

        use_lenstronomy_k: bool
            A flag indicating whether to use lenstronomy to compute the value of k.


        """
        super().__init__(name=name)
        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.q = Param("q", q, units="unitless", valid=(0, 1))
        self.phi = Param("phi", phi, units="radians", valid=(0, pi), cyclic=True)
        self.n = Param("n", n, units="unitless", valid=(0.36, 10))
        self.Re = Param("Re", Re, units="arcsec", valid=(0, None))
        self.Ie = Param("Ie", Ie, units="flux", valid=(0, None))
        self.s = s

        self.lenstronomy_k_mode = use_lenstronomy_k
        self.angle_system = angle_system
        if self.angle_system == "e1_e2":
            self.e1 = e1
            self.e2 = e2
        elif self.angle_system == "c1_c2":
            self.c1 = c1
            self.c2 = c2

    @forward
    def brightness(
        self,
        x,
        y,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        q: Annotated[Tensor, "Param"],
        phi: Annotated[Tensor, "Param"],
        n: Annotated[Tensor, "Param"],
        Re: Annotated[Tensor, "Param"],
        Ie: Annotated[Tensor, "Param"],
    ):
        """
        Implements the `brightness` method for `Sersic`. The brightness at a given point is
        determined by the Sersic profile formula.

        Parameters
        ----------
        x: Tensor
            The x-coordinate(s) at which to calculate the source brightness.
            This could be a single value or a tensor of values.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate(s) at which to calculate the source brightness.
            This could be a single value or a tensor of values.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The brightness of the source at the given point(s).
            The output tensor has the same shape as `x` and `y`.

            *Unit: flux*

        Notes
        -----
        The Sersic profile is defined as: I(r) = Ie * exp(-k * ((r / r_e)^(1/n) - 1)),
        where Ie is the intensity at the effective radius r_e, n is the Sersic index
        that describes the concentration of the source, and k is a parameter that
        depends on n. In this implementation, we use elliptical coordinates ex and ey,
        and the transformation from Cartesian coordinates is handled by `to_elliptical`.
        The value of k can be calculated in two ways, controlled by `lenstronomy_k_mode`.
        If `lenstronomy_k_mode` is True, we use the approximation from Lenstronomy,
        otherwise, we use the approximation from Ciotti & Bertin (1999).
        """

        if self.lenstronomy_k_mode:
            k = func.k_lenstronomy(n)
        else:
            k = func.k_sersic(n)

        return func.brightness_sersic(x0, y0, q, phi, n, Re, Ie, x, y, k, self.s)
