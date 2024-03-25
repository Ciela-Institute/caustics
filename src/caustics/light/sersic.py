# mypy: disable-error-code="operator,union-attr"
from typing import Optional, Union, Annotated

from torch import Tensor

from ..utils import to_elliptical, translate_rotate
from .base import Source, NameType
from ..parametrized import unpack
from ..packed import Packed

__all__ = ("Sersic",)


class Sersic(Source):
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
        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("n", n)
        self.add_param("Re", Re)
        self.add_param("Ie", Ie)
        self.s = s

        self.lenstronomy_k_mode = use_lenstronomy_k

    @unpack
    def brightness(
        self,
        x,
        y,
        *args,
        params: Optional["Packed"] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        n: Optional[Tensor] = None,
        Re: Optional[Tensor] = None,
        Ie: Optional[Tensor] = None,
        **kwargs,
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

        params: Packed, optional
            Dynamic parameter container.

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
        x, y = translate_rotate(x, y, x0, y0, phi)
        ex, ey = to_elliptical(x, y, q)
        e = (ex**2 + ey**2).sqrt() + self.s

        if self.lenstronomy_k_mode:
            k = 1.9992 * n - 0.3271
        else:
            k = 2 * n - 1 / 3 + 4 / 405 / n + 46 / 25515 / n**2  # fmt: skip

        exponent = -k * ((e / Re) ** (1 / n) - 1)
        return Ie * exponent.exp()
