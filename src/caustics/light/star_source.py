# mypy: disable-error-code="operator,union-attr"
from typing import Optional, Union, Annotated

from caskade import forward, Param

from .base import Source, NameType
from . import func
from ..backend_obj import ArrayLike

__all__ = ("StarSource",)


class StarSource(Source):
    """
    `Star` is a subclass of the abstract class `Source`.
    It represents a star source in a gravitational lensing system.

    The Star profile is meant to describe individual light sources.

    Attributes
    -----------
    x0: Optional[ArrayLike]
        The x-coordinate of the Star source's center.

        *Unit: arcsec*

    y0: Optional[ArrayLike]
        The y-coordinate of the Star source's center.

        *Unit: arcsec*

    theta_s: Optional[ArrayLike]
        The radius of the star.

        *Unit: arcsec*

    Ie: Optional[ArrayLike]
        The intensity at the center of the star.

        *Unit: flux*

    gamma: Optional[ArrayLike]
        The linear limb darkening coefficient.

        *Unit: unitless*

    """

    def __init__(
        self,
        x0: Annotated[
            Optional[Union[ArrayLike, float]],
            "The x-coordinate of the star source's center",
            True,
        ] = None,
        y0: Annotated[
            Optional[Union[ArrayLike, float]],
            "The y-coordinate of the star source's center",
            True,
        ] = None,
        theta_s: Annotated[
            Optional[Union[ArrayLike, float]],
            "The radius of the star source",
            True,
        ] = None,
        Ie: Annotated[
            Optional[Union[ArrayLike, float]],
            "The intensity at the effective radius",
            True,
        ] = None,
        gamma: Annotated[
            Optional[Union[ArrayLike, float]],
            "The linear limb darkening coefficient",
            True,
        ] = None,
        name: NameType = None,
    ):
        """
        Constructs the `Star` object with the given parameters.

        Parameters
        ----------
        name: str
            The name of the source.

        x0: Optional[ArrayLike]
            The x-coordinate of the star source's center.

            *Unit: arcsec*

        y0: Optional[ArrayLike]
            The y-coordinate of the star source's center.

            *Unit: arcsec*

        theta_s: Optional[ArrayLike]
            The radius of the star.

            *Unit: arcsec*

        Ie: Optional[ArrayLike]
            The intensity at the center of the source.

            *Unit: flux*

        gamma: Optional[ArrayLike]
            The linear limb darkening coefficient.

            *Unit: unitless*

        """
        super().__init__(name=name)
        self.x0 = Param("x0", x0, shape=(), units="arcsec")
        self.y0 = Param("y0", y0, shape=(), units="arcsec")
        self.theta_s = Param(
            "theta_s", theta_s, shape=(), units="arcsec", valid=(0, None)
        )
        self.Ie = Param("Ie", Ie, shape=(), units="flux", valid=(0, None))
        self.gamma = Param("gamma", gamma, shape=(), units="unitless", valid=(0, 1))

    @forward
    def brightness(
        self,
        x,
        y,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        theta_s: Annotated[ArrayLike, "Param"],
        Ie: Annotated[ArrayLike, "Param"],
        gamma: Annotated[ArrayLike, "Param"],
    ):
        """
        Implements the `brightness` method for `star`. This method calculates the
        brightness of the source at the given point(s).

        Parameters
        ----------
        x: ArrayLike
            The x-coordinate(s) at which to calculate the source brightness.
            This could be a single value or a tensor of values.

            *Unit: arcsec*

        y: ArrayLike
            The y-coordinate(s) at which to calculate the source brightness.
            This could be a single value or a tensor of values.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            The brightness of the source at the given point(s).
            The output tensor has the same shape as `x` and `y`.

            *Unit: flux*

        """

        return func.brightness_star(x0, y0, theta_s, Ie, x, y, gamma)
