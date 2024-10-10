# mypy: disable-error-code="operator,union-attr"
from typing import Optional, Union, Annotated

from torch import Tensor

from .base import Source, NameType
from ..parametrized import unpack
from ..packed import Packed
from . import func

__all__ = ("StarSource",)


class StarSource(Source):
    """
    `Star` is a subclass of the abstract class `Source`.
    It represents a star source in a gravitational lensing system.

    The Star profile is meant to describe individual light sources.

    Attributes
    -----------
    x0: Optional[Tensor]
        The x-coordinate of the Star source's center.

        *Unit: arcsec*

    y0: Optional[Tensor]
        The y-coordinate of the Star source's center.

        *Unit: arcsec*

    theta_s: Optional[Tensor]
        The radius of the star.

        *Unit: arcsec*

    Ie: Optional[Tensor]
        The intensity at the center of the star.

        *Unit: flux*

    gamma: Optional[Tensor]
        The linear limb darkening coefficient.

        *Unit: unitless*

    """

    def __init__(
        self,
        x0: Annotated[
            Optional[Union[Tensor, float]],
            "The x-coordinate of the star source's center",
            True,
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]],
            "The y-coordinate of the star source's center",
            True,
        ] = None,
        theta_s: Annotated[
            Optional[Union[Tensor, float]],
            "The radius of the star source",
            True,
        ] = None,
        Ie: Annotated[
            Optional[Union[Tensor, float]],
            "The intensity at the effective radius",
            True,
        ] = None,
        gamma: Annotated[
            Optional[Union[Tensor, float]],
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

        x0: Optional[Tensor]
            The x-coordinate of the star source's center.

            *Unit: arcsec*

        y0: Optional[Tensor]
            The y-coordinate of the star source's center.

            *Unit: arcsec*

        theta_s: Optional[Tensor]
            The radius of the star.

            *Unit: arcsec*

        Ie: Optional[Tensor]
            The intensity at the center of the source.

            *Unit: flux*

        gamma: Optional[Tensor]
            The linear limb darkening coefficient.

            *Unit: unitless*

        """
        super().__init__(name=name)
        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("theta_s", theta_s)
        self.add_param("Ie", Ie)
        self.add_param("gamma", gamma)

    @unpack
    def brightness(
        self,
        x,
        y,
        *args,
        params: Optional["Packed"] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        theta_s: Optional[Tensor] = None,
        Ie: Optional[Tensor] = None,
        gamma: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Implements the `brightness` method for `star`. This method calculates the
        brightness of the source at the given point(s).

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

        """

        return func.brightness_star(x0, y0, theta_s, Ie, x, y, gamma)
