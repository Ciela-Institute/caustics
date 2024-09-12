# mypy: disable-error-code="operator,union-attr"
from typing import Optional, Union, Annotated

from torch import Tensor

from .base import Source, NameType
from ..parametrized import unpack
from ..packed import Packed
from . import func

__all__ = ("PointSource",)


class PointSource(Source):
    """
    `Point` is a subclass of the abstract class `Source`.
    It represents a point source in a gravitational lensing system.

    The Point profile is meant to describe individual light sources
    whose angular extent is much less than the Einstein radius of the lens.

    Attributes
    -----------
    x0: Optional[Tensor]
        The x-coordinate of the Point source's center.

        *Unit: arcsec*

    y0: Optional[Tensor]
        The y-coordinate of the Point source's center.

        *Unit: arcsec*

    Ie: Optional[Tensor]
        The intensity at the point.

        *Unit: flux*

    s: float
        A small constant for numerical stability.

        *Unit: arcsec*



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
        theta_s: Annotated[
            Optional[Union[Tensor, float]],
            "The radius of the point source",
            True,
        ] = None,
        Ie: Annotated[
            Optional[Union[Tensor, float]],
            "The intensity at the effective radius",
            True,
        ] = None,
        s: Annotated[float, "A small constant for numerical stability"] = 0.0,
        name: NameType = None,
    ):
        """
        Constructs the `Point` object with the given parameters.

        Parameters
        ----------
        name: str
            The name of the source.

        x0: Optional[Tensor]
            The x-coordinate of the Point source's center.

            *Unit: arcsec*

        y0: Optional[Tensor]
            The y-coordinate of the Point source's center.

            *Unit: arcsec*


        Ie: Optional[Tensor]
            The intensity at the Point.

            *Unit: flux*

        s: float
            A small constant for numerical stability.

            *Unit: arcsec*


        """
        super().__init__(name=name)
        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("theta_s", theta_s)
        self.add_param("Ie", Ie)
        self.s = s

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
        **kwargs,
    ):
        """
        Implements the `brightness` method for `Point`. The brightness at a given point is
        determined by the value at the Source's location.

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

        # return func.brightness_sersic(x0, y0, q, phi, n, Re, Ie, x, y, k, self.s)
        return func.brightness_point(x0, y0, theta_s, Ie, x, y, self.s)

