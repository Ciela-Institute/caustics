# mypy: disable-error-code="operator,union-attr"
from typing import Optional, Annotated, List

import torch

from .base import Source, NameType
from ..parametrized import unpack
from ..packed import Packed

__all__ = ("LightStack",)


class LightStack(Source):
    """
    ``LightStack`` is a subclass of the abstract class ``Source`` which takes
    the sum of multiple ``Source`` models to make a single brightness model.

    Attributes
    -----------
    light_models: List[Source]
        A list of light models to sum.

    """

    def __init__(
        self,
        light_models: Annotated[
            List[Source], "a list of light models to sum their brightnesses"
        ],
        name: NameType = None,
    ):
        """
        Constructs the ``LightStack`` object to sum multiple light models.

        Parameters
        ----------
        name: str
            The name of the source.

        light_models: List[Source]
            A list of light models to sum.

        """
        super().__init__(name=name)
        self.light_models = light_models
        for model in light_models:
            self.add_parametrized(model)

    @unpack
    def brightness(
        self,
        x,
        y,
        *args,
        params: Optional["Packed"] = None,
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

        """

        brightness = torch.zeros_like(x)
        for light_model in self.light_models:
            brightness += light_model.brightness(x, y, params=params, **kwargs)
        return brightness
