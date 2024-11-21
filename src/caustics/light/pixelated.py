# mypy: disable-error-code="union-attr"
from typing import Optional, Union, Annotated

from torch import Tensor
from caskade import forward, Param

from ..utils import interp2d
from .base import Source, NameType

__all__ = ("Pixelated",)


class Pixelated(Source):
    """
    ``Pixelated`` is a subclass of the abstract class ``Source``.
    It represents the brightness profile of source
    with a pixelated grid of intensity values.

    This class provides a concrete implementation
    of the ``brightness`` method required by the ``Source`` superclass.
    In this implementation, brightness is determined
    by interpolating values from the provided source image.

    Attributes
    ----------
    x0 : Tensor, optional
        The x-coordinate of the source image's center.

       *Unit: arcsec*

    y0 : Tensor, optional
        The y-coordinate of the source image's center.

        *Unit: arcsec*

    image : Tensor, optional
        The source image from which brightness values will be interpolated.

        *Unit: flux*

    pixelscale : Tensor, optional
        The pixelscale of the source image in the lens plane.

        *Unit: arcsec/pixel*

    shape : Tuple of ints, optional
        The shape of the source image.

    """

    def __init__(
        self,
        image: Annotated[
            Optional[Tensor],
            "The source image from which brightness values will be interpolated.",
            True,
        ] = None,
        x0: Annotated[
            Optional[Union[Tensor, float]],
            "The x-coordinate of the source image's center.",
            True,
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]],
            "The y-coordinate of the source image's center.",
            True,
        ] = None,
        pixelscale: Annotated[
            Optional[Union[Tensor, float]],
            "The pixelscale of the source image in the lens plane",
            True,
            "arcsec/pixel",
        ] = None,
        shape: Annotated[
            Optional[tuple[int, ...]], "The shape of the source image."
        ] = None,
        name: NameType = None,
    ):
        """
        Constructs the `Pixelated` object with the given parameters.

        Parameters
        ----------
        name : str
            The name of the source.

        x0 : Tensor, optional
            The x-coordinate of the source image's center.

            *Unit: arcsec*

        y0 : Tensor, optional
            The y-coordinate of the source image's center.

            *Unit: arcsec*

        image : Tensor, optional
            The source image from which brightness values will be interpolated.

        pixelscale : Tensor, optional
            The pixelscale of the source image in the lens plane.

            *Unit: arcsec/pixel*

        shape : Tuple of ints, optional
            The shape of the source image.

        """
        if image is not None and image.ndim not in [2, 3]:
            raise ValueError(
                f"image must be 2D or 3D (channels first). Received a {image.ndim}D tensor)"
            )
        elif shape is not None and len(shape) not in [2, 3]:
            raise ValueError(
                f"shape must be specify 2D or 3D tensors. Received shape={shape}"
            )
        super().__init__(name=name)
        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.image = Param("image", image, shape, units="flux")
        self.pixelscale = Param(
            "pixelscale", pixelscale, units="arcsec/pixel", valid=(0, None)
        )

    @forward
    def brightness(
        self,
        x,
        y,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        image: Annotated[Tensor, "Param"],
        pixelscale: Annotated[Tensor, "Param"],
        padding_mode: str = "zeros",
    ):
        """
        Implements the `brightness` method for `Pixelated`.
        The brightness at a given point is determined
        by interpolating values from the source image.

        Parameters
        ----------
        x : Tensor
            The x-coordinate(s) at which to calculate the source brightness.
            This could be a single value or a tensor of values.

            *Unit: arcsec*

        y : Tensor
            The y-coordinate(s) at which to calculate the source brightness.
            This could be a single value or a tensor of values.

            *Unit: arcsec*

        params : Packed, optional
            A dictionary containing additional parameters that might be required to
            calculate the brightness.

        Returns
        -------
        Tensor
            The brightness of the source at the given coordinate(s).
            The brightness is determined by interpolating values
            from the source image.

            *Unit: flux*

        """
        fov_x = pixelscale * image.shape[1]
        fov_y = pixelscale * image.shape[0]
        return interp2d(
            image,
            (x - x0).view(-1) / fov_x * 2,
            (y - y0).view(-1) / fov_y * 2,  # make coordinates bounds at half the fov
            padding_mode=padding_mode,
        ).reshape(x.shape)
