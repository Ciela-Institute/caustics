# mypy: disable-error-code="operator,union-attr"
from typing import Optional, Union, Annotated

from torch import Tensor
from caskade import forward, Param

from ..utils import interp3d
from .base import Source, NameType

__all__ = ("PixelatedTime",)


class PixelatedTime(Source):
    """
    ``PixelatedTime`` is a subclass of the abstract class ``Source``. It
    represents the brightness profile of source with a pixelated grid of
    intensity values that also may vary over time.

    This class provides a concrete implementation of the ``brightness`` method
    required by the ``Source`` superclass. In this implementation, brightness is
    determined by interpolating values from the provided source image.

    Attributes
    ----------
    x0 : Tensor, optional
        The x-coordinate of the source image's center.

       *Unit: arcsec*

    y0 : Tensor, optional
        The y-coordinate of the source image's center.

        *Unit: arcsec*

    cube : Tensor, optional
        The source image cube from which brightness values will be interpolated.

        *Unit: flux*

    pixelscale : Tensor, optional
        The pixelscale of the source image in the lens plane.

        *Unit: arcsec/pixel*

    t_end : Tensor, optional
        The end time of the source image cube. Time in the cube is assumed to be
        in the range (0, t_end) in seconds.

        *Unit: seconds*

    shape : Tuple of ints, optional
        The shape of the source image and time dim.

    """

    def __init__(
        self,
        cube: Annotated[
            Optional[Tensor],
            "The source image cube from which brightness values will be interpolated.",
            True,
            "flux",
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
            False,
            "arcsec/pixel",
        ] = None,
        t_end: Annotated[
            Optional[Union[Tensor, float]],
            "The end time of the source image cube.",
            False,
            "seconds",
        ] = None,
        scale: Annotated[
            Optional[Union[Tensor, float]],
            "A scale factor to multiply by the image",
            True,
            "flux",
        ] = 1.0,
        shape: Annotated[
            Optional[tuple[int, ...]], "The shape of the source image."
        ] = None,
        name: NameType = None,
    ):
        """
        Constructs the `PixelatedTime` object with the given parameters.

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

        cube : Tensor, optional
            The source cube from which brightness values will be interpolated. Note the indexing of the cube should be cube[time][y][x]

        pixelscale : Tensor, optional
            The pixelscale of the source image in the lens plane.

            *Unit: arcsec/pixel*

        shape : Tuple of ints, optional
            The shape of the source image.

        """
        if cube is not None and cube.ndim not in [3, 4]:
            raise ValueError(
                f"image must be 3D or 4D (channels first). Received a {cube.ndim}D tensor)"
            )
        elif shape is not None and len(shape) not in [3, 4]:
            raise ValueError(
                f"shape must be specify 3D or 4D tensors. Received shape={shape}"
            )
        super().__init__(name=name)
        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.cube = Param("cube", cube, shape, units="flux")
        self.scale = Param("scale", scale, units="flux", valid=(0, None))
        self.pixelscale = pixelscale
        self.t_end = t_end

    @forward
    def brightness(
        self,
        x,
        y,
        t,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        cube: Annotated[Tensor, "Param"],
        scale: Annotated[Tensor, "Param"],
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

        t : Tensor
            The time coordinate(s) at which to calculate the source brightness.
            This could be a single value or a tensor of values.

            *Unit: seconds*

        Returns
        -------
        Tensor
            The brightness of the source at the given coordinate(s).
            The brightness is determined by interpolating values
            from the source image.

            *Unit: flux*

        """
        fov_x = self.pixelscale * cube.shape[2]
        fov_y = self.pixelscale * cube.shape[1]
        return interp3d(
            cube * scale,
            (x - x0).view(-1) / fov_x * 2,
            (y - y0).view(-1) / fov_y * 2,
            (t - self.t_end / 2).view(-1) / self.t_end * 2,
        ).reshape(x.shape)
