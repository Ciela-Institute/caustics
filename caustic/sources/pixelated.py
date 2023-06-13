from typing import Optional

from torch import Tensor

from ..utils import interp2d
from .base import Source

__all__ = ("Pixelated",)


class Pixelated(Source):
    """
    `Pixelated` is a subclass of the abstract class `Source`. It represents a source in a strong 
    gravitational lensing system where the source is an image.
    
    This class provides a concrete implementation of the `brightness` method required by the `Source` 
    superclass. In this implementation, brightness is determined by interpolating values from the 
    provided source image.

    Attributes:
        x0 (Optional[Tensor]): The x-coordinate of the source image's center. 
        y0 (Optional[Tensor]): The y-coordinate of the source image's center.
        image (Optional[Tensor]): The source image from which brightness values will be interpolated.
        pixelscale (Optional[Tensor]): The pixelscale of the source image in the lens plane in units of arcsec/pixel.
        image_shape (Optional[tuple[int, ...]]): The shape of the source image.
    """
    def __init__(
        self,
        name: str,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        image: Optional[Tensor] = None, 
        pixelscale: Optional[Tensor] = None,
        image_shape: Optional[tuple[int, ...]] = None,
    ):
        """
        Constructs the `Pixelated` object with the given parameters. 

        Args:
            name (str): The name of the source.
            x0 (Optional[Tensor]): The x-coordinate of the source image's center.
            y0 (Optional[Tensor]): The y-coordinate of the source image's center.
            image (Optional[Tensor]): The source image from which brightness values will be interpolated.
            pixelscale (Optional[Tensor]): The pixelscale of the source image in the lens plane in units of arcsec/pixel.
            image_shape (Optional[tuple[int, ...]]): The shape of the source image.
        """
        super().__init__(name)
        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("image", image, image_shape)
        self.add_param("pixelscale", pixelscale)

    def brightness(self, x, y, params: Optional["Packed"]):
        """
        Implements the `brightness` method for `Pixelated`. The brightness at a given point is 
        determined by interpolating values from the source image.

        Args:
            x (Tensor): The x-coordinate(s) at which to calculate the source brightness. 
                This could be a single value or a tensor of values.
            y (Tensor): The y-coordinate(s) at which to calculate the source brightness. 
                This could be a single value or a tensor of values.
            P (Optional[Packed]): A dictionary containing additional parameters that might be required to 
                calculate the brightness. 

        Returns:
            Tensor: The brightness of the source at the given coordinate(s). The brightness is 
            determined by interpolating values from the source image.
        """
        x0, y0, image, pixelscale = self.unpack(params)
        return interp2d(
            image, (x - x0).view(-1) / pixelscale, (y - y0).view(-1) / pixelscale
        ).reshape(x.shape)
