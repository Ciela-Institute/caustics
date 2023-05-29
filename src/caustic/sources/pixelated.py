from typing import Optional

from torch import Tensor

from ..utils import interp2d
from .base import Source

__all__ = ("ImageSource",)


class ImageSource(Source):
    """
    `ImageSource` is a subclass of the abstract class `Source`. It represents a source in a strong 
    gravitational lensing system where the source is an image.
    
    This class provides a concrete implementation of the `brightness` method required by the `Source` 
    superclass. In this implementation, brightness is determined by interpolating values from the 
    provided source image.

    Attributes:
        thx0 (Optional[Tensor]): The x-coordinate of the source image's center. 
        thy0 (Optional[Tensor]): The y-coordinate of the source image's center.
        image (Optional[Tensor]): The source image from which brightness values will be interpolated.
        scale (Optional[Tensor]): The scale of the source image in the lens plane.
        image_shape (Optional[tuple[int, ...]]): The shape of the source image.
    """
    def __init__(
        self,
        name: str,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        image_shape: Optional[tuple[int, ...]] = None,
    ):
        """
        Constructs the `ImageSource` object with the given parameters. 

        Args:
            name (str): The name of the source.
            thx0 (Optional[Tensor]): The x-coordinate of the source image's center.
            thy0 (Optional[Tensor]): The y-coordinate of the source image's center.
            image (Optional[Tensor]): The source image from which brightness values will be interpolated.
            scale (Optional[Tensor]): The scale of the source image in the lens plane.
            image_shape (Optional[tuple[int, ...]]): The shape of the source image.
        """
        super().__init__(name)
        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("image", image, image_shape)
        self.add_param("scale", scale)

    def brightness(self, thx, thy, x):
        """
        Implements the `brightness` method for `ImageSource`. The brightness at a given point is 
        determined by interpolating values from the source image.

        Args:
            thx (Tensor): The x-coordinate(s) at which to calculate the source brightness. 
                This could be a single value or a tensor of values.
            thy (Tensor): The y-coordinate(s) at which to calculate the source brightness. 
                This could be a single value or a tensor of values.
            x (dict): A dictionary containing additional parameters that might be required to 
                calculate the brightness. 

        Returns:
            Tensor: The brightness of the source at the given coordinate(s). The brightness is 
            determined by interpolating values from the source image.
        """
        thx0, thy0, image, scale = self.unpack(x)
        return interp2d(
            image, (thx - thx0).view(-1) / scale, (thy - thy0).view(-1) / scale
        ).reshape(thx.shape)
