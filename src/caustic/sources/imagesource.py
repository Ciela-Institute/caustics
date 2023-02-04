from ..utils import interpolate_image
from .base import Source

__all__ = ("ImageSource",)


class ImageSource(Source):
    def brightness(self, thx, thy, thx0, thy0, image, scale):
        return interpolate_image(thx, thy, thx0, thy0, image, scale)
