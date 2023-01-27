from ..interpolate_image import interpolate_image
from .base import AbstractSource


class ImageSource(AbstractSource):
    def brightness(self, thx, thy, thx0, thy0, image, scale):
        return interpolate_image(thx, thy, thx0, thy0, image, scale)
