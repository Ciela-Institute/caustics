from ..interpolatedimage import InterpolatedImage
from .base import AbstractSource


class ImageSource(AbstractSource):
    def __init__(
        self,
        image,
        thx0=0.0,
        thy0=0.0,
        th_scale=1.0,
        max_brightness=None,
        z_ref=None,
        cosmology=None,
        device=None,
    ):
        super().__init__(cosmology, device)
        self.interpolated_image = InterpolatedImage(
            image, thx0, thy0, th_scale, max_brightness, z_ref, cosmology, device
        )

    def brightness(self, thx, thy, z=None):
        return self.interpolated_image(thx, thy, z)
