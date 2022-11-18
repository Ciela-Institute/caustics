import torch
from torch.nn.functional import grid_sample

from .base import Base


class InterpolatedImage(Base):
    def __init__(self, fov, device=None):
        super().__init__(device)
        self.fov = fov

    def __call__(self, thx, thy, thx0, thy0, image):
        """
        Shifts and interpolates the image.
        """
        if image.ndim != 4:
            raise ValueError("image must have four dimensions")

        # Batch grid to match image batching
        grid = torch.stack((thx - thx0, thy - thy0), dim=-1).reshape(
            -1, *thx.shape[-2:], 2
        ) / (self.fov / 2)
        grid = grid.repeat((len(image), 1, 1, 1))
        return grid_sample(image, grid, align_corners=False)

# Old code for rescaling image size with redshift:
# if z_ref:
#     self.z_ref = torch.as_tensor(z_ref, dtype=torch.float32, device=self.device)
#     self.d_l_ref = self.cosmology.angular_diameter_dist(self.z_ref)
# else:
#     self.z_ref = None

# if self.z_ref:
#     # Rescale angular size to new redshift
#     d_l = self.cosmology.angular_diameter_dist(z)
#     th_scale = self.th_extent * self.d_l_ref / d_l
# else:
#     # Keep angular size fixed
#     th_scale = self.th_extent

