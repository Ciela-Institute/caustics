import torch
from torch.nn.functional import grid_sample

from .base import Base


class InterpolatedImage(Base):
    def __init__(
        self,
        image,
        thx0=0.0,
        thy0=0.0,
        th_scale=1.0,
        max_value=None,
        z_ref=None,
        cosmology=None,
        device=None,
    ):
        super().__init__(cosmology, device)

        if image.ndim != 4:
            raise ValueError("image must have four dimensions")

        self.image = torch.as_tensor(image, dtype=torch.float32, device=device)
        if max_value:
            self.image *= max_value / self.image.max()
        self.thx0 = torch.as_tensor(thx0, dtype=torch.float32, device=device)
        self.thy0 = torch.as_tensor(thy0, dtype=torch.float32, device=device)
        self.th_extent = torch.as_tensor(th_scale, dtype=torch.float32, device=device)

        if z_ref:
            self.z_ref = torch.as_tensor(z_ref, dtype=torch.float32, device=self.device)
            self.d_l_ref = self.cosmology.angular_diameter_dist(self.z_ref)
        else:
            self.z_ref = None

    def __call__(self, thx, thy, z):
        """
        Interpolates the image, rescaling the image's extent to a new redshift if
        necessary.
        """
        if self.z_ref:
            # Rescale angular size to new redshift
            d_l = self.cosmology.angular_diameter_dist(z)
            th_scale = self.th_extent * self.d_l_ref / d_l
        else:
            # Keep angular size fixed
            th_scale = self.th_extent

        grid = torch.stack((thx - self.thx0, thy - self.thy0), dim=-1).reshape(
            -1, *thx.shape[-2:], 2
        ) / (th_scale / 2)
        # Batch grid to match image batching
        grid = grid.repeat((len(self.image), 1, 1, 1))
        return grid_sample(self.image, grid, align_corners=False)
