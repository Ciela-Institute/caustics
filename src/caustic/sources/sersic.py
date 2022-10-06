import torch

from ..cosmology import AbstractCosmology
from ..utils import flip_axis_ratio, to_elliptical, translate_rotate
from .base import AbstractSource


class Sersic(AbstractSource):
    def __init__(
        self,
        thx0,
        thy0,
        phi,
        q,
        index,
        th_e,
        I_e,
        z_ref,
        cosmology: AbstractCosmology,
        device: torch.device,
    ):
        super().__init__(cosmology, device)
        self.thx0 = torch.as_tensor(thx0, dtype=torch.float32, device=device)
        self.thy0 = torch.as_tensor(thy0, dtype=torch.float32, device=device)
        self.phi = torch.as_tensor(phi, dtype=torch.float32, device=device)
        self.q = torch.as_tensor(q, dtype=torch.float32, device=device)
        self.index = torch.as_tensor(index, dtype=torch.float32, device=device)
        self.th_e = torch.as_tensor(th_e, dtype=torch.float32, device=device)
        self.I_e = torch.as_tensor(I_e, dtype=torch.float32, device=device)
        if z_ref:
            self.z_ref = torch.as_tensor(z_ref, dtype=torch.float32, device=self.device)
            self.d_l_ref = self.cosmology.angular_diameter_dist(self.z_ref)
        else:
            self.z_ref = None

    def brightness(self, thx, thy, z=None):
        q, phi = flip_axis_ratio(self.q, self.phi)
        thx, thy = translate_rotate(thx, thy, -phi, self.thx0, self.thy0)
        ex, ey = to_elliptical(thx, thy, q)
        e = (ex**2 + ey**2).sqrt()
        k = 2 * self.index - 1 / 3 + 4 / 405 / self.index + 46 / 25515 / self.index**2

        if self.z_ref:
            # Rescale angular size to new redshift
            d_l = self.cosmology.angular_diameter_dist(z)
            th_e = self.th_e * self.d_l_ref / d_l
        else:
            # Keep angular size fixed
            th_e = self.th_e

        exponent = -k * ((e / th_e) ** (1 / self.index) - 1)
        return self.I_e * exponent.exp()
