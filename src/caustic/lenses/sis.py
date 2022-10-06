from math import pi

import torch

from ..constants import G, arcsec_to_rad, c_km_s, rad_to_arcsec
from .base import AbstractLens


class SIS(AbstractLens):
    def __init__(self, thx0, thy0, sigma_v, cosmology=None, device=None):
        """
        Args:
            sigma_v: velocity dispersion [km/s]
        """
        super().__init__(cosmology, device)
        self.sigma_v = torch.as_tensor(sigma_v, dtype=torch.float32, device=device)
        self.thx0 = torch.as_tensor(thx0, dtype=torch.float32, device=device)
        self.thy0 = torch.as_tensor(thy0, dtype=torch.float32, device=device)

    def th_ein(self, z_l, z_s):
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s)
        return 4 * pi * (self.sigma_v / c_km_s) ** 2 * d_ls / d_s * rad_to_arcsec

    def Sigma(self, thx, thy, d_l):
        thx = thx - self.thx0
        thx = thx - self.thy0
        xi = d_l * (thx**2 + thy**2).sqrt() * arcsec_to_rad
        return self.sigma_v**2 / (2 * G * xi)

    def Psi_hat(self, thx, thy, d_l, d_s, d_ls):
        thx = thx - self.thx0
        thy = thy - self.thy0
        th = (thx**2 + thy**2).sqrt()
        return 4 * pi * (self.sigma_v / c_km_s) ** 2 * d_ls / d_s * th * arcsec_to_rad

    def alpha_hat(self, thx, thy, z_l):
        thx = thx - self.thx0
        thy = thy - self.thy0
        th = (thx**2 + thy**2).sqrt()
        alpha = 4 * pi * (self.sigma_v / c_km_s) ** 2 * rad_to_arcsec
        return alpha * thx / th, alpha * thy / th
