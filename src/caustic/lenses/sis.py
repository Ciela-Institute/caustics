from math import pi

import torch

from ..constants import G, arcsec_to_rad, c_km_s, c_Mpc_s, rad_to_arcsec
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

    def xi_0(self, z_l, z_s):
        d_l = self.cosmology.angular_diameter_dist(z_l)
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s)
        return 4 * pi * (self.sigma_v / c_km_s)**2 * d_l * d_ls / d_s  # Mpc

    def th_ein(self, z_l, z_s):
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s)
        return 4 * pi * (self.sigma_v / c_km_s) ** 2 * d_ls / d_s * rad_to_arcsec

    def Sigma(self, thx, thy, d_l):
        thx = thx - self.thx0
        thx = thx - self.thy0
        xi = d_l * (thx**2 + thy**2).sqrt() * arcsec_to_rad
        return self.sigma_v**2 / (2 * G * xi)

    def Psi(self, thx, thy, z_l, z_s):
        thx = thx - self.thx0
        thy = thy - self.thy0
        return (thx**2 + thy**2).sqrt()  # arcsec

    def Psi_hat(self, thx, thy, z_l, z_s):
        d_l = self.cosmology.angular_diameter_dist(z_l)
        return self.Psi(thx, thy, z_l, z_s) * (self.xi_0(z_l, z_s) / d_l)**2  # arcsec

    def alpha(self, thx, thy, z_l, z_s):
        thx = thx - self.thx0
        thy = thy - self.thy0
        th = (thx**2 + thy**2).sqrt()
        return thx / th, thy / th  # arcsec

    def alpha_hat(self, thx, thy, z_l, z_s):
        d_l = self.cosmology.angular_diameter_dist(z_l)
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s)
        factor = self.xi_0(z_l, z_s) * d_s / d_l / d_ls
        ax, ay = self.alpha(thx, thy, z_l, z_s)
        return factor * ax, factor * ay  # arcsec

    # def time_delay(self, thx, thy, z_l, z_s):
    #     d_l = self.cosmology.angular_diameter_dist(z_l)
    #     d_s = self.cosmology.angular_diameter_dist(z_s)
    #     d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s)
    #     factor = (1 + z_l) / c_Mpc_s * d_s * self.xi_0(z_l, z_s)**2 / d_l / d_ls
    #     ax, ay = self.alpha(thx, thy, z_l, z_s)
    #     fp_hat = 0.5 * (ax**2 + ay**2) - self.Psi(thx, thy, z_l, z_s)
    #     return factor * fp_hat * arcsec_to_rad**2
