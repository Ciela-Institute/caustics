import torch

from ..constants import G_over_c2, arcsec_to_rad
from .base import AbstractLens


class Point(AbstractLens):
    def __init__(self, thx0, thy0, m, cosmology=None, device=None):
        super().__init__(cosmology, device)
        self.thx0 = thx0
        self.thy0 = thy0
        self.m = m

    def Sigma(self, thx, thy, d_l):
        return torch.where((thx == self.thx0) & (thy == self.thy0), torch.inf, 0.0)

    def set_th_ein(self, th_ein, d_l, d_s, d_ls):
        """
        Args:
            th_ein: [arcsec]
        """
        self.m = (th_ein * arcsec_to_rad) ** 2 / (4 * G_over_c2) / (d_ls / d_l / d_s)

    def alpha_hat(self, thx, thy, d_l):
        thx = thx - self.thx0
        thy = thy - self.thy0
        th = (thx**2 + thy**2).sqrt()
        xi = d_l * th * arcsec_to_rad
        alpha = 4 * G_over_c2 * self.m / xi
        return alpha * thx / th, alpha * thy / th

    def Psi_hat(self, thx, thy, d_l, d_s, d_ls):
        thx = thx - self.thx0
        thy = thy - self.thy0
        th_rad = (thx**2 + thy**2).sqrt() * arcsec_to_rad
        factor = (4 * G_over_c2 * self.m) * (d_ls / d_l / d_s)
        return factor * th_rad.log()
