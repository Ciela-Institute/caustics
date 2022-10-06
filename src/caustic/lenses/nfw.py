from functools import cached_property
from math import pi

import torch

from ..constants import G_over_c2, arcsec_to_rad, rad_to_arcsec
from .base import AbstractLens


class NFW(AbstractLens):
    def __init__(self, thx0, thy0, m, c, delta=200.0, cosmology=None, device=None):
        """
        Args:
            thx0: [arcsec]
            thy0: [arcsec]
            m: [solMass]
            c: [1]
            delta: [1]
        """
        super().__init__(cosmology, device)
        self.thx0 = torch.as_tensor(thx0, dtype=torch.float32, device=device)
        self.thy0 = torch.as_tensor(thy0, dtype=torch.float32, device=device)
        self.m = torch.as_tensor(m, dtype=torch.float32, device=device)
        self.c = torch.as_tensor(c, dtype=torch.float32, device=device)
        self.delta = torch.as_tensor(delta, dtype=torch.float32, device=device)

    def r_s(self, z):
        """
        [Mpc]
        """
        r_delta = (3 * self.m / (4 * pi * self.delta * self.cosmology.rho_cr(z))) ** (
            1 / 3
        )
        return 1 / self.c * r_delta

    def rho_s(self, z):
        """
        [solMass / Mpc^3]
        """
        return (
            self.delta
            / 3
            * self.cosmology.rho_cr(z)
            * self.c**3
            / ((1 + self.c).log() - self.c / (1 + self.c))
        )

    def kappa_s(self, z_l, z_s):
        """
        [1]
        """
        Sigma_cr = self.Sigma_cr(z_l, z_s)
        return self.rho_s(z_l) * self.r_s(z_l) / Sigma_cr

    @classmethod
    def _f(cls, x):
        # TODO: generalize beyond torch, or patch Tensor
        return torch.where(
            x > 1,
            1 - 2 / (x**2 - 1).sqrt() * ((x - 1) / (x + 1)).sqrt().arctan(),
            torch.where(
                x < 1,
                1 - 2 / (1 - x**2).sqrt() * ((1 - x) / (1 + x)).sqrt().arctanh(),
                0.0,
            ),
        )

    @classmethod
    def _g(cls, x):
        # TODO: generalize beyond torch, or patch Tensor
        term_1 = 1 / 2 * (x / 2).log() ** 2
        term_2 = torch.where(
            x > 1,
            2 * ((x - 1) / (x + 1)).sqrt().arctan() ** 2,
            torch.where(x < 1, -2 * ((1 - x) / (1 + x)).sqrt().arctanh() ** 2, 0.0),
        )
        return term_1 + term_2

    @classmethod
    def _h(cls, x):
        term_1 = (x / 2).log()
        term_2 = torch.where(
            x > 1,
            2 / (x**2 - 1).sqrt() * ((x - 1) / (x + 1)).sqrt().arctan(),
            torch.where(
                x < 1,
                2 / (1 - x**2).sqrt() * ((1 - x) / (1 + x)).sqrt().arctanh(),
                1.0,
            ),
        )
        return term_1 + term_2

    def Sigma(self, thx, thy, z_l):
        """
        [solMass / Mpc^2]
        """
        thx = thx - self.thx0
        thy = thy - self.thy0
        d_l = self.cosmology.angular_diameter_dist(z_l)
        x = (thx**2 + thy**2).sqrt() * d_l / self.r_s(z_l) * arcsec_to_rad  # [rad]
        return 2 * self.rho_s(z_l) * self.r_s(z_l) / (x**2 - 1) * self._f(x)

    def alpha_hat(self, thx, thy, z_l):
        """
        [arcsec]
        """
        thx = thx - self.thx0
        thy = thy - self.thy0
        th = (thx**2 + thy**2).sqrt()
        d_l = self.cosmology.angular_diameter_dist(z_l)
        xi = d_l * th * arcsec_to_rad
        x = xi / self.r_s(z_l)
        alpha = (
            16
            * pi
            * G_over_c2
            * self.rho_s(z_l)
            * self.r_s(z_l) ** 3
            * self._h(x)
            * rad_to_arcsec
            / xi
        )
        return alpha * thx / th, alpha * thy / th

    def Psi_hat(self, thx, thy, z_l, z_s):
        ...  # TODO: implement!
