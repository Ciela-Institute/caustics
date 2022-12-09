from math import pi

import torch

from ..constants import G_over_c2, arcsec_to_rad, rad_to_arcsec
from ..utils import translate_rotate
from .base import AbstractLens

DELTA = 200.0


class NFW(AbstractLens):
    def __init__(self, device: torch.device = torch.device("cpu")):
        """
        Args:
            thx0: [arcsec]
            thy0: [arcsec]
            m: [solMass]
            c: [1]
        """
        super().__init__(device)

    def get_r_s(self, z_l, cosmology, m, c):
        """
        [Mpc]
        """
        rho_cr = cosmology.rho_cr(z_l)
        r_delta = (3 * m / (4 * pi * DELTA * rho_cr)) ** (1 / 3)
        return 1 / c * r_delta

    def get_rho_s(self, z_l, cosmology, c):
        """
        [solMass / Mpc^3]
        """
        return (
            DELTA / 3 * cosmology.rho_cr(z_l) * c**3 / ((1 + c).log() - c / (1 + c))
        )

    def get_kappa_s(self, z_l, z_s, cosmology, m, c):
        """
        [1]
        """
        Sigma_cr = cosmology.Sigma_cr(z_l, z_s)
        return (
            self.get_rho_s(z_l, cosmology, c)
            * self.get_r_s(z_l, cosmology, m, c)
            / Sigma_cr
        )

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

    def alpha_hat(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, m, c):
        """
        [arcsec]
        """
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        d_l = cosmology.angular_diameter_dist(z_l)
        th = (thx**2 + thy**2).sqrt()
        r_s = self.get_r_s(z_l, cosmology, m, c)
        xi = d_l * th * arcsec_to_rad
        x = xi / r_s

        alpha = (
            16
            * pi
            * G_over_c2
            * self.get_rho_s(z_l, cosmology, c)
            * r_s**3
            * self._h(x)
            * rad_to_arcsec
            / xi
        )

        ax = alpha * thx / th
        ay = alpha * thy / th
        return ax, ay

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, m, c):
        d_s = cosmology.angular_diameter_dist(z_s)
        d_ls = cosmology.angular_diameter_dist_z1z2(z_l, z_s)
        ahx, ahy = self.alpha_hat(thx, thy, z_l, z_s, cosmology, thx0, thy0, m, c)
        return d_ls / d_s * ahx, d_ls / d_s * ahy

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, m, c):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        d_l = cosmology.angular_diameter_dist(z_l)
        th = (thx**2 + thy**2).sqrt()
        r_s = self.get_r_s(z_l, cosmology, m, c)
        xi = d_l * th * arcsec_to_rad
        x = xi / r_s  # xi / xi_0
        kappa_s = self.get_kappa_s(z_l, z_s, cosmology, m, c)
        return 2 * kappa_s * self._f(x) / (x**2 - 1)

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, m, c):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        d_l = cosmology.angular_diameter_dist(z_l)
        th = (thx**2 + thy**2).sqrt()
        r_s = self.get_r_s(z_l, cosmology, m, c)
        xi = d_l * th * arcsec_to_rad
        x = xi / r_s  # xi / xi_0
        kappa_s = self.get_kappa_s(z_l, z_s, cosmology, m, c)
        return 4 * kappa_s(z_s) * self._g(x)
