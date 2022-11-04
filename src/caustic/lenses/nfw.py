from functools import cached_property
from math import pi

import torch

from ..constants import G_over_c2, arcsec_to_rad, rad_to_arcsec
from ..utils import transform_scalar_fn, transform_vector_fn
from .base import AbstractLens

DELTA = 200.0


class NFW(AbstractLens):
    def __init__(self, thx0, thy0, m, c, z_l, cosmology=None, device=None):
        """
        Args:
            thx0: [arcsec]
            thy0: [arcsec]
            m: [solMass]
            c: [1]
        """
        super().__init__(z_l, cosmology, device)
        self.thx0 = thx0
        self.thy0 = thy0
        self.m = m
        self.c = c

    @cached_property
    def r_s(self):
        """
        [Mpc]
        """
        rho_cr = self.cosmology.rho_cr(self.z_l)
        r_delta = (3 * self.m / (4 * pi * DELTA * rho_cr)) ** (1 / 3)
        return 1 / self.c * r_delta

    @cached_property
    def rho_s(self):
        """
        [solMass / Mpc^3]
        """
        return (
            DELTA
            / 3
            * self.cosmology.rho_cr(self.z_l)
            * self.c**3
            / ((1 + self.c).log() - self.c / (1 + self.c))
        )

    def kappa_s(self, z_s):
        """
        [1]
        """
        Sigma_cr = self.Sigma_cr(z_s)
        return self.rho_s * self.r_s / Sigma_cr

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

    @transform_vector_fn
    def alpha_hat(self, thx, thy, z_s):
        """
        [arcsec]
        """
        th = (thx**2 + thy**2).sqrt()
        xi = self.d_l * th * arcsec_to_rad
        x = xi / self.r_s
        alpha = (
            16
            * pi
            * G_over_c2
            * self.rho_s
            * self.r_s**3
            * self._h(x)
            * rad_to_arcsec
            / xi
        )
        return alpha * thx / th, alpha * thy / th

    def alpha(self, thx, thy, z_s):
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(self.z_l, z_s)
        ahx, ahy = self.alpha_hat(thx, thy, z_s)
        return d_ls / d_s * ahx, d_ls / d_s * ahy

    @transform_scalar_fn
    def kappa(self, thx, thy, z_s):
        th = (thx**2 + thy**2).sqrt()
        xi = self.d_l * th * arcsec_to_rad
        x = xi / self.r_s  # xi / xi_0
        return 2 * self.kappa_s(z_s) * self._f(x) / (x**2 - 1)

    @transform_scalar_fn
    def Psi(self, thx, thy, z_s):
        th = (thx**2 + thy**2).sqrt()
        xi = self.d_l * th * arcsec_to_rad
        x = xi / self.r_s  # xi / xi_0
        return 4 * self.kappa_s(z_s) * self._g(x)
