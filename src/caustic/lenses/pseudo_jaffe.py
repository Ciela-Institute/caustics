from math import pi

import torch

from ..utils import translate_rotate
from .base import AbstractLens


class PseudoJaffe(AbstractLens):
    """
    Notes:
        Based on `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_ and
        the `lenstronomy` source code.
    """

    def __init__(self, s=0.0001, device=torch.device("cpu")):
        super().__init__(device)
        self.s = torch.as_tensor(s, device=device)

    def mass_enclosed_2d(self, th, z_l, z_s, cosmology, kappa_0, r_core, r_s):
        Sigma_0 = kappa_0 * cosmology.Sigma_cr(z_l, z_s)
        return (
            2
            * pi
            * Sigma_0
            * r_core
            * r_s
            / (r_s - r_core)
            * (
                (r_core**2 + th**2).sqrt()
                - r_core
                - (r_s**2 + th**2).sqrt()
                + r_s
            )
        )

    def kappa_0(self, z_l, z_s, cosmology, rho_0, r_core, r_s):
        return pi * rho_0 * r_core * r_s / (r_core + r_s) / cosmology.Sigma_cr(z_l, z_s)

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, kappa_0, r_core, r_s):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        # th = torch.maximum((thx**2 + thy**2).sqrt(), self.s)
        th = (thx**2 + thy**2).sqrt()
        f = th / r_core / (1 + (1 + (th / r_core) ** 2).sqrt()) - th / r_s / (
            1 + (1 + (th / r_s) ** 2).sqrt()
        )
        alpha = 2 * kappa_0 * r_core * r_s / (r_s - r_core) * f
        ax = alpha * thx / th
        ay = alpha * thy / th
        return ax, ay

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, kappa_0, r_core, r_s):
        """
        Lensing potential (eq. A18).
        """
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        th = torch.maximum((thx**2 + thy**2).sqrt(), self.s)
        coeff = -2 * kappa_0 * r_core * r_s / (r_s - r_core)
        return coeff * (
            (r_s**2 + th**2).sqrt()
            - (r_core**2 + th**2).sqrt()
            + r_core * (r_core + (r_core**2 + th**2).sqrt()).log()
            - r_s * (r_s + (r_s**2 + th**2).sqrt()).log()
        )

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, kappa_0, r_core, r_s):
        """
        Projected mass density (eq. A6).
        """
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        th = torch.maximum((thx**2 + thy**2).sqrt(), self.s)
        coeff = kappa_0 * r_core * r_s / (r_s - r_core)
        return coeff * (
            1 / (r_core**2 + th**2).sqrt() - 1 / (r_s**2 + th**2).sqrt()
        )
