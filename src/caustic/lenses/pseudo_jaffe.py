from math import pi

import torch

from ..utils import translate_rotate
from .base import AbstractThinLens


class PseudoJaffe(AbstractThinLens):
    """
    Notes:
        Based on `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_ and
        the `lenstronomy` source code.
    """

    def mass_enclosed_2d(self, th, z_l, z_s, cosmology, kappa_0, th_core, th_s, s=None):
        s = (
            torch.tensor(0.0, device=self.device, dtype=kappa_0.dtype)
            if s is None
            else s
        )
        th += s
        Sigma_0 = kappa_0 * cosmology.Sigma_cr(z_l, z_s)
        return (
            2
            * pi
            * Sigma_0
            * th_core
            * th_s
            / (th_s - th_core)
            * (
                (th_core**2 + th**2).sqrt()
                - th_core
                - (th_s**2 + th**2).sqrt()
                + th_s
            )
        )

    @staticmethod
    def kappa_0(z_l, z_s, cosmology, rho_0, th_core, th_s):
        return (
            pi
            * rho_0
            * th_core
            * th_s
            / (th_core + th_s)
            / cosmology.Sigma_cr(z_l, z_s)
        )

    def alpha(
        self, thx, thy, z_l, z_s, cosmology, thx0, thy0, kappa_0, th_core, th_s, s=None
    ):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + s
        f = th / th_core / (1 + (1 + (th / th_core) ** 2).sqrt()) - th / th_s / (
            1 + (1 + (th / th_s) ** 2).sqrt()
        )
        alpha = 2 * kappa_0 * th_core * th_s / (th_s - th_core) * f
        ax = alpha * thx / th
        ay = alpha * thy / th
        return ax, ay

    def Psi(
        self, thx, thy, z_l, z_s, cosmology, thx0, thy0, kappa_0, th_core, th_s, s=None
    ):
        """
        Lensing potential (eq. A18).
        """
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + s
        coeff = -2 * kappa_0 * th_core * th_s / (th_s - th_core)
        return coeff * (
            (th_s**2 + th**2).sqrt()
            - (th_core**2 + th**2).sqrt()
            + th_core * (th_core + (th_core**2 + th**2).sqrt()).log()
            - th_s * (th_s + (th_s**2 + th**2).sqrt()).log()
        )

    def kappa(
        self, thx, thy, z_l, z_s, cosmology, thx0, thy0, kappa_0, th_core, th_s, s=None
    ):
        """
        Projected mass density (eq. A6).
        """
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + s
        coeff = kappa_0 * th_core * th_s / (th_s - th_core)
        return coeff * (
            1 / (th_core**2 + th**2).sqrt() - 1 / (th_s**2 + th**2).sqrt()
        )
