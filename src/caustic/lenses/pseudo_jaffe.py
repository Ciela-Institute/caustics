from math import pi
from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

__all__ = ("PseudoJaffe",)


class PseudoJaffe(ThinLens):
    """
    Notes:
        Based on `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_ and
        the `lenstronomy` source code.
    """

    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        kappa_0: Optional[Tensor] = None,
        th_core: Optional[Tensor] = None,
        th_s: Optional[Tensor] = None,
        s: Optional[Tensor] = torch.tensor(0.0),
    ):
        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("kappa_0", kappa_0)
        self.add_param("th_core", th_core)
        self.add_param("th_s", th_s)
        self.add_param("s", s)

    def mass_enclosed_2d(self, th, z_s, x: Optional[dict[str, Any]] = None):
        z_l, thx0, thy0, kappa_0, th_core, th_s, s = self.unpack(x)

        th += s
        Sigma_0 = kappa_0 * self.cosmology.Sigma_cr(z_l, z_s, x)
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
    def kappa_0(
        z_l,
        z_s,
        rho_0,
        th_core,
        th_s,
        cosmology: Cosmology,
        x: Optional[dict[str, Any]] = None,
    ):
        return (
            pi
            * rho_0
            * th_core
            * th_s
            / (th_core + th_s)
            / cosmology.Sigma_cr(z_l, z_s, x)
        )

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        z_l, thx0, thy0, kappa_0, th_core, th_s, s = self.unpack(x)

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
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Lensing potential (eq. A18).
        """
        z_l, thx0, thy0, kappa_0, th_core, th_s, s = self.unpack(x)

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
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Projected mass density (eq. A6).
        """
        z_l, thx0, thy0, kappa_0, th_core, th_s, s = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + s
        coeff = kappa_0 * th_core * th_s / (th_s - th_core)
        return coeff * (
            1 / (th_core**2 + th**2).sqrt() - 1 / (th_s**2 + th**2).sqrt()
        )
