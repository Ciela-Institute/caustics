from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from .base import ThinLens

__all__ = ("SinglePlane",)


class SinglePlane(ThinLens):
    """
    Single lens plane containing multiple thin lenses.
    """

    def __init__(self, name: str, cosmology: Cosmology, lenses: list[ThinLens]):
        super().__init__(name, cosmology)
        self.lenses = lenses
        for lens in lenses:
            self.add_parametrized(lens)

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        ax = torch.zeros_like(thx)
        ay = torch.zeros_like(thx)
        for lens in self.lenses:
            ax_cur, ay_cur = lens.alpha(thx, thy, z_s, x)
            ax = ax + ax_cur
            ay = ay + ay_cur
        return ax, ay

    def kappa(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        kappa = torch.zeros_like(thx)
        for lens in self.lenses:
            kappa_cur = lens.kappa(thx, thy, z_s, x)
            kappa = kappa + kappa_cur
        return kappa

    def Psi(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        Psi = torch.zeros_like(thx)
        for lens in self.lenses:
            Psi_cur = lens.Psi(thx, thy, z_s, x)
            Psi = Psi + Psi_cur
        return Psi
