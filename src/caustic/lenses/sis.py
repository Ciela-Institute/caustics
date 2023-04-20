from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

__all__ = ("SIS",)


class SIS(ThinLens):
    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        th_ein: Optional[Tensor] = None,
        s: float = 0.0,
    ):
        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("th_ein", th_ein)
        self.s = s

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        z_l, thx0, thy0, th_ein = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        ax = th_ein * thx / th
        ay = th_ein * thy / th
        return ax, ay

    def Psi(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        z_l, thx0, thy0, th_ein = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        return th_ein * th

    def kappa(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        z_l, thx0, thy0, th_ein = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        return 0.5 * th_ein / th
