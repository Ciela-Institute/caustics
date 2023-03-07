from typing import Optional

import torch
from torch import Tensor

from ..utils import to_elliptical, translate_rotate
from .base import Source

__all__ = ("Sersic",)


class Sersic(Source):
    def __init__(
        self,
        name: str,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        index: Optional[Tensor] = None,
        th_e: Optional[Tensor] = None,
        I_e: Optional[Tensor] = None,
        s: Optional[Tensor] = torch.tensor(0.0),
        use_lenstronomy_k=False,
    ):
        super().__init__(name)
        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("index", index)
        self.add_param("th_e", th_e)
        self.add_param("I_e", I_e)
        self.add_param("s", s)

        self.lenstronomy_k_mode = use_lenstronomy_k

    def brightness(self, thx, thy, x):
        thx0, thy0, q, phi, index, th_e, I_e, s = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        ex, ey = to_elliptical(thx, thy, q)
        e = (ex**2 + ey**2).sqrt() + s

        if self.lenstronomy_k_mode:
            k = 1.9992 * index - 0.3271
        else:
            k = 2 * index - 1 / 3 + 4 / 405 / index + 46 / 25515 / index**2

        exponent = -k * ((e / th_e) ** (1 / index) - 1)
        return I_e * exponent.exp()
