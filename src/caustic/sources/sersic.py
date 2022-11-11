import torch

from ..cosmology import AbstractCosmology
from ..utils import (
    flip_axis_ratio,
    to_elliptical,
    transform_scalar_fn,
)
from .base import AbstractSource


class Sersic(AbstractSource):
    def __init__(
        self,
        thx0,
        thy0,
        phi,
        q,
        index,
        th_e,
        I_e,
        cosmology: AbstractCosmology,
        device: torch.device,
    ):
        super().__init__(cosmology, device)
        self.thx0 = thx0
        self.thy0 = thy0
        self.q, self.phi = flip_axis_ratio(q, phi)
        self.index = index
        self.th_e = th_e
        self.I_e = I_e

    @transform_scalar_fn
    def brightness(self, thx, thy):
        ex, ey = to_elliptical(thx, thy, self.q)
        e = (ex**2 + ey**2).sqrt()
        k = 2 * self.index - 1 / 3 + 4 / 405 / self.index + 46 / 25515 / self.index**2
        # k = 1.9992 * self.index - 0.3271  # lenstronomy
        exponent = -k * ((e / self.th_e) ** (1 / self.index) - 1)
        return self.I_e * exponent.exp()
