import torch

from ..utils import to_elliptical, transform_scalar_fn
from .base import AbstractSource


class Sersic(AbstractSource):
    def __init__(self, device=torch.device("cpu"), use_lenstronomy_k=False):
        """
        Args:
            lenstronomy_k_mode: set to `True` to calculate k in the Sersic exponential
                using the same formula as lenstronomy. Intended primarily for testing.
        """
        super().__init__(device)
        self.lenstronomy_k_mode = use_lenstronomy_k

    def brightness(self, thx, thy, thx0, thy0, q, phi, index, th_e, I_e):
        @transform_scalar_fn(thx0, thy0, phi)
        def helper(thx, thy):
            ex, ey = to_elliptical(thx, thy, q)
            e = (ex**2 + ey**2).sqrt()

            if self.lenstronomy_k_mode:
                k = 1.9992 * index - 0.3271
            else:
                k = 2 * index - 1 / 3 + 4 / 405 / index + 46 / 25515 / index**2

            exponent = -k * ((e / th_e) ** (1 / index) - 1)
            return I_e * exponent.exp()

        return helper(thx, thy)
