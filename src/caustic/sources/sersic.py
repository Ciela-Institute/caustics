import torch

from ..utils import to_elliptical, translate_rotate
from .base import Source


class Sersic(Source):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        use_lenstronomy_k=False,
    ):
        """
        Args:
            lenstronomy_k_mode: set to `True` to calculate k in the Sersic exponential
                using the same formula as lenstronomy. Intended primarily for testing.
        """
        super().__init__(device, dtype)
        self.lenstronomy_k_mode = use_lenstronomy_k

    def brightness(self, thx, thy, thx0, thy0, q, phi, index, th_e, I_e, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        ex, ey = to_elliptical(thx, thy, q)
        e = (ex**2 + ey**2).sqrt() + s

        if self.lenstronomy_k_mode:
            k = 1.9992 * index - 0.3271
        else:
            k = 2 * index - 1 / 3 + 4 / 405 / index + 46 / 25515 / index**2

        exponent = -k * ((e / th_e) ** (1 / index) - 1)
        return I_e * exponent.exp()
