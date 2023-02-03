import torch

from ..utils import derotate, translate_rotate
from .base import ThinLens


class SIE(ThinLens):
    """
    References:
        Keeton 2001, https://arxiv.org/abs/astro-ph/0102341
    """

    def __init__(self, device=torch.device("cpu")):
        super().__init__(device)

    def _get_psi(self, x, y, q, s):
        return (q**2 * (x**2 + s**2) + y**2).sqrt()

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = self._get_psi(thx, thy, q, s)
        f = (1 - q**2).sqrt()
        ax = b * q.sqrt() / f * (f * thx / (psi + s)).atan()
        ay = b * q.sqrt() / f * (f * thy / (psi + q**2 * s)).atanh()

        return derotate(ax, ay, phi)

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, s=None):
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        # Only transform coordinates once: pass thx0=0, thy=0, phi=None to alpha
        ax, ay = self.alpha(thx, thy, z_l, z_s, cosmology, 0.0, 0.0, q, None, b, s)
        return thx * ax + thy * ay

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = self._get_psi(thx, thy, q, s)
        return 0.5 * q.sqrt() * b / psi
