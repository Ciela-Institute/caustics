import torch

from ..utils import transform_scalar_fn, transform_vector_fn
from .base import AbstractLens


class SIE(AbstractLens):
    """
    References:
        Keeton 2001, https://arxiv.org/abs/astro-ph/0102341
    """

    def __init__(self, device=torch.device("cpu")):
        super().__init__(device)

    def _get_psi(self, x, y, q, s):
        return (q**2 * (x**2 + s**2) + y**2).sqrt()

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, s=None):
        if s is None:
            s = 0.0

        @transform_vector_fn(thx0, thy0, phi)
        def helper(thx, thy):
            psi = self._get_psi(thx, thy, q, s)
            f = (1 - q**2).sqrt()
            ax = b * q.sqrt() / f * (f * thx / (psi + s)).atan()
            ay = b * q.sqrt() / f * (f * thy / (psi + q**2 * s)).atanh()
            return ax, ay

        return helper(thx, thy)

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, s=None):
        @transform_scalar_fn(thx0, thy0, phi)
        def helper(thx, thy):
            # Only transform coordinates once!
            ax, ay = self.alpha(thx, thy, z_l, z_s, cosmology, 0.0, 0.0, q, None, b, s)
            return thx * ax + thy * ay

        return helper(thx, thy)

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, s=None):
        if s is None:
            s = torch.tensor(0.0, device=self.device)

        @transform_scalar_fn(thx0, thy0, phi)
        def helper(thx, thy):
            psi = self._get_psi(thx, thy, q, s)
            return 0.5 * q.sqrt() * b / psi

        return helper(thx, thy)
