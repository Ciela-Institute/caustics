import torch

from ..utils import derotate, translate_rotate
from .base import AbstractThinLens


class EPL(AbstractThinLens):
    """
    Elliptical power law (aka singular power-law ellipsoid) profile.
    """

    def __init__(self, device=torch.device("cpu")):
        super().__init__(device)

    def _get_psi(self, x, y, q, s):
        return (q**2 * (x**2 + s**2) + y**2).sqrt()

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, t, s=None):
        """
        Args:
            b: scale length.
            t: power law slope.
            s: core radius.
        """
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)

        # TODO: fill in code to compute the deflection angles ax and ay
        ax = ...
        ay = ...

        return derotate(ax, ay, phi)

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, t, s=None):
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)

        # Only transform coordinates once: pass thx0=0, thy=0, phi=None to alpha
        ax, ay = self.alpha(thx, thy, z_l, z_s, cosmology, 0.0, 0.0, q, None, b, s)
        return (thx * ax + thy * ay) / (2 - t)

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, t, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = self._get_psi(thx, thy, q, s)
        return (2 - t) / 2 * (b / psi) ** t
