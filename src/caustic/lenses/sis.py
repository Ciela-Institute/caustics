import torch

from ..utils import translate_rotate
from .base import ThinLens

__all__ = ("SIS",)

class SIS(ThinLens):
    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + s
        ax = th_ein * thx / th
        ay = th_ein * thy / th
        return ax, ay

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + s
        return th_ein * th

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + s
        return th_ein / th
