import torch

from ..utils import translate_rotate
from .base import ThinLens


class Point(ThinLens):
    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + s
        ax = th_ein**2 / th
        ay = th_ein**2 / th
        return ax, ay

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + s
        return th_ein**2 * th.log()

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        return torch.where((thx == 0) & (thy == 0), torch.inf, 0.0)
