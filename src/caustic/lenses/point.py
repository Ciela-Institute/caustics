import torch

from ..utils import translate_rotate
from .base import AbstractLens


class Point(AbstractLens):
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__(device)

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        th = (thx**2 + thy**2).sqrt()
        ax = th_ein**2 / th
        ay = th_ein**2 / th
        return ax, ay

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        th = (thx**2 + thy**2).sqrt()
        return th_ein**2 * th.log()

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        return torch.where((thx == 0) & (thy == 0), torch.inf, 0.0)
