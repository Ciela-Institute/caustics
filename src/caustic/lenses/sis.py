import torch

from ..utils import translate_rotate
from .base import AbstractLens


class SIS(AbstractLens):
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__(device)

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        th = (thx**2 + thy**2).sqrt()
        ax = th_ein * thx / th
        ay = th_ein * thy / th
        return ax, ay

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        th = (thx**2 + thy**2).sqrt()
        return th_ein * th

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)

        th = (thx**2 + thy**2).sqrt()
        return th_ein / th
