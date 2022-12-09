import torch

from ..utils import translate_rotate
from .base import AbstractThinLens


class ExternalShear(AbstractThinLens):
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__(device)

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, gamma_1, gamma_2):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        return ...

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, gamma_1, gamma_2):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        return ...

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, gamma_1, gamma_2):
        raise ValueError("convergence undefined for external shear")
