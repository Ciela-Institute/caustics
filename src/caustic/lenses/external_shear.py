import torch

from ..utils import translate_rotate
from .base import AbstractThinLens


class ExternalShear(AbstractThinLens):
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__(device)

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, gamma_1, gamma_2):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        # Meneghetti eq 3.83
        a1 = thx * gamma_1 + thy * gamma_2
        a2 = thx * gamma_2 - thy * gamma_1
        return a1, a2  #I'm not sure but I think no derotation necessary

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, gamma_1, gamma_2):
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        ax, ay = self.alpha(thx, thy, z_l, z_s, cosmology, 0.0, 0.0, gamma_1, gamma_2)
        return thx * ax + thy * ay

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, gamma_1, gamma_2):
        raise ValueError("convergence undefined for external shear")
        
    def external_shear_potential(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, gamma_1, gamma_2):
        #not sure if we need this fct - I added it
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        return 0.5 * (gamma_1 * thx**2. + 2 * gamma_2 * thx * thy - gamma_1 * thy**2.)
