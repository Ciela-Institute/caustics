from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from ..base import Base
from ..constants import arcsec_to_rad, c_Mpc_s


class AbstractLens(Base):
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__(device)

    # TODO: implement .to(device)

    @abstractmethod
    def alpha(
        self, thx, thy, z_l, z_s, cosmology, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Reduced deflection angle [arcsec]
        """
        ...

    def alpha_hat(
        self, thx, thy, z_l, z_s, cosmology, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Physical deflection angle [arcsec]
        """
        d_s = cosmology.angular_diameter_dist(z_s)
        d_ls = cosmology.angular_diameter_dist_z1z2(z_l, z_s)
        return d_s / d_ls * self.alpha(thx, thy, z_l, z_s, cosmology, *args, **kwargs)

    @abstractmethod
    def kappa(self, thx, thy, z_l, z_s, cosmology, *args, **kwargs) -> Tensor:
        """
        Convergence [1]
        """
        ...

    @abstractmethod
    def Psi(self, thx, thy, z_l, z_s, cosmology, *args, **kwargs) -> Tensor:
        """
        Potential [arcsec^2]
        """
        ...

    def Sigma(self, thx, thy, z_l, z_s, cosmology, *args, **kwargs) -> Tensor:
        """
        Surface mass density.

        Returns:
            [solMass / Mpc^2]
        """
        Sigma_cr = cosmology.Sigma_cr(z_l, z_s)
        return self.kappa(thx, thy, z_l, z_s, cosmology, *args, **kwargs) * Sigma_cr

    def raytrace(
        self, thx, thy, z_l, z_s, cosmology, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        ax, ay = self.alpha(thx, thy, z_l, z_s, cosmology, *args, **kwargs)
        return thx - ax, thy - ay

    def time_delay(self, thx, thy, z_l, z_s, cosmology, *args, **kwargs):
        d_l = cosmology.angular_diameter_dist(z_l)
        d_s = cosmology.angular_diameter_dist(z_s)
        d_ls = cosmology.angular_diameter_dist_z1z2(z_l, z_s)
        ax, ay = self.alpha(thx, thy, z_l, z_s, cosmology, *args, **kwargs)
        Psi = self.Psi(thx, thy, z_l, z_s, cosmology, *args, **kwargs)
        factor = (1 + z_l) / c_Mpc_s * d_s * d_l / d_ls
        fp = 0.5 * d_ls**2 / d_s**2 * (ax**2 + ay**2) - Psi
        return factor * fp * arcsec_to_rad**2

    # @abstractmethod
    # def magnification(self, x: Tensor, y: Tensor, z=None, w=None, t=None) -> Tensor:
    #     ...
