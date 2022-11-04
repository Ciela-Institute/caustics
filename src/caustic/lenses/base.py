from abc import abstractmethod
from functools import cached_property
from math import pi
from typing import Tuple

from torch import Tensor

from ..base import Base
from ..constants import G_over_c2, arcsec_to_rad, c_Mpc_s


class AbstractLens(Base):
    def __init__(self, z_l, cosmology, device):
        super().__init__(cosmology, device)
        self.z_l = z_l

    @cached_property
    def d_l(self):
        return self.cosmology.angular_diameter_dist(self.z_l)

    @abstractmethod
    def alpha(self, thx, thy, z_s) -> Tuple[Tensor, Tensor]:
        """
        Reduced deflection angle [arcsec]
        """
        ...

    def alpha_hat(self, thx, thy, z_s) -> Tuple[Tensor, Tensor]:
        """
        Physical deflection angle [arcsec]
        """
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(self.z_l, z_s)
        return d_s / d_ls * self.alpha(thx, thy, z_s)

    @abstractmethod
    def kappa(self, thx, thy, z_s) -> Tensor:
        """
        Convergence [1]
        """
        ...

    @abstractmethod
    def Psi(self, thx, thy, z_s) -> Tensor:
        """
        Potential [arcsec^2]
        """
        ...

    def Sigma_cr(self, z_s) -> Tensor:
        """
        Critical lensing density [solMass / Mpc^2]
        """
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(self.z_l, z_s)
        return d_s / self.d_l / d_ls / (4 * pi * G_over_c2)

    def Sigma(self, thx, thy, z_s) -> Tensor:
        """
        Surface mass density.

        Returns:
            [solMass / Mpc^2]
        """
        Sigma_cr = self.Sigma_cr(z_s)
        return self.kappa(thx, thy, z_s) * Sigma_cr

    def raytrace(self, thx, thy, z_s) -> Tuple[Tensor, Tensor]:
        ax, ay = self.alpha(thx, thy, z_s)
        return thx - ax, thy - ay

    def time_delay(self, thx, thy, z_s):
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(self.z_l, z_s)
        ax, ay = self.alpha(thx, thy, z_s)
        Psi = self.Psi(thx, thy, z_s)
        factor = (1 + self.z_l) / c_Mpc_s * d_s * self.d_l / d_ls
        fp = 0.5 * d_ls**2 / d_s**2 * (ax**2 + ay**2) - Psi
        return factor * fp * arcsec_to_rad**2

    # @abstractmethod
    # def magnification(self, x: Tensor, y: Tensor, z=None, w=None, t=None) -> Tensor:
    #     ...
