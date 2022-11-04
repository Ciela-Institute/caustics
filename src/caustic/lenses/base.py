from abc import abstractmethod
from math import pi
from typing import Tuple

from torch import Tensor

from ..base import Base
from ..constants import G_over_c2, c_Mpc_s, arcsec_to_rad


class AbstractLens(Base):
    def __init__(self, cosmology, device):
        super().__init__(cosmology, device)

    def xi_0(self, z_l, z_s) -> Tensor:
        ...

    def Sigma_cr(self, z_l, z_s) -> Tensor:
        """
        Args:
            d_l: [Mpc]

        Returns:
            [solMass / Mpc^2]
        """
        d_l = self.cosmology.angular_diameter_dist(z_l)
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s)
        return d_s / d_l / d_ls / (4 * pi * G_over_c2)

    @abstractmethod
    def Sigma(self, thx, thy, z_l, z_s) -> Tensor:
        """
        Computes the surface mass density.

        Returns:
            [solMass / Mpc^2]
        """
        ...

    def kappa(self, thx, thy, z_l, z_s) -> Tensor:
        """
        Computes the convergence.

        Returns:
            [1]
        """
        Sigma_cr = self.Sigma_cr(z_l, z_s)
        return self.Sigma(thx, thy, z_l, z_s) / Sigma_cr

    @abstractmethod
    def alpha_hat(self, thx, thy, z_l, z_s) -> Tuple[Tensor, Tensor]:
        """
        Computes the deflection angle.

        Returns:
            [arcsec]
        """
        # TODO: grad(Psi) or convolution of kappa
        ...

    def alpha(self, thx, thy, z_l, z_s) -> Tuple[Tensor, Tensor]:
        """
        Computes the reduced deflection angle.

        Returns:
            [arcsec]
        """
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s)
        ahx, ahy = self.alpha_hat(thx, thy, z_l, z_s)
        return d_ls / d_s * ahx, d_ls / d_s * ahy

    def raytrace(self, thx, thy, z_l, z_s) -> Tuple[Tensor, Tensor]:
        ax, ay = self.alpha(thx, thy, z_l, z_s)
        return thx - ax, thy - ay

    def Psi_hat(self, thx, thy, z_l, z_s) -> Tensor:
        """
        Computes the effective lensing potential.
        """
        # TODO: compute from kappa by default?
        ...

    def Psi(self, thx, thy, z_l, z_s) -> Tensor:
        """
        Computes the dimensionless lensing potential.
        """
        ...

    def fermat_potential(self, thx, thy, z_l, z_s) -> Tensor:
        ax, ay = self.alpha(thx, thy, z_l, z_s)
        return (ax**2 + ay**2) / 2 - self.Psi(thx, thy, z_l, z_s)

    def time_delay(self, thx, thy, z_l, z_s):
        d_l = self.cosmology.angular_diameter_dist(z_l)
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s)
        factor = (1 + z_l) / c_Mpc_s * d_s * self.xi_0(z_l, z_s)**2 / d_l / d_ls
        ax, ay = self.alpha(thx, thy, z_l, z_s)
        fp_hat = 0.5 * (ax**2 + ay**2) - self.Psi(thx, thy, z_l, z_s)
        return factor * fp_hat * arcsec_to_rad**2



    # @abstractmethod
    # def magnification(self, x: Tensor, y: Tensor, z=None, w=None, t=None) -> Tensor:
    #     ...
