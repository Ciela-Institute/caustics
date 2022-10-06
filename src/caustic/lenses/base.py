from abc import abstractmethod
from math import pi

from torch import Tensor

from ..base import Base
from ..constants import G_over_c2


class AbstractLens(Base):
    def __init__(self, cosmology, device):
        super().__init__(cosmology, device)

    def Sigma_cr(self, z_l, z_s):
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
    def Sigma(self, xix, xiy, z_l):
        """
        Computes the surface mass density.

        Returns:
            [solMass / Mpc^2]
        """
        ...

    def kappa(self, thx, thy, z_l, z_s):
        """
        Computes the convergence.

        Returns:
            [1]
        """
        Sigma_cr = self.Sigma_cr(z_l, z_s)
        return self.Sigma(thx, thy, z_l) / Sigma_cr

    @abstractmethod
    def alpha_hat(self, thx, thy, z_l):
        """
        Computes the deflection angle.

        Returns:
            [arcsec]
        """
        # TODO: grad(Psi) or convolution of kappa
        ...

    def alpha(self, thx, thy, z_l, z_s):
        """
        Computes the reduced deflection angle.

        Returns:
            [arcsec]
        """
        d_s = self.cosmology.angular_diameter_dist(z_s)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s)
        ahx, ahy = self.alpha_hat(thx, thy, z_l)
        return d_ls / d_s * ahx, d_ls / d_s * ahy

    def raytrace(self, thx, thy, z_l, z_s):
        ax, ay = self.alpha(thx, thy, z_l, z_s)
        return thx - ax, thy - ay

    @abstractmethod
    def Psi_hat(self, thx, thy, z_l, z_s):
        """
        Computes the effective lensing potential.
        """
        # TODO: compute from kappa
        ...

    # @abstractmethod
    # def Psi(self, x: Tensor, y: Tensor, z=None, w=None, t=None) -> Tensor:
    #     """
    #     Computes the dimensionless lensing potential.
    #     """
    #     ...

    # def fermat_potential(self, x: Tensor, y: Tensor, z=None, w=None, t=None) -> Tensor:
    #     ax, ay = self.alpha(x, y, z, w, t)
    #     return (ax**2 + ay**2) / 2 - self.Psi(x, y, z, w, t)

    # @abstractmethod
    # def time_delay(self, x: Tensor, y: Tensor, z=None, w=None, t=None) -> Tensor:
    #     ...

    # @abstractmethod
    # def magnification(self, x: Tensor, y: Tensor, z=None, w=None, t=None) -> Tensor:
    #     ...
