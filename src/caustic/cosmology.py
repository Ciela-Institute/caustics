from abc import ABC, abstractmethod
from math import pi

import torch
from astropy.cosmology import default_cosmology
from scipy.special import hyp2f1
from torchinterp1d import Interp1d

from .constants import G_over_c2, c_Mpc_s, km_to_mpc

_interp1d = Interp1d()

h0_default = float(default_cosmology.get().h)
rho_cr_0_default = float(
    default_cosmology.get().critical_density(0).to("solMass/Mpc^3").value
)
Om0_default = float(default_cosmology.get().Om0)

# Set up interpolator to speed up comoving distance calculations in Lambda-CDM cosmologies
_comoving_dist_helper_x_grid = 10 ** torch.linspace(-3, 1, 500)
_comoving_dist_helper_y_grid = torch.as_tensor(
    _comoving_dist_helper_x_grid
    * hyp2f1(1 / 3, 1 / 2, 4 / 3, -(_comoving_dist_helper_x_grid**3)),
    dtype=torch.float32,
)


class AbstractCosmology(ABC):
    def __init__(self, h0, device):
        self.h0 = h0
        self.device = device

    @abstractmethod
    def rho_cr(self, z) -> torch.Tensor:
        ...

    # TODO: cache all these things to make multiplane lensing faster?
    @abstractmethod
    def comoving_dist(self, z) -> torch.Tensor:
        ...

    @property
    def dist_hubble(self):
        """
        [Mpc]
        """
        return c_Mpc_s / (100 * km_to_mpc) / self.h0

    # TODO: are these relations the same in every cosmology (even non-FLRW ones)?
    def comoving_dist_z1z2(self, z1, z2):
        return self.comoving_dist(z2) - self.comoving_dist(z1)

    def angular_diameter_dist(self, z):
        """
        [Mpc]
        """
        return self.comoving_dist(z) / (1 + z)

    def angular_diameter_dist_z1z2(self, z1, z2):
        """
        [Mpc]
        """
        return self.comoving_dist_z1z2(z1, z2) / (1 + z2)

    def time_delay_dist(self, z_l, z_s):
        """
        [Mpc]
        """
        d_l = self.angular_diameter_dist(z_l)
        d_s = self.angular_diameter_dist(z_s)
        d_ls = self.angular_diameter_dist_z1z2(z_l, z_s)
        return (1 + z_l) * d_l * d_s / d_ls

    def Sigma_cr(self, z_l, z_s):
        """
        Critical lensing density [solMass / Mpc^2]
        """
        d_l = self.angular_diameter_dist(z_l)
        d_s = self.angular_diameter_dist(z_s)
        d_ls = self.angular_diameter_dist_z1z2(z_l, z_s)
        return d_s / d_l / d_ls / (4 * pi * G_over_c2)


class FlatLambdaCDMCosmology(AbstractCosmology):
    """
    Flat LCDM cosmology with no radiation.

    Notes:
        Compute comoving_dist on the fly
    """

    def __init__(
        self,
        h0=h0_default,
        rho_cr_0=rho_cr_0_default,
        Om0=Om0_default,
        device=None,
    ):
        super().__init__(h0, device)
        self.rho_cr_0 = rho_cr_0
        self.Om0 = Om0
        self.Ode0 = 1 - Om0
        # TODO: can we avoid this operation? Changing device after initializing
        # will break this.
        self._comoving_dist_helper_x_grid = _comoving_dist_helper_x_grid.to(self.device)
        self._comoving_dist_helper_y_grid = _comoving_dist_helper_y_grid.to(self.device)

    def rho_cr(self, z) -> torch.Tensor:
        return self.rho_cr_0 * (self.Om0 * (1 + z) ** 3 + self.Ode0)

    def _comoving_dist_helper(self, x):
        return _interp1d(
            self._comoving_dist_helper_x_grid,
            self._comoving_dist_helper_y_grid,
            torch.atleast_1d(x),
        ).reshape(x.shape)

    def comoving_dist(self, z):
        """
        [Mpc]
        """
        ratio = torch.tensor((self.Om0 / self.Ode0) ** (1 / 3), device=self.device)
        return (
            self.dist_hubble
            * (
                self._comoving_dist_helper((1 + z) * ratio)
                - self._comoving_dist_helper(ratio)
            )
            / (self.Om0 ** (1 / 3) * self.Ode0 ** (1 / 6))
        )
