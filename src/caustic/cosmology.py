from abc import abstractmethod
from math import pi
from typing import Optional

import torch
from astropy.cosmology import default_cosmology
from scipy.special import hyp2f1
from torchinterp1d import interp1d

from .base import Base
from .constants import G_over_c2, c_Mpc_s, km_to_mpc

h0_default = float(default_cosmology.get().h)
rho_cr_0_default = float(
    default_cosmology.get().critical_density(0).to("solMass/Mpc^3").value
)
Om0_default = float(default_cosmology.get().Om0)

# Set up interpolator to speed up comoving distance calculations in Lambda-CDM
# cosmologies. Construct with float64 precision.
_comoving_dist_helper_x_grid = 10 ** torch.linspace(-3, 1, 500, dtype=torch.float32)
_comoving_dist_helper_y_grid = torch.as_tensor(
    _comoving_dist_helper_x_grid
    * hyp2f1(1 / 3, 1 / 2, 4 / 3, -(_comoving_dist_helper_x_grid**3)),
    dtype=torch.float64,
)


class AbstractCosmology(Base):
    def __init__(
        self,
        h0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device, dtype)
        self.h0 = h0

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
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(h0, device, dtype)
        self.rho_cr_0 = rho_cr_0
        self.Om0 = Om0
        self.Ode0 = 1 - Om0

        self._comoving_dist_helper_x_grid = _comoving_dist_helper_x_grid.to(
            self.device, dtype
        )
        self._comoving_dist_helper_y_grid = _comoving_dist_helper_y_grid.to(
            self.device, dtype
        )

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        # NOTE: initializing with float32 and calling .to(float64) will give different
        # results than initializing with float64.
        self._comoving_dist_helper_x_grid = self._comoving_dist_helper_x_grid.to(
            device=device, dtype=dtype
        )
        self._comoving_dist_helper_y_grid = self._comoving_dist_helper_y_grid.to(
            device=device, dtype=dtype
        )

    def rho_cr(self, z) -> torch.Tensor:
        return self.rho_cr_0 * (self.Om0 * (1 + z) ** 3 + self.Ode0)

    def _comoving_dist_helper(self, x):
        return interp1d(
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
