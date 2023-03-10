from abc import abstractmethod
from math import pi
from typing import Any, Optional

import torch
from astropy.cosmology import default_cosmology
from scipy.special import hyp2f1
from torch import Tensor
from torchinterp1d import interp1d

from .constants import G_over_c2, c_Mpc_s, km_to_Mpc
from .parametrized import Parametrized

__all__ = (
    "h0_default",
    "rho_cr_0_default",
    "Om0_default",
    "Cosmology",
    "FlatLambdaCDM",
)

h0_default = float(default_cosmology.get().h)
rho_cr_0_default = float(
    default_cosmology.get().critical_density(0).to("solMass/Mpc^3").value
)
Om0_default = float(default_cosmology.get().Om0)

# Set up interpolator to speed up comoving distance calculations in Lambda-CDM
# cosmologies. Construct with float64 precision.
_comoving_dist_helper_x_grid = 10 ** torch.linspace(-3, 1, 500, dtype=torch.float64)
_comoving_dist_helper_y_grid = torch.as_tensor(
    _comoving_dist_helper_x_grid
    * hyp2f1(1 / 3, 1 / 2, 4 / 3, -(_comoving_dist_helper_x_grid**3)),
    dtype=torch.float64,
)


class Cosmology(Parametrized):
    """
    Units:
        - Distance: Mpc
        - Mass: solMass
    """

    def __init__(self, name: str):
        super().__init__(name)

    @abstractmethod
    def rho_cr(self, z: Tensor, x: Optional[dict[str, Any]] = None) -> Tensor:
        ...

    @abstractmethod
    def comoving_dist(self, z: Tensor, x: Optional[dict[str, Any]] = None) -> Tensor:
        ...

    def comoving_dist_z1z2(
        self, z1: Tensor, z2: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        return self.comoving_dist(z2, x) - self.comoving_dist(z1, x)

    def angular_diameter_dist(
        self, z: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        return self.comoving_dist(z, x) / (1 + z)

    def angular_diameter_dist_z1z2(
        self, z1: Tensor, z2: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        return self.comoving_dist_z1z2(z1, z2, x) / (1 + z2)

    def time_delay_dist(
        self, z_l: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        d_l = self.angular_diameter_dist(z_l, x)
        d_s = self.angular_diameter_dist(z_s, x)
        d_ls = self.angular_diameter_dist_z1z2(z_l, z_s, x)
        return (1 + z_l) * d_l * d_s / d_ls

    def Sigma_cr(
        self, z_l: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        d_l = self.angular_diameter_dist(z_l, x)
        d_s = self.angular_diameter_dist(z_s, x)
        d_ls = self.angular_diameter_dist_z1z2(z_l, z_s, x)
        return d_s / d_l / d_ls / (4 * pi * G_over_c2)


class FlatLambdaCDM(Cosmology):
    """

    Flat LCDM cosmology with no radiation.
    """

    def __init__(
        self,
        name: str,
        h0: Optional[Tensor] = torch.tensor(h0_default),
        rho_cr_0: Optional[Tensor] = torch.tensor(rho_cr_0_default),
        Om0: Optional[Tensor] = torch.tensor(Om0_default),
    ):
        super().__init__(name)

        self.add_param("h0", h0)
        self.add_param("rho_cr_0", rho_cr_0)
        self.add_param("Om0", Om0)

        self._comoving_dist_helper_x_grid = _comoving_dist_helper_x_grid.to(
            dtype=torch.float32
        )
        self._comoving_dist_helper_y_grid = _comoving_dist_helper_y_grid.to(
            dtype=torch.float32
        )

    def dist_hubble(self, h0):
        return c_Mpc_s / (100 * km_to_Mpc) / h0

    def rho_cr(self, z: Tensor, x: Optional[dict[str, Any]] = None) -> torch.Tensor:
        _, rho_cr_0, Om0 = self.unpack(x)
        Ode0 = 1 - Om0
        return rho_cr_0 * (Om0 * (1 + z) ** 3 + Ode0)

    def _comoving_dist_helper(self, x: Tensor) -> Tensor:
        return interp1d(
            self._comoving_dist_helper_x_grid,
            self._comoving_dist_helper_y_grid,
            torch.atleast_1d(x),
        ).reshape(x.shape)

    def comoving_dist(self, z: Tensor, x: Optional[dict[str, Any]] = None) -> Tensor:
        h0, _, Om0 = self.unpack(x)

        Ode0 = 1 - Om0
        ratio = (Om0 / Ode0) ** (1 / 3)
        return (
            self.dist_hubble(h0)
            * (
                self._comoving_dist_helper((1 + z) * ratio)
                - self._comoving_dist_helper(ratio)
            )
            / (Om0 ** (1 / 3) * Ode0 ** (1 / 6))
        )
