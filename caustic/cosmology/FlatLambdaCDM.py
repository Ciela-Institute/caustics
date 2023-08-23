from .base import Cosmology
from typing import Any, Optional

import torch
from astropy.cosmology import default_cosmology
from scipy.special import hyp2f1
from torch import Tensor

from ..utils import interp1d
from ..constants import G_over_c2, c_Mpc_s, km_to_Mpc
from ..parametrized import Parametrized, unpack

__all__ = (
    "FlatLambdaCDM",
)

H0_default = float(default_cosmology.get().h)*100
critical_density_0_default = float(
    default_cosmology.get().critical_density(0).to("solMass/Mpc^3").value
)
Om0_default = float(default_cosmology.get().Om0)

# Set up interpolator to speed up comoving distance calculations in Lambda-CDM
# cosmologies. Construct with float64 precision.
_comoving_distance_helper_x_grid = 10 ** torch.linspace(-3, 1, 500, dtype=torch.float64)
_comoving_distance_helper_y_grid = torch.as_tensor(
    _comoving_distance_helper_x_grid
    * hyp2f1(1 / 3, 1 / 2, 4 / 3, -(_comoving_distance_helper_x_grid ** 3)),
    dtype=torch.float64,
)

class FlatLambdaCDM(Cosmology):
    """
    Subclass of Cosmology representing a Flat Lambda Cold Dark Matter (LCDM) cosmology with no radiation.
    NOTE: THIS IS THE VERSION WITHOUT GRADIENTS AND WHICH USES AN APPROXIMATION SCHEME. It is kept for backwards compatibility.
    """

    def __init__(
        self,
        H0: Optional[Tensor] = torch.tensor(H0_default),
        critical_density_0: Optional[Tensor] = torch.tensor(critical_density_0_default),
        Om0: Optional[Tensor] = torch.tensor(Om0_default),
        name: str = None,
    ):
        """
        Initialize a new instance of the FlatLambdaCDM class.

        Args:
            name (str): Name of the cosmology.
            h0 (Optional[Tensor]): Hubble constant over 100. Default is h0_default.
            critical_density_0 (Optional[Tensor]): Critical density at z=0. Default is critical_density_0_default.
            Om0 (Optional[Tensor]): Matter density parameter at z=0. Default is Om0_default.
        """
        super().__init__(name)

        self.add_param("H0", H0)
        self.add_param("critical_density_0", critical_density_0)
        self.add_param("Om0", Om0)

        self._comoving_distance_helper_x_grid = _comoving_distance_helper_x_grid.to(
            dtype=torch.float32
        )
        self._comoving_distance_helper_y_grid = _comoving_distance_helper_y_grid.to(
            dtype=torch.float32
        )

    def hubble_distance(self, H0):
        """
        Calculate the Hubble distance.

        Args:
            h0 (Tensor): Hubble constant.

        Returns:
            Tensor: Hubble distance.
        """
        return c_Mpc_s / (km_to_Mpc) / H0

    @unpack(1)
    def critical_density(self, z: Tensor, H0, central_critical_density, Om0, *args, params: Optional["Packed"] = None) -> torch.Tensor:
        """
        Calculate the critical density at redshift z.

        Args:
            z (Tensor): Redshift.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            torch.Tensor: Critical density at redshift z.
        """
        Ode0 = 1 - Om0
        return central_critical_density * (Om0 * (1 + z) ** 3 + Ode0)

    @unpack(1)
    def _comoving_distance_helper(self, x: Tensor, *args, params: Optional["Packed"] = None) -> Tensor:
        """
        Helper method for computing comoving distances.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Computed comoving distances.
        """
        return interp1d(
            self._comoving_distance_helper_x_grid,
            self._comoving_distance_helper_y_grid,
            torch.atleast_1d(x),
        ).reshape(x.shape)

    @unpack(1)
    def comoving_distance(self, z: Tensor, H0, central_critical_density, Om0, *args, params: Optional["Packed"] = None) -> Tensor:
        """
        Calculate the comoving distance to redshift z.

        Args:
            z (Tensor): Redshift.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            Tensor: Comoving distance to redshift z.
        """
        Ode0 = 1 - Om0
        ratio = (Om0 / Ode0) ** (1 / 3)
        return (
            self.hubble_distance(H0)
            * (
                self._comoving_distance_helper((1 + z) * ratio, params)
                - self._comoving_distance_helper(ratio, params)
            )
            / (Om0 ** (1 / 3) * Ode0 ** (1 / 6))
        )
