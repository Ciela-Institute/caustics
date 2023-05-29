from abc import abstractmethod
from math import pi
from typing import Any, Optional

import torch
from astropy.cosmology import default_cosmology
from scipy.special import hyp2f1
from torch import Tensor

from .utils import interp1d
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
    Abstract base class for cosmological models.

    This class provides an interface for cosmological computations used in lensing 
    such as comoving distance and critical surface density. 

    Units:
        - Distance: Mpc
        - Mass: solar mass

    Attributes:
        name (str): Name of the cosmological model.
    """

    def __init__(self, name: str):
        """
        Initialize the Cosmology.

        Args:
            name (str): Name of the cosmological model.
        """
        super().__init__(name)

    @abstractmethod
    def rho_cr(self, z: Tensor, x: Optional[dict[str, Any]] = None) -> Tensor:
        """
        Compute the critical density at redshift z.

        Args:
            z (Tensor): The redshifts.
            x (Optional[dict[str, Any]]): Additional parameters for the computation.

        Returns:
            Tensor: The critical density at each redshift.
        """
        ...

    @abstractmethod
    def comoving_dist(self, z: Tensor, x: Optional[dict[str, Any]] = None) -> Tensor:
        """
        Compute the comoving distance to redshift z.

        Args:
            z (Tensor): The redshifts.
            x (Optional[dict[str, Any]]): Additional parameters for the computation.

        Returns:
            Tensor: The comoving distance to each redshift.
        """
        ...

    def comoving_dist_z1z2(
        self, z1: Tensor, z2: Tensor, x: Optional[dict[str, Any]] = None) -> Tensor:
        """
        Compute the comoving distance between two redshifts.

        Args:
            z1 (Tensor): The starting redshifts.
            z2 (Tensor): The ending redshifts.
            x (Optional[dict[str, Any]]): Additional parameters for the computation.

        Returns:
            Tensor: The comoving distance between each pair of redshifts.
        """
        return self.comoving_dist(z2, x) - self.comoving_dist(z1, x)

    def angular_diameter_dist(
        self, z: Tensor, x: Optional[dict[str, Any]] = None) -> Tensor:
        """
        Compute the angular diameter distance to redshift z.

        Args:
            z (Tensor): The redshifts.
            x (Optional[dict[str, Any]]): Additional parameters for the computation.

        Returns:
            Tensor: The angular diameter distance to each redshift.
        """
        return self.comoving_dist(z, x) / (1 + z)

    def angular_diameter_dist_z1z2(
        self, z1: Tensor, z2: Tensor, x: Optional[dict[str, Any]] = None) -> Tensor:
        """
        Compute the angular diameter distance between two redshifts.

        Args:
            z1 (Tensor): The starting redshifts.
            z2 (Tensor): The ending redshifts.
            x (Optional[dict[str, Any]]): Additional parameters for the computation.

        Returns:
            Tensor: The angular diameter distance between each pair of redshifts.
        """
        return self.comoving_dist_z1z2(z1, z2, x) / (1 + z2)

    def time_delay_dist(
        self, z_l: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None) -> Tensor:
        """
        Compute the time delay distance between lens and source planes.

        Args:
            z_l (Tensor): The lens redshifts.
            z_s (Tensor): The source redshifts.
            x (Optional[dict[str, Any]]): Additional parameters for the computation.

        Returns:
            Tensor: The time delay distance for each pair of lens and source redshifts.
        """
        d_l = self.angular_diameter_dist(z_l, x)
        d_s = self.angular_diameter_dist(z_s, x)
        d_ls = self.angular_diameter_dist_z1z2(z_l, z_s, x)
        return (1 + z_l) * d_l * d_s / d_ls

    def Sigma_cr(
        self, z_l: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None) -> Tensor:
        """
        Compute the critical surface density between lens and source planes.

        Args:
            z_l (Tensor): The lens redshifts.
            z_s (Tensor): The source redshifts.
            x (Optional[dict[str, Any]]): Additional parameters for the computation.

        Returns:
            Tensor: The critical surface density for each pair of lens and source redshifts.
        """
        d_l = self.angular_diameter_dist(z_l, x)
        d_s = self.angular_diameter_dist(z_s, x)
        d_ls = self.angular_diameter_dist_z1z2(z_l, z_s, x)
        return d_s / d_l / d_ls / (4 * pi * G_over_c2)


class FlatLambdaCDM(Cosmology):
    """
    Subclass of Cosmology representing a Flat Lambda Cold Dark Matter (LCDM) cosmology with no radiation.
    """
    
    def __init__(
        self,
        name: str,
        h0: Optional[Tensor] = torch.tensor(h0_default),
        rho_cr_0: Optional[Tensor] = torch.tensor(rho_cr_0_default),
        Om0: Optional[Tensor] = torch.tensor(Om0_default),
    ):
        """
        Initialize a new instance of the FlatLambdaCDM class.

        Args:
            name (str): Name of the cosmology.
            h0 (Optional[Tensor]): Hubble constant. Default is h0_default.
            rho_cr_0 (Optional[Tensor]): Critical density at z=0. Default is rho_cr_0_default.
            Om0 (Optional[Tensor]): Matter density parameter at z=0. Default is Om0_default.
        """
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
        """
        Calculate the Hubble distance.

        Args:
            h0 (Tensor): Hubble constant.

        Returns:
            Tensor: Hubble distance.
        """
        return c_Mpc_s / (100 * km_to_Mpc) / h0

    def rho_cr(self, z: Tensor, x: Optional[dict[str, Any]] = None) -> torch.Tensor:
        """
        Calculate the critical density at redshift z.

        Args:
            z (Tensor): Redshift.
            x (Optional[dict[str, Any]]): Additional parameters for the computation.

        Returns:
            torch.Tensor: Critical density at redshift z.
        """
        _, rho_cr_0, Om0 = self.unpack(x)
        Ode0 = 1 - Om0
        return rho_cr_0 * (Om0 * (1 + z) ** 3 + Ode0)

    def _comoving_dist_helper(self, x: Tensor) -> Tensor:
        """
        Helper method for computing comoving distances.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Computed comoving distances.
        """
        return interp1d(
            self._comoving_dist_helper_x_grid,
            self._comoving_dist_helper_y_grid,
            torch.atleast_1d(x),
        ).reshape(x.shape)

    def comoving_dist(self, z: Tensor, x: Optional[dict[str, Any]] = None) -> Tensor:
        """
        Calculate the comoving distance to redshift z.

        Args:
            z (Tensor): Redshift.
            x (Optional[dict[str, Any]]): Additional parameters for the computation.

        Returns:
            Tensor: Comoving distance to redshift z.
        """
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
