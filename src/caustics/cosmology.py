from abc import abstractmethod
from math import pi
from typing import Optional

import torch
from astropy.cosmology import default_cosmology
from scipy.special import hyp2f1
from torch import Tensor

from .utils import interp1d
from .constants import G_over_c2, c_Mpc_s, km_to_Mpc
from .parametrized import Parametrized, unpack

__all__ = (
    "h0_default",
    "critical_density_0_default",
    "Om0_default",
    "Cosmology",
    "FlatLambdaCDM",
)

h0_default = float(default_cosmology.get().h)
critical_density_0_default = float(
    default_cosmology.get().critical_density(0).to("solMass/Mpc^3").value
)
Om0_default = float(default_cosmology.get().Om0)

# Set up interpolator to speed up comoving distance calculations in Lambda-CDM
# cosmologies. Construct with float64 precision.
_comoving_distance_helper_x_grid = 10 ** torch.linspace(-3, 1, 500, dtype=torch.float64)
_comoving_distance_helper_y_grid = torch.as_tensor(
    _comoving_distance_helper_x_grid
    * hyp2f1(1 / 3, 1 / 2, 4 / 3, -(_comoving_distance_helper_x_grid**3)),
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

    def __init__(self, name: str = None):
        """
        Initialize the Cosmology.

        Args:
            name (str): Name of the cosmological model.
        """
        super().__init__(name)

    @abstractmethod
    def critical_density(self, z: Tensor, params: Optional["Packed"] = None) -> Tensor:
        """
        Compute the critical density at redshift z.

        Args:
            z (Tensor): The redshifts.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            Tensor: The critical density at each redshift.
        """
        ...

    @abstractmethod
    @unpack(1)
    def comoving_distance(
        self, z: Tensor, *args, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the comoving distance to redshift z.

        Args:
            z (Tensor): The redshifts.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            Tensor: The comoving distance to each redshift.
        """
        ...

    @abstractmethod
    @unpack(1)
    def transverse_comoving_distance(
        self, z: Tensor, *args, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the transverse comoving distance to redshift z (Mpc).

        Args:
            z (Tensor): The redshifts.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            Tensor: The transverse comoving distance to each redshift in Mpc.
        """
        ...

    @unpack(2)
    def comoving_distance_z1z2(
        self, z1: Tensor, z2: Tensor, *args, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the comoving distance between two redshifts.

        Args:
            z1 (Tensor): The starting redshifts.
            z2 (Tensor): The ending redshifts.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            Tensor: The comoving distance between each pair of redshifts.
        """
        return self.comoving_distance(z2, params) - self.comoving_distance(z1, params)

    @unpack(2)
    def transverse_comoving_distance_z1z2(
        self, z1: Tensor, z2: Tensor, *args, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the transverse comoving distance between two redshifts (Mpc).

        Args:
            z1 (Tensor): The starting redshifts.
            z2 (Tensor): The ending redshifts.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            Tensor: The transverse comoving distance between each pair of redshifts in Mpc.
        """
        return self.transverse_comoving_distance(
            z2, params
        ) - self.transverse_comoving_distance(z1, params)

    @unpack(1)
    def angular_diameter_distance(
        self, z: Tensor, *args, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the angular diameter distance to redshift z.

        Args:
            z (Tensor): The redshifts.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            Tensor: The angular diameter distance to each redshift.
        """
        return self.comoving_distance(z, params) / (1 + z)

    @unpack(2)
    def angular_diameter_distance_z1z2(
        self, z1: Tensor, z2: Tensor, *args, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the angular diameter distance between two redshifts.

        Args:
            z1 (Tensor): The starting redshifts.
            z2 (Tensor): The ending redshifts.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            Tensor: The angular diameter distance between each pair of redshifts.
        """
        return self.comoving_distance_z1z2(z1, z2, params) / (1 + z2)

    @unpack(2)
    def time_delay_distance(
        self, z_l: Tensor, z_s: Tensor, *args, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the time delay distance between lens and source planes.

        Args:
            z_l (Tensor): The lens redshifts.
            z_s (Tensor): The source redshifts.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            Tensor: The time delay distance for each pair of lens and source redshifts.
        """
        d_l = self.angular_diameter_distance(z_l, params)
        d_s = self.angular_diameter_distance(z_s, params)
        d_ls = self.angular_diameter_distance_z1z2(z_l, z_s, params)
        return (1 + z_l) * d_l * d_s / d_ls

    @unpack(2)
    def critical_surface_density(
        self, z_l: Tensor, z_s: Tensor, *args, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the critical surface density between lens and source planes.

        Args:
            z_l (Tensor): The lens redshifts.
            z_s (Tensor): The source redshifts.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            Tensor: The critical surface density for each pair of lens and source redshifts.
        """
        d_l = self.angular_diameter_distance(z_l, params)
        d_s = self.angular_diameter_distance(z_s, params)
        d_ls = self.angular_diameter_distance_z1z2(z_l, z_s, params)
        return d_s / (4 * pi * G_over_c2 * d_l * d_ls)


class FlatLambdaCDM(Cosmology):
    """
    Subclass of Cosmology representing a Flat Lambda Cold Dark Matter (LCDM) cosmology with no radiation.
    """

    def __init__(
        self,
        h0: Optional[Tensor] = torch.tensor(h0_default),
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

        self.add_param("h0", h0)
        self.add_param("critical_density_0", critical_density_0)
        self.add_param("Om0", Om0)

        self._comoving_distance_helper_x_grid = _comoving_distance_helper_x_grid.to(
            dtype=torch.float32
        )
        self._comoving_distance_helper_y_grid = _comoving_distance_helper_y_grid.to(
            dtype=torch.float32
        )

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        super().to(device, dtype)
        self._comoving_distance_helper_y_grid = (
            self._comoving_distance_helper_y_grid.to(device, dtype)
        )
        self._comoving_distance_helper_x_grid = (
            self._comoving_distance_helper_x_grid.to(device, dtype)
        )

    def hubble_distance(self, h0):
        """
        Calculate the Hubble distance.

        Args:
            h0 (Tensor): Hubble constant.

        Returns:
            Tensor: Hubble distance.
        """
        return c_Mpc_s / (100 * km_to_Mpc) / h0

    @unpack(1)
    def critical_density(
        self,
        z: Tensor,
        h0,
        central_critical_density,
        Om0,
        *args,
        params: Optional["Packed"] = None,
    ) -> torch.Tensor:
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
    def _comoving_distance_helper(
        self, x: Tensor, *args, params: Optional["Packed"] = None
    ) -> Tensor:
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
    def comoving_distance(
        self,
        z: Tensor,
        h0,
        central_critical_density,
        Om0,
        *args,
        params: Optional["Packed"] = None,
    ) -> Tensor:
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
            self.hubble_distance(h0)
            * (
                self._comoving_distance_helper((1 + z) * ratio, params)
                - self._comoving_distance_helper(ratio, params)
            )
            / (Om0 ** (1 / 3) * Ode0 ** (1 / 6))
        )

    @unpack(1)
    def transverse_comoving_distance(
        self,
        z: Tensor,
        h0,
        central_critical_density,
        Om0,
        *args,
        params: Optional["Packed"] = None,
    ) -> Tensor:
        return self.comoving_distance(z, params)
