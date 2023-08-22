from .base import Cosmology
from typing import Any, Optional
from phytorch.cosmology.drivers.analytic import LambdaCDM as phy_LCDM
from phytorch.units import si, astro
from astropy.cosmology import default_cosmology

import torch
from torch import Tensor

from ..constants import G_over_c2, c_Mpc_s, km_to_Mpc
from ..parametrized import unpack

__all__ = (
    "LambdaCDM",
)

#Default values for parameters, from astropy
H0_default = float(default_cosmology.get().h)*100.
Om0_default = float(default_cosmology.get().Om0)
Ob0_default = float(default_cosmology.get().Ob0)
Ode0_default = float(default_cosmology.get().Ode0)

class LambdaCDM(Cosmology):
    """
    Subclass of Cosmology representing a Lambda Cold Dark Matter (LCDM) cosmology with no radiation.
    """

    def __init__(
        self,
        H0: Optional[Tensor] = torch.tensor(H0_default),
        Om0: Optional[Tensor] = torch.tensor(Om0_default),
        Ob0: Optional[Tensor] = torch.tensor(Ob0_default),
        Ode0: Optional[Tensor] = torch.tensor(Ode0_default),
        name: str = None,
    ):
        """
        Initialize a new instance of the LambdaCDM class.

        Args:
            name (str): Name of the cosmology.
            H0 (Optional[Tensor]): Hubble constant over 100. Default is H0_default from Astropy.
            Om0 (Optional[Tensor]): Matter density parameter at z=0. Default is Om0_default from Astropy.
            Ob0 (Optional[Tensor]): Baryon density parameter at z=0. Default is Ob0_default from Astropy.
            Ode0 (Optional[Tensor]): Dark Energy density parameter at z=0. Default is Ode0_default from Astropy.
        """
        super().__init__(name = name)

        self.add_param("H0", H0)
        self.add_param("Om0", Om0)
        self.add_param("Ob0", Ob0)
        self.add_param("Ode0", Ode0)
        self.cosmo = phy_LCDM(H0 = H0, Om0 = Om0, Ob0 = Ob0, Ode0 = Ode0)

    def hubble_distance(self, H0):
        """
        Calculate the Hubble distance in Mpc.

        Args:
            h0 (Tensor): Hubble constant.

        Returns:
            Tensor: Hubble distance in Mpc.
        """
        self.cosmo.H0 = H0 * si.km / si.s / astro.Mpc
        return self.cosmo.hubble_distance.to(astro.Mpc)

    @unpack(1)
    def critical_density(self, z: Tensor, H0, Om0, Ob0, Ode0, *args, params: Optional["Packed"] = None) -> torch.Tensor:
        """
        Calculate the critical density at redshift z.

        Args:
            z (Tensor): Redshift.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            torch.Tensor: Critical density at redshift z.
        """
        self.cosmo.H0 = H0 * si.km / si.s / astro.Mpc
        self.cosmo.Om0 = Om0
        self.cosmo.Ob0 = Ob0
        self.cosmo.Ode0 = Ode0
        return self.cosmo.critical_density(z)

    @unpack(1)
    def comoving_distance(self, z: Tensor, H0, Om0, Ob0, Ode0, *args, params: Optional["Packed"] = None) -> Tensor:
        """
        Calculate the comoving distance to redshift z.

        Args:
            z (Tensor): Redshift.
            params (Packed, optional): Dynamic parameter container for the computation.

        Returns:
            Tensor: Comoving distance to redshift z.
        """
        self.cosmo.H0 = H0 * si.km / si.s / astro.Mpc
        self.cosmo.Om0 = Om0
        self.cosmo.Ob0 = Ob0
        self.cosmo.Ode0 = Ode0
        return self.cosmo.comoving_distance_dimless(z)*self.hubble_distance(H0)
