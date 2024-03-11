# mypy: disable-error-code="operator"
from typing import Optional, Annotated

import torch
from torch import Tensor

from astropy.cosmology import default_cosmology
from scipy.special import hyp2f1

from ..utils import interp1d
from ..parametrized import unpack
from ..packed import Packed
from ..constants import c_Mpc_s, km_to_Mpc
from .base import Cosmology, NameType

_h0_default = float(default_cosmology.get().h)
_critical_density_0_default = float(
    default_cosmology.get().critical_density(0).to("solMass/Mpc^3").value
)
_Om0_default = float(default_cosmology.get().Om0)

# Set up interpolator to speed up comoving distance calculations in Lambda-CDM
# cosmologies. Construct with float64 precision.
_comoving_distance_helper_x_grid = 10 ** torch.linspace(-3, 1, 500, dtype=torch.float64)
_comoving_distance_helper_y_grid = torch.as_tensor(
    _comoving_distance_helper_x_grid
    * hyp2f1(1 / 3, 1 / 2, 4 / 3, -(_comoving_distance_helper_x_grid**3)),
    dtype=torch.float64,
)

h0_default = torch.tensor(_h0_default)
critical_density_0_default = torch.tensor(_critical_density_0_default)
Om0_default = torch.tensor(_Om0_default)


class FlatLambdaCDM(Cosmology):
    """
    Subclass of Cosmology representing a Flat Lambda Cold Dark Matter (LCDM)
    cosmology with no radiation.
    """

    def __init__(
        self,
        h0: Annotated[Optional[Tensor], "Hubble constant over 100", True] = h0_default,
        critical_density_0: Annotated[
            Optional[Tensor], "Critical density at z=0", True
        ] = critical_density_0_default,
        Om0: Annotated[
            Optional[Tensor], "Matter density parameter at z=0", True
        ] = Om0_default,
        name: NameType = None,
    ):
        """
        Initialize a new instance of the FlatLambdaCDM class.

        Parameters
        ----------
        name: str
        Name of the cosmology.
        h0: Optional[Tensor]
            Hubble constant over 100. Default is h0_default.
        critical_density_0: (Optional[Tensor])
            Critical density at z=0. Default is critical_density_0_default.
        Om0: Optional[Tensor]
            Matter density parameter at z=0. Default is Om0_default.
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

        return self

    def hubble_distance(self, h0):
        """
        Calculate the Hubble distance.

        Parameters
        ----------
        h0: Tensor
            Hubble constant.

        Returns
        -------
        Tensor
            Hubble distance.
        """
        return c_Mpc_s / (100 * km_to_Mpc) / h0

    @unpack
    def critical_density(
        self,
        z: Tensor,
        *args,
        params: Optional["Packed"] = None,
        h0: Optional[Tensor] = None,
        critical_density_0: Optional[Tensor] = None,
        Om0: Optional[Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculate the critical density at redshift z.

        Parameters
        ----------
        z: Tensor
            Redshift.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        torch.Tensor
            Critical density at redshift z.
        """
        Ode0 = 1 - Om0
        return critical_density_0 * (Om0 * (1 + z) ** 3 + Ode0)  # fmt: skip

    @unpack
    def _comoving_distance_helper(
        self, x: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Helper method for computing comoving distances.

        Parameters
        ----------
        x: Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Computed comoving distances.
        """
        return interp1d(
            self._comoving_distance_helper_x_grid,
            self._comoving_distance_helper_y_grid,
            torch.atleast_1d(x),
        ).reshape(x.shape)

    @unpack
    def comoving_distance(
        self,
        z: Tensor,
        *args,
        params: Optional["Packed"] = None,
        h0: Optional[Tensor] = None,
        critical_density_0: Optional[Tensor] = None,
        Om0: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the comoving distance to redshift z.

        Parameters
        ----------
        z: Tensor
            Redshift.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            Comoving distance to redshift z.
        """
        Ode0 = 1 - Om0
        ratio = (Om0 / Ode0) ** (1 / 3)
        DH = self.hubble_distance(h0)
        DC1z = self._comoving_distance_helper((1 + z) * ratio, params)
        DC = self._comoving_distance_helper(ratio, params)
        return DH * (DC1z - DC) / (Om0 ** (1 / 3) * Ode0 ** (1 / 6))  # fmt: skip

    @unpack
    def transverse_comoving_distance(
        self,
        z: Tensor,
        *args,
        params: Optional["Packed"] = None,
        h0: Optional[Tensor] = None,
        critical_density_0: Optional[Tensor] = None,
        Om0: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        return self.comoving_distance(z, params, **kwargs)
