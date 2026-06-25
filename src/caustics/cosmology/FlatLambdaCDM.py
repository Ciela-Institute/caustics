# mypy: disable-error-code="operator"
from typing import Optional, Annotated

from caskade import forward, Param
from astropy.cosmology import default_cosmology
from scipy.special import hyp2f1

from ..utils import interp1d
from ..backend_obj import backend, ArrayLike
from ..constants import c_Mpc_s, km_to_Mpc
from .base import Cosmology, NameType

_h0_default = float(default_cosmology.get().h)
_critical_density_0_default = float(
    default_cosmology.get().critical_density(0).to("solMass/Mpc^3").value
)
_Om0_default = float(default_cosmology.get().Om0)


class FlatLambdaCDM(Cosmology):
    """
    Subclass of Cosmology representing a Flat Lambda Cold Dark Matter (LCDM)
    cosmology with no radiation.
    """

    def __init__(
        self,
        h0: Annotated[
            Optional[ArrayLike], "Hubble constant over 100", True
        ] = _h0_default,
        critical_density_0: Annotated[
            Optional[ArrayLike], "Critical density at z=0", True
        ] = _critical_density_0_default,
        Om0: Annotated[
            Optional[ArrayLike], "Matter density parameter at z=0", True
        ] = _Om0_default,
        name: NameType = None,
    ):
        """
        Initialize a new instance of the FlatLambdaCDM class.

        Parameters
        ----------
        name: str
        Name of the cosmology.
        h0: Optional[ArrayLike]
            Hubble constant over 100. Default is h0_default.
        critical_density_0: (Optional[ArrayLike])
            Critical density at z=0. Default is critical_density_0_default.
        Om0: Optional[ArrayLike]
            Matter density parameter at z=0. Default is Om0_default.
        """
        super().__init__(name)

        self.h0 = Param("h0", h0, units="unitless", valid=(0, None))
        self.critical_density_0 = Param(
            "critical_density_0",
            critical_density_0,
            units="Msun/Mpc^3",
            valid=(0, None),
        )
        self.Om0 = Param("Om0", Om0, units="unitless", valid=(0, 1))

        # Set up interpolator to speed up comoving distance calculations in Lambda-CDM
        # cosmologies. Construct with float64 precision.
        _comoving_distance_helper_x_grid = 10 ** backend.linspace(
            -3, 1, 500, dtype=backend.float64
        )
        _comoving_distance_helper_y_grid = backend.as_array(
            _comoving_distance_helper_x_grid
            * hyp2f1(1 / 3, 1 / 2, 4 / 3, -(_comoving_distance_helper_x_grid**3)),
            dtype=backend.float64,
        )
        self._comoving_distance_helper_x_grid = backend.to(
            _comoving_distance_helper_x_grid, dtype=backend.float32
        )
        self._comoving_distance_helper_y_grid = backend.to(
            _comoving_distance_helper_y_grid, dtype=backend.float32
        )

    def to(self, device=None, dtype=None):
        super().to(device, dtype)
        self._comoving_distance_helper_y_grid = backend.to(
            self._comoving_distance_helper_y_grid, device=device, dtype=dtype
        )
        self._comoving_distance_helper_x_grid = backend.to(
            self._comoving_distance_helper_x_grid, device=device, dtype=dtype
        )

        return self

    def hubble_distance(self, h0: Annotated[ArrayLike, "Param"]):
        """
        Calculate the Hubble distance.

        Parameters
        ----------
        h0: ArrayLike
            Hubble constant.

        Returns
        -------
        ArrayLike
            Hubble distance.
        """
        return c_Mpc_s / (100 * km_to_Mpc) / h0

    @forward
    def critical_density(
        self,
        z: ArrayLike,
        critical_density_0: Annotated[ArrayLike, "Param"],
        Om0: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Calculate the critical density at redshift z.

        Parameters
        ----------
        z: ArrayLike
            Redshift.

        Returns
        -------
        ArrayLike
            Critical density at redshift z.
        """
        Ode0 = 1 - Om0
        return critical_density_0 * (Om0 * (1 + z) ** 3 + Ode0)  # fmt: skip

    @forward
    def _comoving_distance_helper(self, x: ArrayLike) -> ArrayLike:
        """
        Helper method for computing comoving distances.

        Parameters
        ----------
        x: ArrayLike
            Input tensor.

        Returns
        -------
        ArrayLike
            Computed comoving distances.
        """
        return backend.view(
            interp1d(
                self._comoving_distance_helper_x_grid,
                self._comoving_distance_helper_y_grid,
                backend.atleast_1d(x),
            ),
            x.shape,
        )

    @forward
    def comoving_distance(
        self,
        z: ArrayLike,
        h0: Annotated[ArrayLike, "Param"],
        Om0: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Calculate the comoving distance to redshift z.

        Parameters
        ----------
        z: ArrayLike
            Redshift.

        Returns
        -------
        ArrayLike
            Comoving distance to redshift z.
        """
        Ode0 = 1 - Om0
        ratio = (Om0 / Ode0) ** (1 / 3)
        DH = self.hubble_distance(h0)
        DC1z = self._comoving_distance_helper((1 + z) * ratio)
        DC = self._comoving_distance_helper(ratio)
        return DH * (DC1z - DC) / (Om0 ** (1 / 3) * Ode0 ** (1 / 6))  # fmt: skip

    @forward
    def transverse_comoving_distance(self, z: ArrayLike) -> ArrayLike:
        return self.comoving_distance(z)
