from math import pi

import equinox as eqx
import jax.numpy as jnp
from astropy.cosmology import default_cosmology
from scipy.special import hyp2f1

from .utils import C_MPC_S, G_OVER_C2, KM_TO_MPC, Array

_H0_DEFAULT = float(default_cosmology.get().h)
_RHO_CR_0_DEFAULT = float(
    default_cosmology.get().critical_density(0).to("solMass/Mpc^3").value
)
_OM0_DEFAULT = float(default_cosmology.get().Om0)
_X_GRID = jnp.geomspace(0.001, 10, 500)
_F_GRID = _X_GRID * hyp2f1(1 / 3, 1 / 2, 4 / 3, -(_X_GRID**3))


def _f(x):
    return jnp.interp(x, _X_GRID, _F_GRID)


class Cosmology(eqx.Module):
    """
    Flat LCDM cosmology with no radiation.

    Notes:
        Compute comoving_dist on the fly
    """

    h0: Array
    rho_cr_0: Array
    Om0: Array
    Ode0: Array

    def __init__(
        self,
        h0=jnp.array(_H0_DEFAULT),
        rho_cr_0=jnp.array(_RHO_CR_0_DEFAULT),
        Om0=jnp.array(_OM0_DEFAULT),
        Ode0=None,
        Or0=None,
    ):
        super().__init__()
        if Ode0 is None:
            self.Ode0 = -Om0 + 1
        else:
            raise NotImplementedError(
                "Since only flat cosmologies are currently supported, Ode0 must be 'None'"
            )

        if Or0 is not None:
            raise NotImplementedError(
                "Since only flat cosmologies are currently supported, Or0 must be 'None'"
            )

        self.h0 = h0
        self.rho_cr_0 = rho_cr_0
        self.Om0 = Om0

    @property
    def dist_hubble(self):
        return C_MPC_S / (100 * KM_TO_MPC) / self.h0

    def critical_density(self, z):
        return self.rho_cr_0 * (self.Om0 * (1 + z) ** 3 + (1 - self.Om0))

    def comoving_dist(self, z):
        """
        Comoving distance for default flat, default astropy cosmology [Mpc].
        """
        ratio = (self.Om0 / self.Ode0) ** (1 / 3)
        return (
            self.dist_hubble
            * (_f((1 + z) * ratio) - _f(ratio))
            / (self.Om0 ** (1 / 3) * self.Ode0 ** (1 / 6))
        )

    def comoving_dist_z1z2(self, z1, z2):
        return self.comoving_dist(z2) - self.comoving_dist(z1)

    def angular_diameter_dist(self, z):
        """
        Angular diameter distance in a flat, default astropy cosmology [Mpc].
        """
        return self.comoving_dist(z) / (1 + z)

    def angular_diameter_dist_z1z2(self, z1, z2):
        """
        Angular diameter distance between objects at two redshifts in the flat,
        default astropy cosmology [Mpc].
        """
        return self.comoving_dist_z1z2(z1, z2) / (1 + z2)

    def Sigma_cr(self, z_lens, z_src):
        """
        Critical density [solMass/Mpc^2].
        """
        dl = self.angular_diameter_dist(z_lens)  # Mpc
        ds = self.angular_diameter_dist(z_src)  # Mpc
        dls = self.angular_diameter_dist_z1z2(z_lens, z_src)  # Mpc
        return 1 / (4 * pi * G_OVER_C2 * dl * dls / ds)  # 1 / (Mpc * G / c**2)

    def time_delay_dist(self, z_lens, z_src):
        """
        Time delay distance between a lens and source [Mpc].
        """
        d_lens = self.angular_diameter_dist(z_lens)
        d_src = self.angular_diameter_dist(z_src)
        d_lens_src = self.angular_diameter_dist_z1z2(z_lens, z_src)
        return (1 + z_lens) * d_lens * d_src / d_lens_src
