from jax.config import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np

from caustic.cosmology import Cosmology
from caustic.lenses.utils import to_alpha, multiplane_alpha
from caustic.lenses.nfw import pot


def test_defl_rec():
    """
    Checks multiplane raytracing of NFW lenses against for loop calculation.
    """
    n_nfws = 10
    x, y = np.random.randn(2)

    x0s = jnp.array(np.random.randn(n_nfws))
    y0s = jnp.array(np.random.randn(n_nfws))
    zs = jnp.array(2.0 * np.sort(np.random.rand(n_nfws)))

    pot_nfw = lambda x, y, x0, y0, z: pot(
        x, y, x0, y0, m_200=1e10, c=15.0, z=z, z_src=2.5, cosmo=Cosmology()
    )
    alpha_nfw = to_alpha(pot_nfw)

    # caustic
    alpha_nfws = multiplane_alpha(jax.jit(alpha_nfw))
    res_jl = alpha_nfws(x, y, x0s, y0s, zs)

    # Manual calculation
    res_ref = jnp.array((x, y))
    for i in range(n_nfws):
        res_ref -= alpha_nfw(res_ref[0], res_ref[1], x0s[i], y0s[i], zs[i])

    assert jnp.allclose(res_jl, res_ref)
