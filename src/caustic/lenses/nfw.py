from math import pi

import jax
import jax.numpy as jnp


def _rho_s(c, rho_cr):
    return rho_cr * 1 / 3 * c**3 / (jnp.log(1 + c) - c / (1 + c))


def _r_s(m_200, c, rho_cr):
    return 1 / c * (3 * m_200 / (4 * pi * 200 * rho_cr)) ** (1 / 3)


def _th_s_arcsec(r_s, z, cosmo):
    # Scale radius [arcsec]
    dist_ang_lens = cosmo.angular_diameter_dist(z)
    return r_s / dist_ang_lens * (180 / pi * 60**2)


def _kappa_s(rho_s, r_s, z, z_src, cosmo):
    # Normalization parameter
    Sigma_cr = cosmo.Sigma_cr(z, z_src)
    return rho_s * r_s / Sigma_cr


def _g(s):
    fn_gt1 = lambda s: 2 * jnp.arctan(jnp.sqrt((s - 1) / (s + 1))) ** 2
    fn_lt1 = lambda s: -2 * jnp.arctanh(jnp.sqrt((1 - s) / (1 + s))) ** 2
    fn_neq = lambda s: jax.lax.cond(s > 1, fn_gt1, fn_lt1, s)
    fn_eq = lambda _: 0.0
    return 1 / 2 * jnp.log(s / 2) ** 2 + jax.lax.cond(s == 0, fn_eq, fn_neq, s)


def pot(x, y, x0, y0, m_200, c, z, z_src, cosmo):
    rho_cr = cosmo.critical_density(z)
    r_s = _r_s(m_200, c, rho_cr)
    th_s_arcsec = _th_s_arcsec(r_s, z, cosmo)
    rho_s = _rho_s(c, rho_cr)
    kappa_s = _kappa_s(rho_s, r_s, z, z_src, cosmo)

    x = x - x0
    y = y - y0
    s = jnp.sqrt(x**2 + y**2) / th_s_arcsec

    return 4 * kappa_s * _g(s)
