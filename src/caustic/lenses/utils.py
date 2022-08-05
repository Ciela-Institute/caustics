import jax
import jax.numpy as jnp

from ..cosmology import Cosmology
from ..utils import C_MPC_S


def to_alpha(pot):
    def get_alpha(x, y, *args, **kwargs):
        ax = jax.grad(pot, 0)(x, y, *args, **kwargs)
        ay = jax.grad(pot, 1)(x, y, *args, **kwargs)
        return jnp.stack((ax, ay))

    return get_alpha


def to_mag(alpha):
    def mag(x, y, *args, **kwargs):
        raytrace = lambda x_, y_, *args, **kwargs: jnp.stack(x_, y_) - alpha(
            x_, y_, *args, **kwargs
        )
        jac = jnp.stack(jax.jacfwd(raytrace, (0, 1))(x, y, *args, **kwargs))
        return 1 / jnp.abs(jnp.linalg.det(jac))

    return mag


def to_fermat(pot, alpha):
    def fermat(x, y, *args, **kwargs):
        ax, ay = alpha(x, y, *args, **kwargs)
        return (ax**2 + ay**2) / 2 - pot(x, y, *args, **kwargs)

    return fermat


def alpha_to_kappa(alpha):
    def kappa(x, y, *args, **kwargs):
        dax_dx = jax.grad(lambda x_: alpha(x_, y, *args, **kwargs)[0])(x)
        day_dy = jax.grad(lambda y_: alpha(x, y_, *args, **kwargs)[1])(y)
        return (dax_dx + day_dy) / 2

    return kappa


def pot_to_kappa(pot):
    def kappa(x, y, *args, **kwargs):
        d2p_dx2 = jax.grad(jax.grad(lambda x_: pot(x_, y, *args, **kwargs)))(x)
        d2p_dy2 = jax.grad(jax.grad(lambda y_: pot(x, y_, *args, **kwargs)))(y)
        return (d2p_dx2 + d2p_dy2) / 2

    return kappa


def to_time_delay(pot, alpha, cosmo=Cosmology()):
    fermat = to_fermat(pot, alpha)

    def td(x, y, z, z_src, *args, **kwargs):
        dist_td = cosmo.time_delay_dist(z, z_src)
        return fermat(x, y, *args, **kwargs) * dist_td / C_MPC_S

    return td


def multiplane_alpha(alpha):
    """
    Performs multiple ray-tracing for lenses described by a given deflection field.

    Arguments and keyword arguments to the returned function must be sorted by
    redshift.

    For this to be relatively fast, `alpha` should be jitted. The returned function
    works with forward and reverse-mode AD. It's jittable, but uses a python `for`
    loop, and so will recompile every time the number of lenses changes.

    Notes:
        Kind of brittle -- e.g., requires all lenses to have the same deflection
        field.

    TODO: return alpha, not the ray!!!
    """
    # This implementation is super slow when jitted (and when not). Every time
    # the number of lenses being mapped over changes, the function is recompiled...
    # def mp_a(x, y, *args, **kwargs):
    #     def fn(carry, ak):
    #         ray_x, ray_y = carry
    #         ax, ay = alpha(ray_x, ray_y, *ak[0], **ak[1])
    #         return jnp.stack((ray_x - ax, ray_y - ay)), None

    #     return jax.lax.scan(fn, jnp.array((x, y)), (args, kwargs))[0]

    def mp_a(x, y, *args, **kwargs):
        n_lenses = len(args[0])
        keys = kwargs.keys()
        kwargs_tranpose = [{k: kwargs[k][i] for k in keys} for i in range(n_lenses)]
        args_transpose = [[a[i] for a in args] for i in range(n_lenses)]

        # TODO: return alpha, not ray!
        ray = jnp.array((x, y))
        for a, kw in zip(args_transpose, kwargs_tranpose):
            ray -= alpha(ray[0], ray[1], *a, **kw)

        return ray

    return mp_a


# dominant_lens_alpha(x, y, 2.5, cosmo, alpha_sple, 0.0, 0.0, 1e15, z=1.5)(
#     [-0.3, 0.03], [0.2, 0.4], m=[1e10, 1e10], z=[0.5, 0.9], alpha_nfw
# )([0.3], [0.5], [1e11], [1.8, 1.9], alpha_nfw)


# 1. Compute alpha_ods, alpha_odl
# 2. Compute foreground contribution
# 3. Post-Born for dominant
# 4. Background
# def dominant_lens_alpha(alpha_los, alpha_main):
#     def fn_fg(x, y, *args_fg, **kwargs_fg):
#         def fn_(x, y, *args_ml, **kwargs_ml):
#             def
