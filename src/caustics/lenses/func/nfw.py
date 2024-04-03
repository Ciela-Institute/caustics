import torch

from ...utils import translate_rotate
from ...constants import G_over_c2, arcsec_to_rad, rad_to_arcsec


def scale_radius_nfw(critical_density, m, c, DELTA=200.0):
    """
    Compute the scale radius of the NFW profile.

    Parameters
    ----------
    critical_density: Tensor
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    m: Tensor
        The mass of the halo.

        *Unit: Msun*

    c: Tensor
        The concentration parameter of the halo.

        *Unit: unitless*

    DELTA: float
        The overdensity parameter. Amount above the critical surface density at which the scale radius is computed

        *Unit: unitless*

    Returns
    -------
    Tensor
        The scale radius of the NFW profile.

        *Unit: Mpc*

    """
    r_delta = (3 * m / (4 * torch.pi * DELTA * critical_density)) ** (1 / 3)  # fmt: skip
    return 1 / c * r_delta


def scale_density_nfw(critical_density, c, DELTA=200.0):
    """
    Compute the scale density of the NFW profile.

    Parameters
    ----------
    critical_density: Tensor
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    c: Tensor
        The concentration parameter of the halo.

        *Unit: unitless*

    DELTA: float
        The overdensity parameter. Amount above the critical surface density at which the scale radius is computed

        *Unit: unitless*

    Returns
    -------
    Tensor
        The scale density of the NFW profile.

        *Unit: solar mass per square kiloparsec*

    """
    return DELTA / 3 * critical_density * c**3 / ((1 + c).log() - c / (1 + c))  # fmt: skip


def convergence_s_nfw(critical_surface_density, critical_density, m, c, DELTA):
    """
    Compute the dimensionaless surface mass density of the lens.

    critical_surface_density: Tensor
        The critical surface density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^2*

    critical_density: Tensor
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    m: Tensor
        The mass of the halo.

        *Unit: Msun*

    c: Tensor
        The concentration parameter of the halo.

        *Unit: unitless*

    DELTA: float
        The overdensity parameter. Amount above the critical surface density at which the scale radius is computed

        *Unit: unitless*

    Returns
    -------
    Tensor
        The dimensionless surface mass density of the lens.

        *Unit: unitless*
    """
    Rs = scale_radius_nfw(critical_density, m, c, DELTA)
    Ds = scale_density_nfw(critical_density, c, DELTA)
    return Rs * Ds / critical_surface_density


def _f_differentiable_nfw(x):
    """
    Helper method for computing convergence. This can be found in the Meneghetti
    Lecture notes equation 3.69.

    Parameters
    ----------
    x: Tensor
        The input to the function.

    Returns
    -------
    Tensor
        The value of the function f(x).

    """
    # TODO: generalize beyond torch, or patch Tensor
    f = torch.zeros_like(x)
    f[x > 1] = 1 - 2 / (x[x > 1] ** 2 - 1).sqrt() * ((x[x > 1] - 1) / (x[x > 1] + 1)).sqrt().arctan()  # fmt: skip
    f[x < 1] = 1 - 2 / (1 - x[x < 1] ** 2).sqrt() * ((1 - x[x < 1]) / (1 + x[x < 1])).sqrt().arctanh()  # fmt: skip
    return f


def _f_batchable_nfw(x):
    """
    Helper method for computing convergence. This can be found in the Meneghetti
    Lecture notes equation 3.69.

    Parameters
    ----------
    x: Tensor
        The input to the function.

    Returns
    -------
    Tensor
        The value of the function f(x).

    """
    # TODO: generalize beyond torch, or patch Tensor
    # fmt: off
    return torch.where(
        x > 1,
        1 - 2 / (x**2 - 1).sqrt() * ((x - 1) / (x + 1)).sqrt().arctan(),
        torch.where(
            x < 1,
            1 - 2 / (1 - x**2).sqrt() * ((1 - x) / (1 + x)).sqrt().arctanh(),
            torch.zeros_like(x),  # where: x == 1
        ),
    )
    # fmt: on


def _g_differentiable_nfw(x):
    """
    Helper method for computing potential. This can be found in the Meneghetti
    Lecture notes equation 3.71.

    Parameters
    ----------
    x: Tensor
        The input to the function.

    Returns
    -------
    Tensor
        The value of the function g(x).

    """
    # TODO: generalize beyond torch, or patch Tensor
    term_1 = (x / 2).log() ** 2
    term_2 = torch.zeros_like(x)
    term_2[x > 1] = (1 / x[x > 1]).arccos() ** 2  # fmt: skip
    term_2[x < 1] = -(1 / x[x < 1]).arccosh() ** 2  # fmt: skip
    return term_1 + term_2


def _g_batchable_nfw(x):
    """
    Helper method for computing potential. This can be found in the Meneghetti
    Lecture notes equation 3.71.

    Parameters
    ----------
    x: Tensor
        The input to the function.

    Returns
    -------
    Tensor
        The value of the function g(x).

    """
    # TODO: generalize beyond torch, or patch Tensor
    term_1 = (x / 2).log() ** 2
    term_2 = torch.where(
        x > 1,
        (1 / x).arccos() ** 2,
        torch.where(
            x < 1,
            -(1 / x).arccosh() ** 2,
            torch.zeros_like(x),  # where: x == 1
        ),
    )
    return term_1 + term_2


def _h_differentiable_nfw(x):
    """
    Helper method for computing deflection angles. This can be found in the Meneghetti
    Lecture notes equation 3.73.

    Parameters
    ----------
    x: Tensor
        The input to the function.

    Returns
    -------
    Tensor
        The value of the function h(x).

    """
    term_1 = (x / 2).log()
    term_2 = torch.ones_like(x)
    term_2[x > 1] = (1 / x[x > 1]).arccos() * 1 / (x[x > 1] ** 2 - 1).sqrt()  # fmt: skip
    term_2[x < 1] = (1 / x[x < 1]).arccosh() * 1 / (1 - x[x < 1] ** 2).sqrt()  # fmt: skip
    return term_1 + term_2


def _h_batchable_nfw(x):
    """
    Helper method for computing deflection angles. This can be found in the Meneghetti
    Lecture notes equation 3.73.

    Parameters
    ----------
    x: Tensor
        The input to the function.

    Returns
    -------
    Tensor
        The value of the function h(x).

    """
    term_1 = (x / 2).log()
    term_2 = torch.where(
        x > 1,
        (1 / x).arccos() * 1 / (x**2 - 1).sqrt(),  # fmt: skip
        torch.where(
            x < 1, (1 / x).arccosh() * 1 / (1 - x**2).sqrt(), torch.ones_like(x)  # fmt: skip
        ),
    )
    return term_1 + term_2


def physical_deflection_angle_nfw(
    x0,
    y0,
    m,
    c,
    critical_density,
    d_l,
    x,
    y,
    _h=_h_differentiable_nfw,
    DELTA=200.0,
    s=0.0,
):
    """
    Compute the physical deflection angles. This is an expanded form of the
    Meneghetti notes equation 3.72

    Parameters
    ----------
    x0: Tensor
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Tensor
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    m: Tensor
        Mass of the lens. Default is None.

        *Unit: Msun*

    c: Tensor
        Concentration parameter of the lens. Default is None.

        *Unit: unitless*

    x: Tensor
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: Tensor
        y-coordinates in the lens plane.

        *Unit: arcsec*

    s: float
        Softening parameter to avoid singularities at the center of the lens.
        Default is 0.0.

        *Unit: arcsec*
    """
    x, y = translate_rotate(x, y, x0, y0)
    th = (x**2 + y**2).sqrt() + s
    scale_radius = scale_radius_nfw(critical_density, m, c, DELTA)
    xi = d_l * th * arcsec_to_rad
    r = xi / scale_radius

    deflection_angle = 16 * torch.pi * G_over_c2 * scale_density_nfw(critical_density, c, DELTA) * scale_radius**3 * _h(r) * rad_to_arcsec / xi  # fmt: skip

    ax = deflection_angle * x / th
    ay = deflection_angle * y / th
    return ax, ay  # fmt: skip


def convergence_nfw(
    critical_surface_density,
    critical_density,
    x0,
    y0,
    m,
    c,
    x,
    y,
    d_l,
    _f=_f_differentiable_nfw,
    DELTA=200.0,
    s=0.0,
):
    """
    Compute the convergence. This can be found in the Meneghetti
    Lecture notes equation 3.74.

    Parameters
    ----------
    critical_surface_density: Tensor
        The critical surface density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^2*

    critical_density: Tensor
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    m: Tensor
        The mass of the halo

    c: Optional[Tensor]
        Concentration parameter of the lens. Default is None.

        *Unit: unitless*

    x: Tensor
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: Tensor
        y-coordinates in the lens plane.

        *Unit: arcsec*

    s: float
        Softening parameter to avoid singularities at the center of the lens. Default is 0.0.

        *Unit: arcsec*
    """
    x, y = translate_rotate(x, y, x0, y0)
    th = (x**2 + y**2).sqrt() + s
    scale_radius = scale_radius_nfw(critical_density, m, c, DELTA)
    xi = d_l * th * arcsec_to_rad
    r = xi / scale_radius  # xi / xi_0
    convergence_s = convergence_s_nfw(
        critical_surface_density, critical_density, m, c, DELTA
    )
    return 2 * convergence_s * _f(r) / (r**2 - 1)  # fmt: skip


def potential_nfw(
    critical_surface_density,
    critical_density,
    x0,
    y0,
    m,
    c,
    d_l,
    x,
    y,
    _g=_g_differentiable_nfw,
    DELTA=200.0,
    s=0.0,
):
    """
    Compute the convergence. This can be found in the Meneghetti
    Lecture notes equation 3.70.

    Parameters
    ----------
    critical_surface_density: Tensor
        The critical surface density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^2*

    critical_density: Tensor
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    m: Tensor
        The mass of the halo

    c: Optional[Tensor]
        Concentration parameter of the lens. Default is None.

        *Unit: unitless*

    x: Tensor
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: Tensor
        y-coordinates in the lens plane.

        *Unit: arcsec*

    s: float
        Softening parameter to avoid singularities at the center of the lens. Default is 0.0.

        *Unit: arcsec*
    """
    x, y = translate_rotate(x, y, x0, y0)
    th = (x**2 + y**2).sqrt() + s
    scale_radius = scale_radius_nfw(critical_density, m, c, DELTA)
    xi = d_l * th * arcsec_to_rad
    r = xi / scale_radius  # xi / xi_0
    convergence_s = convergence_s_nfw(
        critical_surface_density, critical_density, m, c, DELTA
    )
    return 2 * convergence_s * _g(r) * scale_radius**2 / (d_l**2 * arcsec_to_rad**2)  # fmt: skip
