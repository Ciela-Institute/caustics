from ...backend_obj import backend
from ...utils import translate_rotate
from ...constants import G_over_c2, arcsec_to_rad, rad_to_arcsec


def scale_radius_nfw(critical_density, mass, c, DELTA=200.0):
    """
    Compute the scale radius of the NFW profile.

    Parameters
    ----------
    critical_density: ArrayLike
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    mass: ArrayLike
        The mass of the halo.

        *Unit: Msun*

    c: ArrayLike
        The concentration parameter of the halo.

        *Unit: unitless*

    DELTA: float
        The overdensity parameter. Amount above the critical surface density at which the scale radius is computed

        *Unit: unitless*

    Returns
    -------
    ArrayLike
        The scale radius of the NFW profile.

        *Unit: Mpc*

    """
    r_delta = (3 * mass / (4 * backend.pi * DELTA * critical_density)) ** (1 / 3)  # fmt: skip
    return r_delta / c


def scale_density_nfw(critical_density, c, DELTA=200.0):
    """
    Compute the scale density of the NFW profile.

    Parameters
    ----------
    critical_density: ArrayLike
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    c: ArrayLike
        The concentration parameter of the halo.

        *Unit: unitless*

    DELTA: float
        The overdensity parameter. Amount above the critical surface density at which the scale radius is computed

        *Unit: unitless*

    Returns
    -------
    ArrayLike
        The scale density of the NFW profile.

        *Unit: solar mass per square kiloparsec*

    """
    c_ = 1 + c
    return DELTA / 3 * critical_density * c**3 / (backend.log(c_) - c / c_)  # fmt: skip


def convergence_s_nfw(critical_surface_density, critical_density, mass, c, DELTA):
    """
    Compute the dimensionaless surface mass density of the lens.

    critical_surface_density: ArrayLike
        The critical surface density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^2*

    critical_density: ArrayLike
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    mass: ArrayLike
        The mass of the halo.

        *Unit: Msun*

    c: ArrayLike
        The concentration parameter of the halo.

        *Unit: unitless*

    DELTA: float
        The overdensity parameter. Amount above the critical surface density at which the scale radius is computed

        *Unit: unitless*

    Returns
    -------
    ArrayLike
        The dimensionless surface mass density of the lens.

        *Unit: unitless*
    """
    Rs = scale_radius_nfw(critical_density, mass, c, DELTA)
    Ds = scale_density_nfw(critical_density, c, DELTA)
    return Rs * Ds / critical_surface_density


def _f_nfw(x):
    """
    Helper method for computing convergence. This can be found in the Meneghetti
    Lecture notes equation 3.69.

    Parameters
    ----------
    x: ArrayLike
        The input to the function.

    Returns
    -------
    ArrayLike
        The value of the function f(x).

    """
    # fmt: off
    x_gt1 = backend.clamp(x, min=1 + 1e-6)
    x_lt1 = backend.clamp(x, max=1 - 1e-6)

    f_pos = 1 - 2 * backend.atan(backend.sqrt((x_gt1 - 1) / (x_gt1 + 1))) / backend.sqrt(x_gt1 ** 2 - 1)
    f_neg = 1 - 2 * backend.atanh(backend.sqrt((1 - x_lt1) / (1 + x_lt1))) / backend.sqrt(1 - x_lt1 ** 2)
    return backend.where(x > 1, f_pos, backend.where(x < 1, f_neg, backend.zeros_like(x)))
    # fmt: on


def _g_nfw(x):
    """
    Helper method for computing potential. This can be found in the Meneghetti
    Lecture notes equation 3.71.

    Parameters
    ----------
    x: ArrayLike
        The input to the function.

    Returns
    -------
    ArrayLike
        The value of the function g(x).

    """
    term_1 = backend.log(x / 2) ** 2
    x_gt1 = backend.clamp(x, min=1 + 1e-6)
    x_lt1 = backend.clamp(x, max=1 - 1e-6)

    g_pos = backend.arccos(1 / x_gt1) ** 2
    g_neg = backend.arccosh(-(1 / x_lt1)) ** 2
    term_2 = backend.where(
        x > 1, g_pos, backend.where(x < 1, g_neg, backend.zeros_like(x))
    )
    return term_1 + term_2


def _h_nfw(x):
    """
    Helper method for computing deflection angles. This can be found in the Meneghetti
    Lecture notes equation 3.73.

    Parameters
    ----------
    x: ArrayLike
        The input to the function.

    Returns
    -------
    ArrayLike
        The value of the function h(x).

    """
    term_1 = backend.log(x / 2)
    x_gt1 = backend.clamp(x, min=1 + 1e-6)
    x_lt1 = backend.clamp(x, max=1 - 1e-6)

    h_pos = backend.arccos(1 / x_gt1) / backend.sqrt(x_gt1**2 - 1)
    h_neg = backend.arccosh(1 / x_lt1) / backend.sqrt(1 - x_lt1**2)
    term_2 = backend.where(
        x > 1, h_pos, backend.where(x < 1, h_neg, backend.ones_like(x))
    )
    return term_1 + term_2


def physical_deflection_angle_nfw(
    x0,
    y0,
    mass,
    c,
    critical_density,
    d_l,
    x,
    y,
    DELTA=200.0,
    s=0.0,
):
    """
    Compute the physical deflection angles. This is an expanded form of the
    Meneghetti notes equation 3.72

    Parameters
    ----------
    x0: ArrayLike
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: ArrayLike
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    mass: ArrayLike
        Mass of the lens. Default is None.

        *Unit: Msun*

    c: ArrayLike
        Concentration parameter of the lens. Default is None.

        *Unit: unitless*

    x: ArrayLike
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: ArrayLike
        y-coordinates in the lens plane.

        *Unit: arcsec*

    s: float
        Softening parameter to avoid singularities at the center of the lens.
        Default is 0.0.

        *Unit: arcsec*
    """
    x, y = translate_rotate(x, y, x0, y0)
    th = backend.sqrt(x**2 + y**2) + s
    scale_radius = scale_radius_nfw(critical_density, mass, c, DELTA)
    xi = d_l * th * arcsec_to_rad
    r = xi / scale_radius

    deflection_angle = 16 * backend.pi * G_over_c2 * scale_density_nfw(critical_density, c, DELTA) * scale_radius**3 * _h_nfw(r) * rad_to_arcsec / xi  # fmt: skip

    ax = deflection_angle * x / th
    ay = deflection_angle * y / th
    return ax, ay  # fmt: skip


def convergence_nfw(
    critical_surface_density,
    critical_density,
    x0,
    y0,
    mass,
    c,
    x,
    y,
    d_l,
    DELTA=200.0,
    s=0.0,
):
    """
    Compute the convergence. This can be found in the Meneghetti
    Lecture notes equation 3.74.

    Parameters
    ----------
    critical_surface_density: ArrayLike
        The critical surface density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^2*

    critical_density: ArrayLike
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    mass: ArrayLike
        The mass of the halo

    c: Optional[ArrayLike]
        Concentration parameter of the lens. Default is None.

        *Unit: unitless*

    x: ArrayLike
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: ArrayLike
        y-coordinates in the lens plane.

        *Unit: arcsec*

    s: float
        Softening parameter to avoid singularities at the center of the lens. Default is 0.0.

        *Unit: arcsec*
    """
    x, y = translate_rotate(x, y, x0, y0)
    th = backend.sqrt(x**2 + y**2) + s
    scale_radius = scale_radius_nfw(critical_density, mass, c, DELTA)
    xi = d_l * th * arcsec_to_rad
    r = xi / scale_radius  # xi / xi_0
    convergence_s = convergence_s_nfw(
        critical_surface_density, critical_density, mass, c, DELTA
    )
    return 2 * convergence_s * _f_nfw(r) / (r**2 - 1)  # fmt: skip


def potential_nfw(
    critical_surface_density,
    critical_density,
    x0,
    y0,
    mass,
    c,
    d_l,
    x,
    y,
    DELTA=200.0,
    s=0.0,
):
    """
    Compute the convergence. This can be found in the Meneghetti
    Lecture notes equation 3.70.

    Parameters
    ----------
    critical_surface_density: ArrayLike
        The critical surface density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^2*

    critical_density: ArrayLike
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    mass: ArrayLike
        The mass of the halo

    c: Optional[ArrayLike]
        Concentration parameter of the lens. Default is None.

        *Unit: unitless*

    x: ArrayLike
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: ArrayLike
        y-coordinates in the lens plane.

        *Unit: arcsec*

    s: float
        Softening parameter to avoid singularities at the center of the lens. Default is 0.0.

        *Unit: arcsec*
    """
    x, y = translate_rotate(x, y, x0, y0)
    th = backend.sqrt(x**2 + y**2) + s
    scale_radius = scale_radius_nfw(critical_density, mass, c, DELTA)
    xi = d_l * th * arcsec_to_rad
    r = xi / scale_radius  # xi / xi_0
    convergence_s = convergence_s_nfw(
        critical_surface_density, critical_density, mass, c, DELTA
    )
    return 2 * convergence_s * _g_nfw(r) * scale_radius**2 / (d_l**2 * arcsec_to_rad**2)  # fmt: skip
