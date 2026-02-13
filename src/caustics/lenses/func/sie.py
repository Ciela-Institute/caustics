from ...backend_obj import backend
from ...utils import translate_rotate, derotate
from ...constants import c_km_s, rad_to_arcsec


def reduced_deflection_angle_sie(x0, y0, q, phi, Rein, x, y, s=0.0):
    """
    Calculate the physical deflection angle. For more detail see Keeton 2002
    equations 34 and 35, although our ``Rein`` is defined as :math:`b/\\sqrt(q)` in
    Keeton's notation.

    Parameters
    ----------
    x0: ArrayLike
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: ArrayLike
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: ArrayLike
        The axis ratio of the lens.

        *Unit: unitless*

    phi: ArrayLike
        The orientation angle of the lens (position angle).

        *Unit: radians*

    Rein: ArrayLike
        The Einstein radius of the lens.

        *Unit: arcsec*

    x: ArrayLike
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: ArrayLike
        The y-coordinate of the lens.

        *Unit: arcsec*

    s: float
        The core radius of the lens (defaults to 0.0).

        *Unit: arcsec*

    Returns
    --------
    x_component: ArrayLike
        The x-component of the deflection angle.

        *Unit: arcsec*

    y_component: ArrayLike
        The y-component of the deflection angle.

        *Unit: arcsec*

    """
    # Handle the case where q = 1.0, numerical instability
    q = backend.where(q == 1.0, q - 1e-6, q)

    x, y = translate_rotate(x, y, x0, y0, phi)

    # intermediary variables
    q2_ = q**2
    f = backend.sqrt(1 - q2_)
    rein_q_sqrt_f_ = Rein * backend.sqrt(q) / f

    psi = backend.sqrt(q2_ * (x**2 + s**2) + y**2)
    ax = rein_q_sqrt_f_ * backend.atan(f * x / (psi + s))  # fmt: skip
    ay = rein_q_sqrt_f_ * backend.atanh(f * y / (psi + q2_ * s))  # fmt: skip

    return derotate(ax, ay, phi)


def potential_sie(x0, y0, q, phi, Rein, x, y, s=0.0):
    """
    Compute the lensing potential. For more detail see Keeton 2002
    equation 33, although our ``Rein`` is defined as :math:`b/\\sqrt(q)` in
    Keeton's notation.

    Parameters
    ----------
    x0: ArrayLike
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: ArrayLike
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: ArrayLike
        The axis ratio of the lens.

        *Unit: unitless*

    phi: ArrayLike
        The orientation angle of the lens (position angle).

        *Unit: radians*

    Rein: ArrayLike
        The Einstein radius of the lens.

        *Unit: arcsec*

    x: ArrayLike
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: ArrayLike
        The y-coordinate of the lens.

        *Unit: arcsec*

    s: float
        The core radius of the lens (defaults to 0.0).

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        The lensing potential.

        *Unit: arcsec^2*

    """
    ax, ay = reduced_deflection_angle_sie(x0, y0, q, phi, Rein, x, y, s)
    ax, ay = derotate(ax, ay, -phi)
    x, y = translate_rotate(x, y, x0, y0, phi)

    # intermediary variables
    q2_ = q**2
    x2_ = x**2
    max_s_ = max(s, 1e-6)
    rein_q_sqrt_s_ = Rein * backend.sqrt(q) * s

    psi = backend.sqrt(q2_ * (x2_ + s**2) + y**2)
    return (
        x * ax
        + y * ay
        - rein_q_sqrt_s_
        * backend.log(backend.sqrt((psi + max_s_) ** 2 + (1 - q2_) * x2_))
        + rein_q_sqrt_s_ * backend.log((1 + q) * max_s_)
    )


def convergence_sie(x0, y0, q, phi, Rein, x, y, s=0.0):
    """
    Calculate the projected mass density. This is converted from the SIS
    convergence definition.

    Parameters
    ----------
    x0: ArrayLike
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: ArrayLike
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: ArrayLike
        The axis ratio of the lens.

        *Unit: unitless*

    phi: ArrayLike
        The orientation angle of the lens (position angle).

        *Unit: radians*

    Rein: ArrayLike
        The Einstein radius of the lens.

        *Unit: arcsec*

    x: ArrayLike
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: ArrayLike
        The y-coordinate of the lens.

        *Unit: arcsec*

    s: float
        The core radius of the lens (defaults to 0.0).

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        The projected mass density.

        *Unit: unitless*

    """
    x, y = translate_rotate(x, y, x0, y0, phi)
    psi = backend.sqrt(q**2 * (x**2 + s**2) + y**2)
    return 0.5 * backend.sqrt(q) * Rein / psi


def sigma_v_to_rein_sie(sigma_v, dls, ds):
    """
    Convert the velocity dispersion to the Einstein radius. See equation 16.22
    in Dynamics and Astrophysics of Galaxies by Jo Bovy

    Parameters
    ----------
    sigma_v: ArrayLike
        The velocity dispersion of the lens.

        *Unit: km/s*

    dls: ArrayLike
        The angular diameter distance between the lens and the source.

        *Unit: Mpc*

    ds: ArrayLike
        The angular diameter distance between the observer and the source.

        *Unit: Mpc*

    Returns
    -------
    ArrayLike
        The Einstein radius.

        *Unit: arcsec*

    """
    return rad_to_arcsec * 4 * backend.pi * (sigma_v / c_km_s) ** 2 * dls / ds
