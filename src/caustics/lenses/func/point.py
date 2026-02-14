from ...backend_obj import backend
from ...utils import translate_rotate
from ...constants import G_over_c2, rad_to_arcsec


def reduced_deflection_angle_point(x0, y0, Rein, x, y, s=0.0):
    """
    Compute the reduced deflection angles. See the Meneghetti lecture notes
    equation 3.1 for more detail.

    Parameters
    ----------
    x0: ArrayLike
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: ArrayLike
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    Rein: ArrayLike
        Einstein radius of the lens.

        *Unit: arcsec*

    x: ArrayLike
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: ArrayLike
        y-coordinates in the lens plane.

        *Unit: arcsec*

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*

    Returns
    -------
    x_component: ArrayLike
        Deflection Angle in the x-direction.

        *Unit: arcsec*

    y_component: ArrayLike
        Deflection Angle in the y-direction.

        *Unit: arcsec*

    """
    x, y = translate_rotate(x, y, x0, y0)
    th = backend.sqrt(x**2 + y**2) + s
    ax = x * Rein**2 / th**2
    ay = y * Rein**2 / th**2
    return ax, ay


def potential_point(x0, y0, Rein, x, y, s=0.0):
    """
    Compute the lensing potential. See the Meneghetti lecture notes
    equation 3.3 for more detail.

    Parameters
    ----------
    x0: ArrayLike
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: ArrayLike
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    Rein: ArrayLike
        Einstein radius of the lens.

        *Unit: arcsec*

    x: ArrayLike
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: ArrayLike
        y-coordinates in the lens plane.

        *Unit: arcsec*

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        The lensing potential.

        *Unit: arcsec^2*

    """
    x, y = translate_rotate(x, y, x0, y0)
    th = backend.sqrt(x**2 + y**2) + s
    return Rein**2 * backend.log(th)


def convergence_point(x0, y0, x, y):
    """
    Compute the convergence (dimensionless surface mass density). This follows
    essenitally by definition.

    Parameters
    ----------
    x0: ArrayLike
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: ArrayLike
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    x: ArrayLike
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: ArrayLike
        y-coordinates in the lens plane.

        *Unit: arcsec*

    Returns
    --------
    ArrayLike
        The convergence (dimensionless surface mass density).

        *Unit: unitless*

    """
    x, y = translate_rotate(x, y, x0, y0)
    return backend.where((x == 0) & (y == 0), backend.inf, 0.0)


def mass_to_rein_point(M, d_ls, d_l, d_s):
    """
    Compute the Einstein radius of a point mass. See Meneghetti lecture notes equation 1.39

    Parameters
    ----------
    M: ArrayLike
        Mass of the lens.

        *Unit: solar masses*

    d_ls: ArrayLike
        Distance between the lens and the source.

        *Unit: Mpc*

    d_l: ArrayLike
        Distance between the observer and the lens.

        *Unit: Mpc*

    d_s: ArrayLike
        Distance between the observer and the source.

        *Unit: Mpc*

    Returns
    -------
    ArrayLike
        The Einstein radius.

        *Unit: arcsec*

    """
    return rad_to_arcsec * backend.sqrt(4 * G_over_c2 * M * d_ls / (d_l * d_s))


def rein_to_mass_point(r, d_ls, d_l, d_s):
    """
    Compute the Einstein radius of a point mass. See Meneghetti lecture notes equation 1.39

    Parameters
    ----------
    r: ArrayLike
        Einstein radius of the lens.

        *Unit: arcsec*

    d_ls: ArrayLike
        Distance between the lens and the source.

        *Unit: Mpc*

    d_l: ArrayLike
        Distance between the observer and the lens.

        *Unit: Mpc*

    d_s: ArrayLike
        Distance between the observer and the source.

        *Unit: Mpc*

    Returns
    -------
    ArrayLike
        The mass of the lens

        *Unit: solar masses*

    """
    return (r / rad_to_arcsec) ** 2 * d_l * d_s / (4 * G_over_c2 * d_ls)
