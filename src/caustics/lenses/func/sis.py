from ...backend_obj import backend
from ...utils import translate_rotate


def reduced_deflection_angle_sis(x0, y0, Rein, x, y, s=0.0):
    """
    Compute the reduced deflection angles. See the Meneghetti lecture notes equation 3.46.

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
    R = backend.sqrt(x**2 + y**2) + s
    ax = Rein * x / R
    ay = Rein * y / R
    return ax, ay


def potential_sis(x0, y0, Rein, x, y, s=0.0):
    """
    Compute the lensing potential. See the Meneghetti lecture notes equation 3.45.

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
    potential: ArrayLike
        Lensing potential.

        *Unit: arcsec^2*

    """
    x, y = translate_rotate(x, y, x0, y0)
    R = backend.sqrt(x**2 + y**2) + s
    return Rein * R


def convergence_sis(x0, y0, Rein, x, y, s=0.0):
    """
    Compute the lensing convergence. See the Meneghetti lecture notes equation 3.44.

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
    convergence: ArrayLike
        Lensing convergence.

        *Unit: unitless*

    """
    x, y = translate_rotate(x, y, x0, y0)
    R = backend.sqrt(x**2 + y**2) + s
    return 0.5 * Rein / R
