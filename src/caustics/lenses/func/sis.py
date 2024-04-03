from ...utils import translate_rotate


def reduced_deflection_angle_sis(x0, y0, th_ein, x, y, s=0.0):
    """
    Compute the reduced deflection angles. See the Meneghetti lecture notes equation 3.46.

    Parameters
    ----------
    x0: Tensor
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Tensor
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    th_ein: Tensor
        Einstein radius of the lens.

        *Unit: arcsec*

    x: Tensor
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: Tensor
        y-coordinates in the lens plane.

        *Unit: arcsec*

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*

    Returns
    -------
    x_component: Tensor
        Deflection Angle in the x-direction.

        *Unit: arcsec*

    y_component: Tensor
        Deflection Angle in the y-direction.

        *Unit: arcsec*

    """
    x, y = translate_rotate(x, y, x0, y0)
    R = (x**2 + y**2).sqrt() + s
    ax = th_ein * x / R
    ay = th_ein * y / R
    return ax, ay


def potential_sis(x0, y0, th_ein, x, y, s=0.0):
    """
    Compute the lensing potential. See the Meneghetti lecture notes equation 3.45.

    Parameters
    ----------
    x0: Tensor
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Tensor
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    th_ein: Tensor
        Einstein radius of the lens.

        *Unit: arcsec*

    x: Tensor
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: Tensor
        y-coordinates in the lens plane.

        *Unit: arcsec*

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*

    Returns
    -------
    potential: Tensor
        Lensing potential.

        *Unit: arcsec^2*

    """
    x, y = translate_rotate(x, y, x0, y0)
    R = (x**2 + y**2).sqrt() + s
    return th_ein * R


def convergence_sis(x0, y0, th_ein, x, y, s=0.0):
    """
    Compute the lensing convergence. See the Meneghetti lecture notes equation 3.44.

    Parameters
    ----------
    x0: Tensor
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Tensor
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    th_ein: Tensor
        Einstein radius of the lens.

        *Unit: arcsec*

    x: Tensor
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: Tensor
        y-coordinates in the lens plane.

        *Unit: arcsec*

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*

    Returns
    -------
    convergence: Tensor
        Lensing convergence.

        *Unit: unitless*

    """
    x, y = translate_rotate(x, y, x0, y0)
    R = (x**2 + y**2).sqrt() + s
    return 0.5 * th_ein / R
