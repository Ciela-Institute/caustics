import torch

from ...utils import translate_rotate


def reduced_deflection_angle_point(x0, y0, th_ein, x, y, s=0.0):
    """
    Compute the reduced deflection angles. See the Meneghetti lecture notes
    equation 3.1 for more detail.

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
    th = (x**2 + y**2).sqrt() + s
    ax = x * th_ein**2 / th**2
    ay = y * th_ein**2 / th**2
    return ax, ay


def potential_point(x0, y0, th_ein, x, y, s=0.0):
    """
    Compute the lensing potential. See the Meneghetti lecture notes
    equation 3.3 for more detail.

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
    Tensor
        The lensing potential.

        *Unit: arcsec^2*

    """
    x, y = translate_rotate(x, y, x0, y0)
    th = (x**2 + y**2).sqrt() + s
    return th_ein**2 * th.log()


def convergence_point(x0, y0, x, y):
    """
    Compute the convergence (dimensionless surface mass density). This follows
    essenitally by definition.

    Parameters
    ----------
    x0: Tensor
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Tensor
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    x: Tensor
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: Tensor
        y-coordinates in the lens plane.

        *Unit: arcsec*

    Returns
    --------
    Tensor
        The convergence (dimensionless surface mass density).

        *Unit: unitless*

    """
    x, y = translate_rotate(x, y, x0, y0)
    return torch.where((x == 0) & (y == 0), torch.inf, 0.0)
