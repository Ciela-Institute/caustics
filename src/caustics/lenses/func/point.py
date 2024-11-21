import torch

from ...utils import translate_rotate
from ...constants import G_over_c2, rad_to_arcsec


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


def mass_to_rein_point(M, d_ls, d_l, d_s):
    """
    Compute the Einstein radius of a point mass. See Meneghetti lecture notes equation 1.39

    Parameters
    ----------
    M: Tensor
        Mass of the lens.

        *Unit: solar masses*

    d_ls: Tensor
        Distance between the lens and the source.

        *Unit: Mpc*

    d_l: Tensor
        Distance between the observer and the lens.

        *Unit: Mpc*

    d_s: Tensor
        Distance between the observer and the source.

        *Unit: Mpc*

    Returns
    -------
    Tensor
        The Einstein radius.

        *Unit: arcsec*

    """
    return rad_to_arcsec * (4 * G_over_c2 * M * d_ls / (d_l * d_s)).sqrt()


def rein_to_mass_point(r, d_ls, d_l, d_s):
    """
    Compute the Einstein radius of a point mass. See Meneghetti lecture notes equation 1.39

    Parameters
    ----------
    r: Tensor
        Einstein radius of the lens.

        *Unit: arcsec*

    d_ls: Tensor
        Distance between the lens and the source.

        *Unit: Mpc*

    d_l: Tensor
        Distance between the observer and the lens.

        *Unit: Mpc*

    d_s: Tensor
        Distance between the observer and the source.

        *Unit: Mpc*

    Returns
    -------
    Tensor
        The mass of the lens

        *Unit: solar masses*

    """
    return (r / rad_to_arcsec) ** 2 * d_l * d_s / (4 * G_over_c2 * d_ls)
