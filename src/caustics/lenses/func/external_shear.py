import torch

from ...utils import translate_rotate


def reduced_deflection_angle_external_shear(x0, y0, gamma_1, gamma_2, x, y):
    """
    Compute the reduced deflection angles for an external shear field. Here we
    use the Meneghetti lecture notes and take derivatives of equation 3.80

    Parameters
    ----------
    x0: Tensor
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Tensor
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    gamma_1: Tensor
        The shear component in the x-direction.

        *Unit: unitless*

    gamma_2: Tensor
        The shear component in the y-direction.

        *Unit: unitless*

    x: Tensor
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: Tensor
        y-coordinates in the lens plane.

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
    # Derivatives of Meneghetti eq 3.80
    x, y = translate_rotate(x, y, x0, y0)
    ax = gamma_1 * x + gamma_2 * y
    ay = gamma_2 * x - gamma_1 * y
    return ax, ay


def potential_external_shear(x0, y0, gamma_1, gamma_2, x, y):
    """
    Compute the lensing potential for an external shear field. Here we use the
    Meneghetti lecture notes equation 3.80

    Parameters
    ----------
    x0: Tensor
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Tensor
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    gamma_1: Tensor
        The shear component in the x-direction.

        *Unit: unitless*

    gamma_2: Tensor
        The shear component in the y-direction.

        *Unit: unitless*

    x: Tensor
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: Tensor
        y-coordinates in the lens plane.

        *Unit: arcsec*

    Returns
    -------
    Tensor
        The lensing potential.

        *Unit: arcsec^2*

    """
    x, y = translate_rotate(x, y, x0, y0)
    return 0.5 * gamma_1 * (x**2 - y**2) + gamma_2 * x * y


def gamma_phi_to_gamma1(gamma, phi):
    """
    Convert the shear magnitude and angle to the gamma_1 component.

    Parameters
    ----------
    gamma: Tensor
        The shear magnitude.

        *Unit: unitless*

    phi: Tensor
        The shear angle.

        *Unit: radians*

    Returns
    -------
    Tensor
        The gamma_1 component of the shear.

        *Unit: unitless*

    """
    return gamma * torch.cos(2 * phi)


def gamma_phi_to_gamma2(gamma, phi):
    """
    Convert the shear magnitude and angle to the gamma_2 component.

    Parameters
    ----------
    gamma: Tensor
        The shear magnitude.

        *Unit: unitless*

    phi: Tensor
        The shear angle.

        *Unit: radians*

    Returns
    -------
    Tensor
        The gamma_2 component of the shear.

        *Unit: unitless*

    """
    return gamma * torch.sin(2 * phi)
