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
