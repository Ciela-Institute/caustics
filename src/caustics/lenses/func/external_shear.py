from ...utils import translate_rotate


def reduced_deflection_angle_external_shear(x0, y0, gamma_1, gamma_2, x, y):
    """
    Compute the reduced deflection angles.

    Parameters
    ----------
    x0: Optional[Tensor]
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Optional[Tensor]
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    gamma_1: Optional[Tensor]
        The shear component in the x-direction.

        *Unit: unitless*

    gamma_2: Optional[Tensor]
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
    # Meneghetti eq 3.83
    # TODO, why is it not:
    # th = (x**2 + y**2).sqrt() + self.s
    # a1 = x/th + x * gamma_1 + y * gamma_2
    # a2 = y/th + x * gamma_2 - y * gamma_1
    x, y = translate_rotate(x, y, x0, y0)
    ax = gamma_1 * x + gamma_2 * y
    ay = gamma_2 * x - gamma_1 * y
    return ax, ay


def potential_external_shear(x0, y0, gamma_1, gamma_2, x, y):
    """
    Compute the lensing potential.

    Parameters
    ----------
    x0: Optional[Tensor]
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Optional[Tensor]
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    gamma_1: Optional[Tensor]
        The shear component in the x-direction.

        *Unit: unitless*

    gamma_2: Optional[Tensor]
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
    ax, ay = reduced_deflection_angle_external_shear(x0, y0, gamma_1, gamma_2, x, y)
    x, y = translate_rotate(x, y, x0, y0)
    return 0.5 * (x * ax + y * ay)
