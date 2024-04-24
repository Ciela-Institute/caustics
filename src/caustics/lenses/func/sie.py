from ...utils import translate_rotate, derotate


def reduced_deflection_angle_sie(x0, y0, q, phi, b, x, y, s=0.0):
    """
    Calculate the physical deflection angle. For more detail see Keeton 2002
    equations 34 and 35, although our ``b`` is defined as :math:`\\sqrt(q)b` in
    Keeton's notation.

    Parameters
    ----------
    x0: Tensor
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Tensor
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: Tensor
        The axis ratio of the lens.

        *Unit: unitless*

    phi: Tensor
        The orientation angle of the lens (position angle).

        *Unit: radians*

    b: Tensor
        The Einstein radius of the lens.

        *Unit: arcsec*

    x: Tensor
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: Tensor
        The y-coordinate of the lens.

        *Unit: arcsec*

    s: float
        The core radius of the lens (defaults to 0.0).

        *Unit: arcsec*

    Returns
    --------
    x_component: Tensor
        The x-component of the deflection angle.

        *Unit: arcsec*

    y_component: Tensor
        The y-component of the deflection angle.

        *Unit: arcsec*

    """
    x, y = translate_rotate(x, y, x0, y0, phi)
    psi = (q**2 * (x**2 + s**2) + y**2).sqrt()
    f = (1 - q**2).sqrt()
    ax = b * q.sqrt() / f * (f * x / (psi + s)).atan()  # fmt: skip
    ay = b * q.sqrt() / f * (f * y / (psi + q**2 * s)).atanh()  # fmt: skip

    return derotate(ax, ay, phi)


def potential_sie(x0, y0, q, phi, b, x, y, s=0.0):
    """
    Compute the lensing potential. For more detail see Keeton 2002
    equation 33, although our ``b`` is defined as :math:`\\sqrt(q)b` in
    Keeton's notation. Also we use the :math:`s \\approx 0` limit here.

    Parameters
    ----------
    x0: Tensor
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Tensor
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: Tensor
        The axis ratio of the lens.

        *Unit: unitless*

    phi: Tensor
        The orientation angle of the lens (position angle).

        *Unit: radians*

    b: Tensor
        The Einstein radius of the lens.

        *Unit: arcsec*

    x: Tensor
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: Tensor
        The y-coordinate of the lens.

        *Unit: arcsec*

    s: float
        The core radius of the lens (defaults to 0.0).

        *Unit: arcsec*

    Returns
    -------
    Tensor
        The lensing potential.

        *Unit: arcsec^2*

    """
    ax, ay = reduced_deflection_angle_sie(x0, y0, q, phi, b, x, y, s)
    ax, ay = derotate(ax, ay, -phi)
    x, y = translate_rotate(x, y, x0, y0, phi)
    return x * ax + y * ay


def convergence_sie(x0, y0, q, phi, b, x, y, s=0.0):
    """
    Calculate the projected mass density. This is converted from the SIS
    convergence definition.

    Parameters
    ----------
    x0: Tensor
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Tensor
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: Tensor
        The axis ratio of the lens.

        *Unit: unitless*

    phi: Tensor
        The orientation angle of the lens (position angle).

        *Unit: radians*

    b: Tensor
        The Einstein radius of the lens.

        *Unit: arcsec*

    x: Tensor
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: Tensor
        The y-coordinate of the lens.

        *Unit: arcsec*

    s: float
        The core radius of the lens (defaults to 0.0).

        *Unit: arcsec*

    Returns
    -------
    Tensor
        The projected mass density.

        *Unit: unitless*

    """
    x, y = translate_rotate(x, y, x0, y0, phi)
    psi = (q**2 * (x**2 + s**2) + y**2).sqrt()
    return 0.5 * q.sqrt() * b / psi
