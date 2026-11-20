from ...constants import arcsec_to_rad, c_Mpc_s, days_to_seconds


def physical_from_reduced_deflection_angle(ax, ay, d_s, d_ls):
    """
    Computes the physical deflection angle of the given the reduced deflection angles [arcsec].

    Parameters
    ----------
    ax: ArrayLike
        ArrayLike of x axis reduced deflection angles in the lens plane.

        *Unit: arcsec*

    y: ArrayLike
        ArrayLike of y axis reduced deflection angles in the lens plane.

        *Unit: arcsec*

    d_s: float
        distance to the source.

        *Unit: Mpc*

    d_ls: float
        distance from lens to source.

        *Unit: Mpc*

    Returns
    --------
    x_component: ArrayLike
        Physical deflection Angle in the x-direction.

        *Unit: arcsec*

    y_component: ArrayLike
        Physical deflection Angle in the y-direction.

        *Unit: arcsec*

    """

    return (d_s / d_ls) * ax, (d_s / d_ls) * ay


def reduced_from_physical_deflection_angle(ax, ay, d_s, d_ls):
    """
    Computes the reduced deflection angle of the lens at given coordinates [arcsec].

    Parameters
    ----------
    ax: ArrayLike
        ArrayLike of x axis physical deflection angles in the lens plane.

        *Unit: arcsec*

    y: ArrayLike
        ArrayLike of y axis physical deflection angles in the lens plane.

        *Unit: arcsec*

    d_s: float
        distance to the source.

        *Unit: Mpc*

    d_ls: float
        distance from lens to source.

        *Unit: Mpc*

    Returns
    --------
    x_component: ArrayLike
        Reduced deflection Angle in the x-direction.

        *Unit: arcsec*

    y_component: ArrayLike
        Reduced deflection Angle in the y-direction.

        *Unit: arcsec*

    """

    return (d_ls / d_s) * ax, (d_ls / d_s) * ay


def time_delay_arcsec2_to_days(d_l, d_s, d_ls, z_l):
    """
    Computes a scaling factor to use in time delay calculations which converts
    the time delay (i.e. potential and deflection angle squared terms) from
    arcsec^2 to units of days.
    """
    return (1 + z_l) / c_Mpc_s * d_s * d_l / d_ls * arcsec_to_rad**2 / days_to_seconds
