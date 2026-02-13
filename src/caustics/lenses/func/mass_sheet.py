from ...backend_obj import backend
from ...utils import translate_rotate


def reduced_deflection_angle_mass_sheet(x0, y0, kappa, x, y):
    """
    Compute the reduced deflection angles. Here we use the Meneghetti lecture
    notes equation 3.84.

    Parameters
    ----------
    x0: ArrayLike
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: ArrayLike
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    kappa: Optional[Union[ArrayLike, float]]
        Convergence. Surface density normalized by the critical surface density.

        *Unit: unitless*

    x: ArrayLike
        x-coordinates in the lens plane.

        *Unit: arcsec*

    y: ArrayLike
        y-coordinates in the lens plane.

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
    # Meneghetti eq 3.84
    return x * kappa, y * kappa


def potential_mass_sheet(x0, y0, kappa, x, y):
    """
    Compute the lensing potential. Here we use the Meneghetti lecture notes
    equation 3.81.

    Parameters
    ----------
    x0: ArrayLike
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: ArrayLike
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    kappa: Optional[Union[ArrayLike, float]]
        Convergence. Surface density normalized by the critical surface density.

        *Unit: unitless*

    x: ArrayLike
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: ArrayLike
        The y-coordinate of the lens.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        The lensing potential.

        *Unit: arcsec^2*

    """
    x, y = translate_rotate(x, y, x0, y0)
    # Meneghetti eq 3.81
    return (kappa / 2) * (x**2 + y**2)


def convergence_mass_sheet(kappa, x):
    """
    Compute the lensing convergence. In the case of a mass sheet, this is just
    the convergence value mapped to the input shape.

    Parameters
    ----------
    kappa: Optional[Union[ArrayLike, float]]
        Convergence. Surface density normalized by the critical surface density.

        *Unit: unitless*

    x: ArrayLike
        The x-coordinate of the lens. Only used for shape and device.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        The lensing potential.

        *Unit: arcsec^2*

    """
    # By definition
    return kappa * backend.ones_like(x)
