import torch

from ...utils import translate_rotate


def reduced_deflection_angle_mass_sheet(x0, y0, surface_density, x, y):
    """
    Compute the reduced deflection angles. Here we use the Meneeghetti lecture
    notes equation 3.84.

    Parameters
    ----------
    x0: Tensor
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Tensor
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    surface_density: Optional[Union[Tensor, float]]
        Surface density normalized by the critical surface density.

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
    x, y = translate_rotate(x, y, x0, y0)
    # Meneghetti eq 3.84
    ax = x * surface_density
    ay = y * surface_density
    return ax, ay


def potential_mass_sheet(x0, y0, surface_density, x, y):
    """
    Compute the lensing potential. Here we use the Meneghetti lecture notes
    equation 3.81.

    Parameters
    ----------
    x0: Tensor
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Tensor
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    surface_density: Optional[Union[Tensor, float]]
        Surface density normalized by the critical surface density.

        *Unit: unitless*

    x: Tensor
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: Tensor
        The y-coordinate of the lens.

        *Unit: arcsec*

    Returns
    -------
    Tensor
        The lensing potential.

        *Unit: arcsec^2*

    """
    x, y = translate_rotate(x, y, x0, y0)
    # Meneghetti eq 3.81
    return (surface_density / 2) * (x**2 + y**2)


def convergence_mass_sheet(surface_density, x):
    """
    Compute the lensing convergence. In the case of a mass sheet, this is just
    the convergence value mapped to the input shape.

    Parameters
    ----------
    surface_density: Optional[Union[Tensor, float]]
        Surface density normalized by the critical surface density.

        *Unit: unitless*

    x: Tensor
        The x-coordinate of the lens. Only used for shape and device.

        *Unit: arcsec*

    Returns
    -------
    Tensor
        The lensing potential.

        *Unit: arcsec^2*

    """
    # By definition
    return surface_density * torch.ones_like(x)
