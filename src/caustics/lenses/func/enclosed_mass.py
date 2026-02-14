from ...utils import translate_rotate, derotate
from ...backend_obj import backend
from ...constants import G_over_c2
import numpy as np


def physical_deflection_angle_enclosed_mass(x0, y0, q, phi, enclosed_mass, x, y, s=0.0):
    """
    Calculate the reduced deflection angle for a lens with an enclosed mass
    profile. See the Meneghetti lecture notes Equation 3.19 for the physical
    deflection angle.

    Parameters
    ----------
    x0: ArrayLike
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: ArrayLike
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: ArrayLike
        The axis ratio of the lens. ratio of semi-minor to semi-major axis (b/a).

        *Unit: unitless*

    phi: ArrayLike
        The position angle of the lens. The angle relative to the positive x-axis.

        *Unit: radians*

    enclosed_mass: Callable
        The enclosed mass profile function, solely a function of r.

    x: ArrayLike
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: ArrayLike
        The y-coordinate of the lens.

        *Unit: arcsec*

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        The physical deflection angle.
    """
    x, y = translate_rotate(x, y, x0, y0, phi)
    r = backend.sqrt(x**2 / q + q * y**2) + s
    alpha = 4 * G_over_c2 * enclosed_mass(r) / r**2
    ax = alpha * x / (q * r)
    ay = alpha * y * q / r
    return derotate(ax, ay, phi)


def convergence_enclosed_mass(
    x0, y0, q, phi, enclosed_mass, x, y, critical_surface_density, s=0.0
):
    """
    Calculate the convergence for a lens with an enclosed mass profile. See the
    Meneghetti lecture notes Equation 3.28 for the convergence from an enclosed
    mass profile.

    Parameters
    ----------
    x0: ArrayLike
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: ArrayLike
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: ArrayLike
        The axis ratio of the lens. ratio of semi-minor to semi-major axis (b/a).

        *Unit: unitless*

    phi: ArrayLike
        The position angle of the lens. The angle relative to the positive x-axis.

        *Unit: radians*

    enclosed_mass: Callable
        The enclosed mass profile function, solely a function of r.

    x: ArrayLike
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: ArrayLike
        The y-coordinate of the lens.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        The convergence.
    """
    x, y = translate_rotate(x, y, x0, y0, phi)
    r = backend.sqrt(x**2 / q + q * y**2) + s
    return (
        0.5
        * backend.vmap(backend.grad(enclosed_mass))(r.reshape(-1)).reshape(r.shape)
        / (r * np.pi * critical_surface_density)
    )
