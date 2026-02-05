from ...backend_obj import backend
from ...utils import translate_rotate


def reduced_deflection_angle_multipole(x0, y0, m, a_m, phi_m, x, y):
    """
    Calculates the reduced deflection angle.

    Parameters
    ----------
    x: ArrayLike
        x-coordinates in the lens plane.
    y: ArrayLike
        y-coordinates in the lens plane.
    x0: ArrayLike
        x-coordinate of the center of the lens.
    y0: ArrayLike
        y-coordinate of the center of the lens.
    m: ArrayLike
        The multipole order(s).
    a_m: ArrayLike
        The multipole amplitude(s).
    phi_m: ArrayLike
        The multipole orientation(s).

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        The reduced deflection angles in the x and y directions.

    Equation (B11) and (B12) https://arxiv.org/pdf/1307.4220, Xu et al. 2014
    """
    x, y = translate_rotate(x, y, x0, y0)

    phi = backend.arctan2(y, x).reshape(1, -1)
    m = m.reshape(-1, 1)
    a_m = a_m.reshape(-1, 1)
    phi_m = phi_m.reshape(-1, 1)
    ax = backend.cos(phi) * a_m / (1 - m**2) * backend.cos(
        m * (phi - phi_m)
    ) + backend.sin(phi) * m * a_m / (1 - m**2) * backend.sin(m * (phi - phi_m))
    ax = backend.sum(ax, dim=0).reshape(x.shape)
    ay = backend.sin(phi) * a_m / (1 - m**2) * backend.cos(
        m * (phi - phi_m)
    ) - backend.cos(phi) * m * a_m / (1 - m**2) * backend.sin(m * (phi - phi_m))
    ay = backend.sum(ay, dim=0).reshape(y.shape)

    return ax, ay


def potential_multipole(x0, y0, m, a_m, phi_m, x, y):
    """
    Compute the lensing potential.

    Parameters
    ----------
    x: ArrayLike
        x-coordinates in the lens plane.
    y: ArrayLike
        y-coordinates in the lens plane.
    x0: ArrayLike
        x-coordinate of the center of the lens.
    y0: ArrayLike
        y-coordinate of the center of the lens.
    m: ArrayLike
        The multipole order(s).
    a_m: ArrayLike
        The multipole amplitude(s).
    phi_m: ArrayLike
        The multipole orientation(s).

    Returns
    -------
    potential: ArrayLike
        Lensing potential.

        *Unit: arcsec^2*

    Equation (B11) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

    """
    x, y = translate_rotate(x, y, x0, y0)
    r = backend.sqrt(x**2 + y**2).reshape(1, -1)
    phi = backend.arctan2(y, x).reshape(1, -1)
    m = m.reshape(-1, 1)
    a_m = a_m.reshape(-1, 1)
    phi_m = phi_m.reshape(-1, 1)

    potential = r * a_m / (1 - m**2) * backend.cos(m * (phi - phi_m))
    potential = backend.sum(potential, dim=0).reshape(x.shape)
    return potential


def convergence_multipole(x0, y0, m, a_m, phi_m, x, y):
    """
    Compute the lensing convergence.

    Parameters
    ----------
    x: ArrayLike
        x-coordinates in the lens plane.
    y: ArrayLike
        y-coordinates in the lens plane.
    x0: ArrayLike
        x-coordinate of the center of the lens.
    y0: ArrayLike
        y-coordinate of the center of the lens.
    m: ArrayLike
        The multipole order(s).
    a_m: ArrayLike
        The multipole amplitude(s).
    phi_m: ArrayLike
        The multipole orientation(s).

    Returns
    -------
    convergence: ArrayLike
        Lensing convergence.

        *Unit: unitless*

    Equation (B10) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

    """
    x, y = translate_rotate(x, y, x0, y0)
    r = backend.sqrt(x**2 + y**2).reshape(1, -1)
    phi = backend.arctan2(y, x).reshape(1, -1)
    m = m.reshape(-1, 1)
    a_m = a_m.reshape(-1, 1)
    phi_m = phi_m.reshape(-1, 1)

    convergence = 1 / (2 * r) * a_m * backend.cos(m * (phi - phi_m))
    convergence = backend.sum(convergence, dim=0).reshape(x.shape)
    return convergence
