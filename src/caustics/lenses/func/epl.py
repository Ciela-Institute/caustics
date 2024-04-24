import torch

from ...utils import translate_rotate, derotate


def _r_omega(z, t, q, n_iter):
    """
    Iteratively compute the omega term given in Tessore et al. 2015 equation 23.
    This is done using the approximation given in Tessore et al. 2015 equation
    29. Note that in Tessore et al. 2015 the omega term is independent of
    radius, our "r_omega" term includes an r-factor for numerical reasons.
    """
    # constants
    f = (1.0 - q) / (1.0 + q)
    phi = z / torch.conj(z)

    # first term in series
    omega_i = z
    part_sum = omega_i

    for i in range(1, n_iter):
        factor = (2.0 * i - (2.0 - t)) / (2.0 * i + (2.0 - t))  # fmt: skip
        omega_i = -f * factor * phi * omega_i  # fmt: skip
        part_sum = part_sum + omega_i  # fmt: skip

    return part_sum


def reduced_deflection_angle_epl(x0, y0, q, phi, b, t, x, y, n_iter):
    """
    Calculate the reduced deflection angle. Given in Tessore et al. 2015 equation 13.

    Parameters
    ----------
    x0: Tensor
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Tensor
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: Tensor
        The axis ratio of the lens. Semi-minor over semi-major axis lengths.

        *Unit: unitless*

    phi: Tensor
        The orientation angle of the lens (position angle).

        *Unit: radians*

    b: Tensor
        Scale length of the lens.

        *Unit: arcsec*

    t: Tensor
        Power law slope (`gamma-1`) of the lens.
        If not provided, it is considered as a free parameter.

        *Unit: unitless*

    x: Tensor
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: Tensor
        The y-coordinate of the lens.

        *Unit: arcsec*

    n_iter: int
        Number of iterations for the iterative solver.

        *Unit: number*

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

    # follow Tessore et al 2015 (eq. 5)
    z = q * x + y * 1j
    r = torch.abs(z)

    # Tessore et al 2015 (eq. 23)
    r_omega = _r_omega(z, t, q, n_iter)
    # Tessore et al 2015 (eq. 13)
    alpha_c = 2.0 / (1.0 + q) * (b / r) ** t * r_omega  # fmt: skip

    alpha_real = torch.nan_to_num(alpha_c.real, posinf=10**10, neginf=-(10**10))
    alpha_imag = torch.nan_to_num(alpha_c.imag, posinf=10**10, neginf=-(10**10))
    return derotate(alpha_real, alpha_imag, phi)


def potential_epl(x0, y0, q, phi, b, t, x, y, n_iter):
    """
    Calculate the potential for the EPL as defined in Tessore et al. 2015 equation 15.

    Parameters
    ----------
    x0: Tensor
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Tensor
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: Tensor
        The axis ratio of the lens. Semi-minor over semi-major axis lengths.

        *Unit: unitless*

    phi: Tensor
        The orientation angle of the lens (position angle).

        *Unit: radians*

    b: Tensor
        Scale length of the lens.

        *Unit: arcsec*

    t: Tensor
        Power law slope (`gamma-1`) of the lens.
        If not provided, it is considered as a free parameter.

        *Unit: unitless*

    x: Tensor
        The x-coordinate of the lens.

        *Unit: arcsec*

    y: Tensor
        The y-coordinate of the lens.

        *Unit: arcsec*

    n_iter: int
        Number of iterations for the iterative solver.

        *Unit: number*

    Returns
    --------
    x_component: Tensor
        The x-component of the deflection angle.

        *Unit: arcsec*

    y_component: Tensor
        The y-component of the deflection angle.

        *Unit: arcsec*

    """
    ax, ay = reduced_deflection_angle_epl(x0, y0, q, phi, b, t, x, y, n_iter)
    ax, ay = derotate(ax, ay, -phi)
    x, y = translate_rotate(x, y, x0, y0, phi)
    return (x * ax + y * ay) / (2 - t)  # fmt: skip


def convergence_epl(x0, y0, q, phi, b, t, x, y, s=0.0):
    """
    Calculate the reduced deflection angle.

    See Tessore et al. 2015 equation 2.

    Parameters
    ----------
    x0: Tensor
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Tensor
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: Tensor
        The axis ratio of the lens. Semi-minor over semi-major axis lengths.

        *Unit: unitless*

    phi: Tensor
        The orientation angle of the lens (position angle).

        *Unit: radians*

    b: Tensor
        Scale length of the lens.

        *Unit: arcsec*

    t: Tensor
        Power law slope (`gamma-1`) of the lens.
        If not provided, it is considered as a free parameter.

        *Unit: unitless*

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
    psi = (q**2 * x**2 + y**2 + s**2).sqrt()  # fmt: skip
    return (2 - t) / 2 * (b / psi) ** t  # fmt: skip
