import torch

from ...utils import translate_rotate


def reduced_deflection_angle_multipole(x0, y0, m, a_m, phi_m, x, y):
    """
    Calculates the reduced deflection angle.

    Parameters
    ----------
    x: Tensor
        x-coordinates in the lens plane.
    y: Tensor
        y-coordinates in the lens plane.
    z_s: Tensor
        Redshifts of the sources.
    x0: Tensor
        x-coordinate of the center of the lens.
    y0: Tensor
        y-coordinate of the center of the lens.
    m: Tensor
        The multipole order(s).
    a_m: Tensor
        The multipole amplitude(s).
    phi_m: Tensor
        The multipole orientation(s).

    Returns
    -------
    tuple[Tensor, Tensor]
        The reduced deflection angles in the x and y directions.

    Equation (B11) and (B12) https://arxiv.org/pdf/1307.4220, Xu et al. 2014
    """
    x, y = translate_rotate(x, y, x0, y0)

    phi = torch.arctan2(y, x).reshape(1, -1)
    m = m.reshape(-1, 1)
    ax = torch.cos(phi) * a_m / (1 - m**2) * torch.cos(m * (phi - phi_m)) + torch.sin(
        phi
    ) * m * a_m / (1 - m**2) * torch.sin(m * (phi - phi_m))
    ax = ax.sum(dim=0).reshape(x.shape)
    ay = torch.sin(phi) * a_m / (1 - m**2) * torch.cos(m * (phi - phi_m)) - torch.cos(
        phi
    ) * m * a_m / (1 - m**2) * torch.sin(m * (phi - phi_m))
    ay = ay.sum(dim=0).reshape(y.shape)

    return ax, ay


def potential_multipole(x0, y0, m, a_m, phi_m, x, y):
    """
    Compute the lensing potential.

    Parameters
    ----------
    x: Tensor
        x-coordinates in the lens plane.
    y: Tensor
        y-coordinates in the lens plane.
    z_s: Tensor
        Redshifts of the sources.
    x0: Tensor
        x-coordinate of the center of the lens.
    y0: Tensor
        y-coordinate of the center of the lens.
    m: Tensor
        The multipole order(s).
    a_m: Tensor
        The multipole amplitude(s).
    phi_m: Tensor
        The multipole orientation(s).

    Returns
    -------
    potential: Tensor
        Lensing potential.

        *Unit: arcsec^2*

    Equation (B11) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

    """
    x, y = translate_rotate(x, y, x0, y0)
    r = torch.sqrt(x**2 + y**2)
    phi = torch.arctan2(y, x).reshape(1, -1)
    m = m.reshape(-1, 1)

    potential = r * a_m / (1 - m**2) * torch.cos(m * (phi - phi_m))
    potential = potential.sum(dim=0).reshape(x.shape)
    return potential


def convergence_multipole(x0, y0, m, a_m, phi_m, x, y):
    """
    Compute the lensing convergence.

    Parameters
    ----------
    x: Tensor
        x-coordinates in the lens plane.
    y: Tensor
        y-coordinates in the lens plane.
    z_s: Tensor
        Redshifts of the sources.
    x0: Tensor
        x-coordinate of the center of the lens.
    y0: Tensor
        y-coordinate of the center of the lens.
    m: Tensor
        The multipole order(s).
    a_m: Tensor
        The multipole amplitude(s).
    phi_m: Tensor
        The multipole orientation(s).

    Returns
    -------
    convergence: Tensor
        Lensing convergence.

        *Unit: unitless*

    Equation (B10) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

    """
    x, y = translate_rotate(x, y, x0, y0)
    r = torch.sqrt(x**2 + y**2)
    phi = torch.arctan2(y, x).reshape(1, -1)
    m = m.reshape(-1, 1)

    convergence = 1 / (2 * r) * a_m * torch.cos(m * (phi - phi_m))
    convergence = convergence.sum(dim=0).reshape(x.shape)
    return convergence
