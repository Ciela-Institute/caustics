import torch

from ...utils import translate_rotate, derotate


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
    m: Tensor, int
        The multipole order.
    a_m: Tensor
        The multipole amplitude.
    phi_m: Tensor
        The multipole orientation.

    Returns
    -------
    tuple[Tensor, Tensor]
        The reduced deflection angles in the x and y directions.

    Equation (B11) and (B12) https://arxiv.org/pdf/1307.4220, Xu et al. 2014
    """
    x, y = translate_rotate(x, y, x0, y0)
    
    phi = torch.arctan2(y, x)
    ax = torch.cos(phi) * a_m / (1 - m**2) * torch.cos(m * (phi - phi_m)) + torch.sin(phi) * m * a_m / (1 - m**2) * torch.sin(m * (phi - phi_m))
    ay = torch.sin(phi) * a_m / (1 - m**2) * torch.cos(m * (phi - phi_m)) - torch.cos(phi) * m * a_m / (1 - m**2) * torch.sin(m * (phi - phi_m))

    return ax, ay #derotate(ax, ay, phi)


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
    m: Tensor, int
        The multipole order.
    a_m: Tensor
        The multipole amplitude.
    phi_m: Tensor
        The multipole orientation.

    Returns
    -------
    potential: Tensor
        Lensing potential.

        *Unit: arcsec^2*

    Equation (B11) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

    """
    x, y = translate_rotate(x, y, x0, y0)
    r = torch.sqrt(x**2 + y**2)
    phi = torch.arctan2(y, x)
    return r * a_m / (1 - m**2) * torch.cos(m * (phi - phi_m)) 


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
    m: Tensor, int
        The multipole order.
    a_m: Tensor
        The multipole amplitude.
    phi_m: Tensor
        The multipole orientation.

    Returns
    -------
    convergence: Tensor
        Lensing convergence.

        *Unit: unitless*

    Equation (B10) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

    """
    x, y = translate_rotate(x, y, x0, y0)
    r = torch.sqrt(x**2 + y**2)
    phi = torch.arctan2(y, x)
    return 1/r * a_m * torch.cos(m * (phi - phi_m)) 
