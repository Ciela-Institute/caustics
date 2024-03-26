import torch

from ...constants import arcsec_to_rad, G_over_c2
from ...utils import translate_rotate


def convergence_0_pseudo_jaffe(
    mass, core_radius, scale_radius, d_l, critical_surface_density
):
    """
    Compute the convergence (dimensionless surface mass density).

    Parameters
    ----------
    mass: Optional[Tensor]
        Total mass of the lens (Msun).

        *Unit: Msun*

    core_radius: Optional[Tensor]
        Core radius of the lens.

        *Unit: arcsec*

    scale_radius: Optional[Tensor]
        Scaling radius of the lens.

        *Unit: arcsec*

    Returns
    --------
    Tensor
        The convergence (dimensionless surface mass density) at the center of the pseudo jaffe.

        *Unit: unitless*

    """
    return mass / (2 * torch.pi * critical_surface_density * core_radius * scale_radius * (d_l * arcsec_to_rad) ** 2)  # fmt: skip


def mass_enclosed_2d_pseudo_jaffe(radius, mass, core_radius, scale_radius, d_l, s=0.0):
    """
    Compute the mass enclosed within a given radius.

    Parameters
    ----------
    radius: Optional[Tensor]
        Radius at which to calculate enclosed mass (arcsec).

            *Unit: arcsec*
    """
    theta = radius + s
    surface_density_0 = convergence_0_pseudo_jaffe(
        mass, core_radius, scale_radius, d_l, 1.0
    )  # Msun / Mpc^2
    total_mass = (
        2
        * torch.pi
        * surface_density_0
        * core_radius
        * scale_radius
        * (d_l * arcsec_to_rad) ** 2
    )  # Msun
    frac_enclosed_num = (
        (core_radius**2 + theta**2).sqrt()
        - core_radius
        - (scale_radius**2 + theta**2).sqrt()
        + scale_radius
    )  # arcsec
    frac_enclosed_denom = scale_radius - core_radius  # arcsec
    return total_mass * frac_enclosed_num / frac_enclosed_denom


def reduced_deflection_angle_pseudo_jaffe(
    x0, y0, mass, core_radius, scale_radius, x, y, d_l, critical_surface_density, s=0.0
):
    """
    Compute the reduced deflection angle.

    Parameters
    ----------
    radius: Optional[Tensor]
        Radius at which to calculate the reduced deflection angle (arcsec).

            *Unit: arcsec*
    """
    x, y = translate_rotate(x, y, x0, y0)
    R = (x**2 + y**2).sqrt() + s
    f = R / core_radius / (1 + (1 + (R / core_radius) ** 2).sqrt()) - R / (scale_radius * (1 + (1 + (R / scale_radius) ** 2).sqrt()))  # fmt: skip
    alpha = 2 * convergence_0_pseudo_jaffe(mass, core_radius, scale_radius, d_l, critical_surface_density) * core_radius * scale_radius / (scale_radius - core_radius) * f  # fmt: skip
    ax = alpha * x / R
    ay = alpha * y / R
    return ax, ay


def potential_pseudo_jaffe(
    x0, y0, mass, core_radius, scale_radius, x, y, d_l, d_s, d_ls, s=0.0
):
    x, y = translate_rotate(x, y, x0, y0)

    R_squared = x**2 + y**2 + s  # arcsec^2
    surface_density_0 = convergence_0_pseudo_jaffe(
        mass, core_radius, scale_radius, d_l, 1.0
    )  # Msun / Mpc^2

    coeff = -(
        8
        * torch.pi
        * G_over_c2
        * surface_density_0
        * (d_l * d_ls / d_s)
        * core_radius
        * scale_radius
        / (scale_radius - core_radius)
    )  # arcsec

    scale_a = (scale_radius**2 + R_squared).sqrt()  # arcsec
    scale_b = (core_radius**2 + R_squared).sqrt()  # arcsec
    scale_c = (
        core_radius * (core_radius + (core_radius**2 + R_squared).sqrt()).log()
    )  # arcsec
    scale_d = (
        scale_radius * (scale_radius + (scale_radius**2 + R_squared).sqrt()).log()
    )  # arcsec
    scale_factor = scale_a - scale_b + scale_c - scale_d  # arcsec
    return coeff * scale_factor


def convergence_pseudo_jaffe(
    x0, y0, mass, core_radius, scale_radius, x, y, d_l, critical_surface_density, s=0.0
):
    """
    Compute the convergence (dimensionless surface mass density).

    Parameters
    ----------
    mass: Optional[Tensor]
        Total mass of the lens
    """

    x, y = translate_rotate(x, y, x0, y0)
    R_squared = x**2 + y**2 + s
    coeff = convergence_0_pseudo_jaffe(mass, core_radius, scale_radius, d_l, critical_surface_density) * core_radius * scale_radius / (scale_radius - core_radius)  # fmt: skip
    return coeff * (1 / (core_radius**2 + R_squared).sqrt() - 1 / (scale_radius**2 + R_squared).sqrt())  # fmt: skip
