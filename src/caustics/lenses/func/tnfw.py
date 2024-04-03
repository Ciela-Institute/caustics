import torch

from ...constants import arcsec_to_rad, G_over_c2, rad_to_arcsec
from ...utils import translate_rotate


def concentration_tnfw(mass, scale_radius, critical_density, d_l, DELTA=200.0):
    """
    Compute the concentration parameter "c" for a TNFW profile.

    Parameters
    ----------
    mass: Tensor
        The mass of the lens.

        *Unit: Msun*

    scale_radius: Tensor
        The scale radius of the TNFW lens.

        *Unit: arcsec*

    critical_density: Tensor
        The critical density of the universe.

        *Unit: Msun / Mpc^3*

    d_l: Tensor
        The angular diameter distance to the lens.

        *Unit: Mpc*

    DELTA: float
        The overdensity parameter.

        *Unit: unitless*

    Returns
    -------
    Tensor
        The concentration parameter "c" for a TNFW profile.

        *Unit: unitless*

    """
    r_delta = (3 * mass / (4 * torch.pi * DELTA * critical_density)) ** (1 / 3)  # fmt: skip
    return r_delta / (scale_radius * d_l * arcsec_to_rad)  # fmt: skip


def _F_batchable_tnfw(x):
    """
    Compute the function F(x) for a TNFW profile.

    Helper method from Baltz et al. 2009 equation A.5
    """
    return torch.where(
        x == 1,
        torch.ones_like(x),
        (
            (1 / x.to(dtype=torch.cdouble)).arccos()
            / (x.to(dtype=torch.cdouble) ** 2 - 1).sqrt()
        ).abs(),
    )


def _F_differentiable(x):
    """
    Compute the function F(x) for a TNFW profile.

    Helper method from Baltz et al. 2009 equation A.5
    """
    f = torch.ones_like(x)
    f[x < 1] = torch.arctanh((1.0 - x[x < 1] ** 2).sqrt()) / (1.0 - x[x < 1] ** 2).sqrt()  # fmt: skip
    f[x > 1] = torch.arctan( (x[x > 1] ** 2 - 1.0).sqrt()) / (x[x > 1] ** 2 - 1.0).sqrt()  # fmt: skip
    return f


def _L_tnfw(x, tau):
    """
    Helper method from Baltz et al. 2009 equation A.6
    """
    return (x / (tau + (tau**2 + x**2).sqrt())).log()  # fmt: skip


def scale_density_tnfw(c, critical_density, DELTA=200.0):
    return DELTA / 3 * critical_density * c**3 / ((1 + c).log() - c / (1 + c))  # fmt: skip


def M0_totmass_tnfw(mass, tau):
    """
    Rearranged from Baltz et al. 2009 equation A.4
    """
    return mass * (tau**2 + 1) ** 2 / (tau**2 * ((tau**2 - 1) * tau.log() + torch.pi * tau - (tau**2 + 1)))  # fmt: skip


def M0_scalemass_tnfw(scale_radius, c, critical_density, d_l, DELTA=200.0):
    """What M0 would be for an NFW

    Parameters
    ----------
    scale_radius: Tensor
        The scale radius of the TNFW lens.

        *Unit: arcsec*

    c: Tensor
        The concentration parameter of an NFW lens with the same parameters.

        *Unit: unitless*

    critical_density: Tensor
        The critical density of the universe.

        *Unit: Msun / Mpc^3*

    d_l: Tensor
        The angular diameter distance to the lens.

        *Unit: Mpc*

    DELTA: float
        The overdensity parameter.

        *Unit: unitless*
    """
    return 4 * torch.pi * (scale_radius * d_l * arcsec_to_rad) ** 3 * scale_density_tnfw(c, critical_density, DELTA)  # fmt: skip


def mass_enclosed_2d_tnfw(
    r,
    scale_radius,
    tau,
    M0,
    _F_mode="differentiable",
):
    """
    Total projected mass (Msun) within a radius r (arcsec). Given in Baltz et
    al. 2009 equation A.11

    Parameters
    ----------
    r: Tensor
        Radius at which to compute the enclosed mass.

        *Unit: arcsec*

    mass: Tensor
        Mass of the lens.

        *Unit: Msun*

    scale_radius: Tensor
        Scale radius of the TNFW lens.

        *Unit: arcsec*

    tau: Tensor
        Truncation scale. Ratio of truncation radius to scale radius.

        *Unit: unitless*

    Returns
    -------
    Tensor
        Integrated mass projected in infinite cylinder within radius r.

        *Unit: Msun*

    """
    g = r / scale_radius
    t2 = tau**2
    if _F_mode == "differentiable":
        F = _F_differentiable(g)
    elif _F_mode == "batchable":
        F = _F_batchable_tnfw(g)
    else:
        raise ValueError("_F_mode must be 'differentiable' or 'batchable'")
    L = _L_tnfw(g, tau)
    a1 = t2 / (t2 + 1) ** 2
    a2 = (t2 + 1 + 2 * (g**2 - 1)) * F
    a3 = tau * torch.pi
    a4 = (t2 - 1) * tau.log()
    a5 = (t2 + g**2).sqrt() * (-torch.pi + (t2 - 1) * L / tau)  # fmt: skip
    return M0 * a1 * (a2 + a3 + a4 + a5)


def physical_deflection_angle_tnfw(
    x0,
    y0,
    scale_radius,
    tau,
    x,
    y,
    M0,
    d_l,
    _F_mode="differentiable",
    s=0.0,
):
    """
    Compute the physical deflection angle for a TNFW profile. Converted from
    Baltz et al. 2009 equation A.18

    Parameters
    ----------
    x0: Tensor
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Tensor
        The y-coordinate of the lens center.

        *Unit: arcsec*

    scale_radius: Tensor
        The scale radius of the TNFW lens.

        *Unit: arcsec*

    tau: Tensor
        The truncation scale. Ratio of truncation radius to scale radius.

        *Unit: unitless*

    x: Tensor
        The x-coordinate in the lens plane.

        *Unit: arcsec*

    y: Tensor
        The y-coordinate in the lens plane.

        *Unit: arcsec*

    M0: Tensor
        The mass normalization constant. See `M0_totmass_tnfw` and
        `M0_scalemass_tnfw`.

        *Unit: Msun*

    d_l: Tensor
        The angular diameter distance to the lens.

        *Unit: Mpc*

    _F_mode: str
        The mode to use for computing the function F(x). Either "differentiable"
        or "batchable".

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*
    """

    x, y = translate_rotate(x, y, x0, y0)
    r = (x**2 + y**2).sqrt() + s
    theta = torch.arctan2(y, x)

    # The below actually equally comes from eq 2.13 in Meneghetti notes
    dr = mass_enclosed_2d_tnfw(r, scale_radius, tau, M0,_F_mode) / (
        r * d_l * arcsec_to_rad
    )  # note dpsi(u)/du = 2x*dpsi(x)/dx when u = x^2  # fmt: skip
    S = 4 * G_over_c2 * rad_to_arcsec
    return S * dr * theta.cos(), S * dr * theta.sin()


def convergence_tnfw(
    x0,
    y0,
    scale_radius,
    tau,
    x,
    y,
    critical_density,
    M0,
    d_l,
    _F_mode="differentiable",
    s=0.0,
):
    """
    Compute the dimensionless convergence for the TNFW. See Baltz et al. 2009
    equation A.8

        Parameters
    ----------
    x0: Tensor
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Tensor
        The y-coordinate of the lens center.

        *Unit: arcsec*

    scale_radius: Tensor
        The scale radius of the TNFW lens.

        *Unit: arcsec*

    tau: Tensor
        The truncation scale. Ratio of truncation radius to scale radius.

        *Unit: unitless*

    x: Tensor
        The x-coordinate in the lens plane.

        *Unit: arcsec*

    y: Tensor
        The y-coordinate in the lens plane.

        *Unit: arcsec*

    M0: Tensor
        The mass normalization constant. See `M0_totmass_tnfw` and
        `M0_scalemass_tnfw`.

        *Unit: Msun*

    d_l: Tensor
        The angular diameter distance to the lens.

        *Unit: Mpc*

    _F_mode: str
        The mode to use for computing the function F(x). Either "differentiable"
        or "batchable".

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*
    """
    x, y = translate_rotate(x, y, x0, y0)
    r = (x**2 + y**2).sqrt() + s
    g = r / scale_radius
    if _F_mode == "differentiable":
        F = _F_differentiable(g)
    elif _F_mode == "batchable":
        F = _F_batchable_tnfw(g)
    else:
        raise ValueError("_F_mode must be 'differentiable' or 'batchable'")
    L = _L_tnfw(g, tau)

    S = M0 / (2 * torch.pi * (scale_radius * d_l * arcsec_to_rad) ** 2)  # fmt: skip

    t2 = tau**2
    a1 = t2 / (t2 + 1) ** 2
    a2 = torch.where(g == 1, (t2 + 1) / 3.0, (t2 + 1) * (1 - F) / (g**2 - 1))  # fmt: skip
    a3 = 2 * F
    a4 = -torch.pi / (t2 + g**2).sqrt()
    a5 = (t2 - 1) * L / (tau * (t2 + g**2).sqrt())
    return a1 * (a2 + a3 + a4 + a5) * S / critical_density  # fmt: skip


def potential_tnfw(
    x0,
    y0,
    scale_radius,
    tau,
    x,
    y,
    M0,
    d_l,
    d_s,
    d_ls,
    _F_mode="differentiable",
    s=0.0,
):
    """
    Compute the lensing potential for a TNFW profile. See Baltz et al. 2009
    equation A.14

        Parameters
    ----------
    x0: Tensor
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Tensor
        The y-coordinate of the lens center.

        *Unit: arcsec*

    scale_radius: Tensor
        The scale radius of the TNFW lens.

        *Unit: arcsec*

    tau: Tensor
        The truncation scale. Ratio of truncation radius to scale radius.

        *Unit: unitless*

    x: Tensor
        The x-coordinate in the lens plane.

        *Unit: arcsec*

    y: Tensor
        The y-coordinate in the lens plane.

        *Unit: arcsec*

    M0: Tensor
        The mass normalization constant. See `M0_totmass_tnfw` and
        `M0_scalemass_tnfw`.

        *Unit: Msun*

    d_l: Tensor
        The angular diameter distance to the lens.

        *Unit: Mpc*

    d_s: Tensor
        The angular diameter distance to the source.

        *Unit: Mpc*

    d_ls: Tensor
        The angular diameter distance between the lens and the source.

        *Unit: Mpc*

    _F_mode: str
        The mode to use for computing the function F(x). Either "differentiable"
        or "batchable".

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*
    """
    x, y = translate_rotate(x, y, x0, y0)
    r = (x**2 + y**2).sqrt() + s
    g = r / scale_radius
    t2 = tau**2
    u = g**2
    if _F_mode == "differentiable":
        F = _F_differentiable(g)
    elif _F_mode == "batchable":
        F = _F_batchable_tnfw(g)
    else:
        raise ValueError("_F_mode must be 'differentiable' or 'batchable'")
    L = _L_tnfw(g, tau)

    # fmt: off
    S = 2 * M0 * G_over_c2 * (d_ls / d_s) / (d_l * arcsec_to_rad**2)
    a1 = 1 / (t2 + 1) ** 2
    a2 = 2 * torch.pi * t2 * (tau - (t2 + u).sqrt() + tau * (tau + (t2 + u).sqrt()).log())
    a3 = 2 * (t2 - 1) * tau * (t2 + u).sqrt() * L
    a4 = t2 * (t2 - 1) * L**2
    a5 = 4 * t2 * (u - 1) * F
    a6 = t2 * (t2 - 1) * (1 / g.to(dtype=torch.cdouble)).arccos().abs() ** 2
    a7 = t2 * ((t2 - 1) * tau.log() - t2 - 1) * u.log()
    a8 = t2 * ((t2 - 1) * tau.log() * (4 * tau).log() + 2 * (tau / 2).log() - 2 * tau * (tau - torch.pi) * (2 * tau).log())

    return S * a1 * (a2 + a3 + a4 + a5 + a6 + a7 - a8)
