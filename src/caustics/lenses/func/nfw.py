import torch


def scale_radius_nfw(critical_density, m, c, DELTA=200.0):
    """
    Compute the scale radius of the NFW profile.

    Parameters
    ----------
    critical_density: Union[Tensor, float]
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    m: Union[Tensor, float]
        The mass of the halo.

        *Unit: Msun*

    c: Union[Tensor, float]
        The concentration parameter of the halo.

        *Unit: unitless*

    DELTA: float
        The overdensity parameter. Amount above the critical surface density at which the scale radius is computed

        *Unit: unitless*

    Returns
    -------
    Union[Tensor, float]
        The scale radius of the NFW profile.

        *Unit: Mpc*

    """
    r_delta = (3 * m / (4 * torch.pi * DELTA * critical_density)) ** (1 / 3)  # fmt: skip
    return 1 / c * r_delta


def scale_density_nfw(critical_density, c, DELTA=200.0):
    """
    Compute the scale density of the NFW profile.

    Parameters
    ----------
    critical_density: Union[Tensor, float]
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    c: Union[Tensor, float]
        The concentration parameter of the halo.

        *Unit: unitless*

    DELTA: float
        The overdensity parameter. Amount above the critical surface density at which the scale radius is computed

        *Unit: unitless*

    Returns
    -------
    Union[Tensor, float]
        The scale density of the NFW profile.

        *Unit: solar mass per square kiloparsec*

    """
    return DELTA / 3 * critical_density * c**3 / ((1 + c).log() - c / (1 + c))  # fmt: skip


def convergence_s_nfw(critical_surface_density, critical_density, m, c, DELTA):
    """
    Compute the dimensionaless surface mass density of the lens.

    critical_surface_density: Union[Tensor, float]
        The critical surface density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^2*

    critical_density: Union[Tensor, float]
        The critical density of the Universe at the appropriate redshift.

        *Unit: Msun/Mpc^3*

    m: Union[Tensor, float]
        The mass of the halo.

        *Unit: Msun*

    c: Union[Tensor, float]
        The concentration parameter of the halo.

        *Unit: unitless*

    DELTA: float
        The overdensity parameter. Amount above the critical surface density at which the scale radius is computed

        *Unit: unitless*
    """
    Rs = scale_radius_nfw(critical_density, m, c, DELTA)
    Ds = scale_density_nfw(critical_density, c, DELTA)
    return Rs * Ds / critical_surface_density


def _f_differentiable(x):
    """
    Helper method for computing deflection angles.

    Parameters
    ----------
    x: Union[Tensor, float]
        The input to the function.

    Returns
    -------
    Union[Tensor, float]
        The value of the function f(x).

    """
    # TODO: generalize beyond torch, or patch Tensor
    f = torch.zeros_like(x)
    f[x > 1] = 1 - 2 / (x[x > 1] ** 2 - 1).sqrt() * ((x[x > 1] - 1) / (x[x > 1] + 1)).sqrt().arctan()  # fmt: skip
    f[x < 1] = 1 - 2 / (1 - x[x < 1] ** 2).sqrt() * ((1 - x[x < 1]) / (1 + x[x < 1])).sqrt().arctanh()  # fmt: skip
    return f


def _f_batchable(x):
    """
    Helper method for computing deflection angles.

    Parameters
    ----------
    x: Union[Tensor, float]
        The input to the function.

    Returns
    -------
    Union[Tensor, float]
        The value of the function f(x).

    """
    # TODO: generalize beyond torch, or patch Tensor
    # fmt: off
    return torch.where(
        x > 1,
        1 - 2 / (x**2 - 1).sqrt() * ((x - 1) / (x + 1)).sqrt().arctan(),
        torch.where(
            x < 1,
            1 - 2 / (1 - x**2).sqrt() * ((1 - x) / (1 + x)).sqrt().arctanh(),
            torch.zeros_like(x),  # where: x == 1
        ),
    )
    # fmt: on
