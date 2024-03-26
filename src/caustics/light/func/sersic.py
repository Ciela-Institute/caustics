from torch import Tensor

from ...utils import translate_rotate, to_elliptical


def k_sersic(n: Tensor) -> Tensor:
    """
    Computes the value of k for the Sersic profile.

    Parameters
    ----------
    n: Tensor
        The Sersic index, which describes the degree of concentration of the source.

        *Unit: unitless*

    Returns
    --------
    k: Tensor
        The value of k for the Sersic profile.

        *Unit: unitless*

    """
    return (
        2 * n
        - 1 / 3
        + 4 / (405 * n)
        + 46 / (25515 * n**2)
        + 131 / (1148175 * n**3)
        - 2194697 / (30690717750 * n**4)
    )


def k_lenstronomy(n: Tensor) -> Tensor:
    """
    Computes the value of k for the Sersic profile, as used in the lenstronomy package.

    Parameters
    ----------
    n: Tensor
        The Sersic index, which describes the degree of concentration of the source.

        *Unit: unitless*

    Returns
    --------
    k: Tensor
        The value of k for the Sersic profile.

        *Unit: unitless*

    """
    return 1.9992 * n - 0.3271


def brightness_sersic(x0, y0, q, phi, n, Re, Ie, x, y, k, s=0.0):

    x, y = translate_rotate(x, y, x0, y0, phi)
    ex, ey = to_elliptical(x, y, q)
    e = (ex**2 + ey**2).sqrt() + s

    exponent = -k * ((e / Re) ** (1 / n) - 1)
    return Ie * exponent.exp()
