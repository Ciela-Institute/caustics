from torch import where

from ...utils import translate_rotate


def brightness_star(x0, y0, theta_s, Ie, x, y, gamma=0.0):

    x, y = translate_rotate(x, y, x0, y0)

    # Calculate the radial distance from the center
    impact_parameter = (x**2 + y**2) ** (1 / 2)
    # linear limb darkening
    mu = (1 - impact_parameter**2) ** (1 / 2)
    intensity = where(impact_parameter <= theta_s, Ie * (1 - gamma * (1 - mu)), 0)
    return intensity
