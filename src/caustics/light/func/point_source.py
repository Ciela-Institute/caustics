from torch import Tensor, tensor, where

from ...utils import translate_rotate



def brightness_point(x0, y0, theta_s, Ie, x, y, gamma=1.0, s=0.0):

    x, y = translate_rotate(x, y, x0, y0, phi=None) # phi=None because a point source is spherically symmetric

    # todo add linear limb darkening 
    # Calculate the radial distance from the center in units of the stellar radius
    impact_parameter = (x**2 + y**2)**(1/2) / theta_s
    #linear limb darkening
    mu = (1 - impact_parameter**2)**(1/2)
    intensity = where(impact_parameter <= 1, Ie * (1 - gamma * (1 - mu)), 0)
    return intensity
