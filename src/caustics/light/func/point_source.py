from torch import Tensor, tensor, where

from ...utils import translate_rotate



def brightness_point(x0, y0, theta_s, Ie, x, y, s=0.0):

    x, y = translate_rotate(x, y, x0, y0, phi=None) # phi=None because a point source is spherically symmetric
    
    mask = (x**2 + y**2)**(1/2) <= theta_s #Uniform surface brightness inside source
    
    # Return Ie where the mask is True, otherwise return 0
    return where(mask, Ie, tensor(0.0))