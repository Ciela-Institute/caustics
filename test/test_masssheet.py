from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import MassSheet
from caustics.utils import get_meshgrid


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = MassSheet(name="sheet", cosmology=cosmology)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.5, 0., 0., 0.7])

    thx, thy = get_meshgrid(0.01, 10, 10)
    
    ax, ay = lens.reduced_deflection_angle(thx, thy, z_s, *x)

    p = lens.potential(thx, thy, z_s, *x)

    c = lens.convergence(thx, thy, z_s, *x)
