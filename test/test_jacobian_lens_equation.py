from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.cosmology import FlatLambdaCDM
from caustic.lenses import SIE
from caustic.utils import get_meshgrid

def test_jacobian_autograd_vs_finitediff():
    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIE(name="sie", cosmology=cosmology)
    thx, thy = get_meshgrid(0.01, 20, 20)
    
    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.5, 0.912, -0.442, 0.7, pi / 3, 1.4])

    # Evaluate Jacobian
    J_autograd = lens.jacobian_lens_equation(thx, thy, z_s, lens.pack(x))
    J_finitediff = lens.jacobian_lens_equation(thx, thy, z_s, lens.pack(x), method = "finitediff", pixelscale = torch.tensor(0.01))
    
    assert torch.sum(((J_autograd - J_finitediff)/J_autograd).abs() < 1e-3) > 0.8 * J_autograd.numel()

    
if __name__ == "__main__":
    test()
