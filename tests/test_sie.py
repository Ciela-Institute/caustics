from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIE(name="sie", cosmology=cosmology)
    lens_model_list = ["SIE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.5, 0.912, -0.442, 0.7, pi / 3, 1.4])
    e1, e2 = param_util.phi_q2_ellipticity(phi=x[4].item(), q=x[3].item())
    kwargs_ls = [
        {
            "theta_E": x[5].item(),
            "e1": e1,
            "e2": e2,
            "center_x": x[1].item(),
            "center_y": x[2].item(),
        }
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol)


if __name__ == "__main__":
    test()
