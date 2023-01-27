from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.lenses import SIE


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    lens = SIE()
    lens_model_list = ["SIE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    thx0 = torch.tensor(0.912)
    thy0 = torch.tensor(-0.442)
    q = torch.tensor(0.7)
    phi = torch.tensor(pi / 3)
    b = torch.tensor(1.4)
    s = torch.tensor(0.0)
    args = (None, None, None, thx0, thy0, q, phi, b, s)
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi.item(), q=q.item())
    kwargs_ls = [
        {
            "theta_E": b.item(),
            "e1": e1,
            "e2": e2,
            "center_x": thx0.item(),
            "center_y": thy0.item(),
        }
    ]

    lens_test_helper(lens, lens_ls, args, kwargs_ls, rtol, atol)
