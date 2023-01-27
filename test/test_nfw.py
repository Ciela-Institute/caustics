from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.cosmology import FlatLambdaCDMCosmology
from caustic.lenses import NFW


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    lens = NFW()
    lens_model_list = ["NFW"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    z_l = torch.tensor(0.5)
    z_s = torch.tensor(2.1)
    cosmology = FlatLambdaCDMCosmology()
    thx0 = torch.tensor(0.457)
    thy0 = torch.tensor(0.141)
    m = torch.tensor(1e12)
    c = torch.tensor(15.0)
    args = (z_l, z_s, cosmology, thx0, thy0, m, c)
    # TODO: parse parameters into form required by lenstronomy
    # See https://lenstronomy.readthedocs.io/en/latest/lenstronomy.LensModel.Profiles.html#lenstronomy.LensModel.Profiles.nfw.NFW
    # e1, e2 = param_util.phi_q2_ellipticity(phi=phi.item(), q=q.item())
    # kwargs_ls = [
    #     {
    #         "theta_E": b.item(),
    #         "e1": e1,
    #         "e2": e2,
    #         "center_x": thx0.item(),
    #         "center_y": thy0.item(),
    #     }
    # ]

    # TODO: uncomment
    # lens_test_helper(lens, lens_ls, args, kwargs_ls, rtol, atol)
