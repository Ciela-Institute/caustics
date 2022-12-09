import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.cosmology import FlatLambdaCDMCosmology
from caustic.lenses import PseudoJaffe


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    lens = PseudoJaffe()
    lens_model_list = ["PJAFFE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    cosmology = FlatLambdaCDMCosmology()
    z_l = torch.tensor(0.5)
    z_s = torch.tensor(2.1)
    thx0 = torch.tensor(0.071)
    thy0 = torch.tensor(0.023)
    r_core = torch.tensor(0.5)
    r_s = torch.tensor(1.5)
    kappa_0 = lens.kappa_0(z_l, z_s, cosmology, torch.tensor(1.0), r_core, r_s)
    args = (z_l, z_s, cosmology, thx0, thy0, kappa_0, r_core, r_s)
    kwargs_ls = [
        {
            "sigma0": kappa_0.item(),
            "Ra": r_core.item(),
            "Rs": r_s.item(),
            "center_x": thx0.item(),
            "center_y": thy0.item(),
        }
    ]

    lens_test_helper(lens, lens_ls, args, kwargs_ls, rtol, atol)
