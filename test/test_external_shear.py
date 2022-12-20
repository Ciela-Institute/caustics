import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.lenses import ExternalShear


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    lens = ExternalShear()
    lens_model_list = ["SHEAR"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    thx0 = torch.tensor(0.912)
    thy0 = torch.tensor(-0.442)
    gamma_1 = torch.tensor(-0.1)
    gamma_2 = torch.tensor(0.1)
    args = (None, None, None, thx0, thy0, gamma_1, gamma_2)
    kwargs_ls = [
        {
            "ra_0": thx0.item(),
            "dec_0": thy0.item(),
            "gamma1": gamma_1.item(),
            "gamma2": gamma_2.item(),
        }
    ]

    lens_test_helper(lens, lens_ls, args, kwargs_ls, rtol, atol, test_kappa=False)
