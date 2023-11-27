import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIS


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIS(name="sis", cosmology=cosmology, z_l=torch.tensor(0.5))
    lens_model_list = ["SIS"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([-0.342, 0.51, 1.4])
    kwargs_ls = [
        {"center_x": x[0].item(), "center_y": x[1].item(), "theta_E": x[2].item()}
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol)


if __name__ == "__main__":
    test()
