import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import Point


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = Point(name="point", cosmology=cosmology, z_l=torch.tensor(0.9))
    lens_model_list = ["POINT_MASS"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.912, -0.442, 1.1])
    kwargs_ls = [
        {"center_x": x[0].item(), "center_y": x[1].item(), "theta_E": x[2].item()}
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol)


if __name__ == "__main__":
    test()
