import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import get_default_cosmologies, lens_test_helper

from caustic.lenses import SIS


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    cosmology, cosmology_ap = get_default_cosmologies()
    lens = SIS("sis", cosmology)
    lens_model_list = ["SIS"]
    lens_ls = LensModel(lens_model_list=lens_model_list, cosmo=cosmology_ap)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.5, -0.342, 0.51, 1.4])
    kwargs_ls = [
        {"center_x": x[1].item(), "center_y": x[2].item(), "theta_E": x[3].item()}
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol)


if __name__ == "__main__":
    test()
