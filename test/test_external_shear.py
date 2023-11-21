import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import ExternalShear


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = ExternalShear(name="shear", cosmology=cosmology)
    lens_model_list = ["SHEAR"]
    lens_ls = LensModel(lens_model_list=lens_model_list)
    print(lens)

    # Parameters
    z_s = torch.tensor(2.0)
    x = torch.tensor([0.7, 0.12, -0.52, -0.1, 0.1])
    kwargs_ls = [
        {
            "ra_0": x[1].item(),
            "dec_0": x[2].item(),
            "gamma1": x[3].item(),
            "gamma2": x[4].item(),
        }
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol, test_kappa=False)


if __name__ == "__main__":
    test()
