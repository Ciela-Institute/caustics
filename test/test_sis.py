from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.cosmology import FlatLambdaCDMCosmology
from caustic.lenses import SIS


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    cosmology = FlatLambdaCDMCosmology("cosmo", None)
    cosmology.nested_cosmo = FlatLambdaCDMCosmology("cosmo_2", None, None, None)
    lens = SIS("sis", cosmology, z_l=torch.tensor(0.5))
    lens_model_list = ["SIS"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    print("cosmology.nested_cosmo:")
    print(cosmology.nested_cosmo)
    print(cosmology.nested_cosmo._params)
    print(cosmology.nested_cosmo._children)
    print(cosmology.nested_cosmo._n_static)
    print(cosmology.nested_cosmo._n_dynamic)
    print(cosmology.nested_cosmo._dynamic_size)
    print()

    print("cosmology:")
    print(cosmology)
    print(cosmology._params)
    print(cosmology._children)
    print(cosmology._n_static)
    print(cosmology._n_dynamic)
    print(cosmology._dynamic_size)
    print()

    print("lens:")
    print(lens)
    print(lens._params)
    print(lens._children)
    print(lens._n_static)
    print(lens._n_dynamic)
    print(lens._dynamic_size)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([-0.342, 0.51, 1.4, 0.7])
    kwargs_ls = [
        {
            "center_x": x[0].item(),
            "center_y": x[1].item(),
            "theta_E": x[2].item()
        }
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol)


if __name__ == "__main__":
    test()
