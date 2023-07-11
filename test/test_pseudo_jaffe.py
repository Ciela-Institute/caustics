from collections import defaultdict

import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.cosmology import FlatLambdaCDM
from caustic.lenses import PseudoJaffe


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    print(cosmology)
    lens = PseudoJaffe(name="pj", cosmology=cosmology)
    lens_model_list = ["PJAFFE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters, computing kappa_0 with a helper function
    z_s = torch.tensor(2.1)
    x = torch.tensor([0.5, 0.071, 0.023, -1e100, 0.5, 1.5])
    x[3] = kappa_0 = lens.convergence_0(
        x[0], z_s, torch.tensor(1.0), x[4], x[5], cosmology, cosmology.pack() # defaultdict(list)
    )
    kwargs_ls = [
        {
            "sigma0": kappa_0.item(),
            "Ra": x[4].item(),
            "Rs": x[5].item(),
            "center_x": x[1].item(),
            "center_y": x[2].item(),
        }
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol)


if __name__ == "__main__":
    test()
