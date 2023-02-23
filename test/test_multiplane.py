from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.cosmology import FlatLambdaCDMCosmology
from caustic.lenses import SIE, MultiplaneLens


def test():
    atol = 1e-2
    rtol = 1e-2

    # Setup
    z_ls = [0.5, 0.1]
    z_s = 1.0
    cosmology = FlatLambdaCDMCosmology("cosmo")

    # Models
    lens = MultiplaneLens(
        "multiplane", cosmology, [SIE("sie_1", cosmology), SIE("sie_1", cosmology)]
    )
    lens_ls = LensModel(
        lens_model_list=["SIE", "SIE"],
        z_source=z_s,
        lens_redshift_list=z_ls,
        multi_plane=True,
    )
    print(lens)

    # Parameters
    thx0 = 0.912
    thy0 = -0.442
    q = 0.7
    phi = pi / 3
    b = 1.4
    x_noz = [thx0, thy0, q, phi, b]
    x = torch.tensor([p for z_l in z_ls for p in [z_l, *x_noz]])

    e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)
    kwargs_ls = [
        {"theta_E": b, "e1": e1, "e2": e2, "center_x": thx0, "center_y": thy0},
        {"theta_E": b, "e1": e1, "e2": e2, "center_x": thx0, "center_y": thy0},
    ]

    lens_test_helper(
        lens, lens_ls, z_s, x, kwargs_ls, rtol, atol, test_Psi=False, test_kappa=False
    )


if __name__ == "__main__":
    test()
