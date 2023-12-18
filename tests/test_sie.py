from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE
from caustics.utils import get_meshgrid


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIE(name="sie", cosmology=cosmology)
    lens_model_list = ["SIE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.5, 0.912, -0.442, 0.7, pi / 3, 1.4])
    e1, e2 = param_util.phi_q2_ellipticity(phi=x[4].item(), q=x[3].item())
    kwargs_ls = [
        {
            "theta_E": x[5].item(),
            "e1": e1,
            "e2": e2,
            "center_x": x[1].item(),
            "center_y": x[2].item(),
        }
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol)


def test_sie_time_delay():
    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIE(name="sie", cosmology=cosmology)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.5, 0.912, -0.442, 0.7, pi / 3, 1.4])

    n_pix = 10
    res = 0.05
    upsample_factor = 2
    thx, thy = get_meshgrid(
        res / upsample_factor,
        upsample_factor * n_pix,
        upsample_factor * n_pix,
        dtype=torch.float32,
    )

    assert torch.all(torch.isfinite(lens.time_delay(thx, thy, z_s, lens.pack(x))))
    assert torch.all(
        torch.isfinite(lens.geometric_time_delay(thx, thy, z_s, lens.pack(x)))
    )
    assert torch.all(
        torch.isfinite(
            lens.geometric_time_delay_source(thx, thy, 0.0, 0.0, z_s, lens.pack(x))
        )
    )
    assert torch.all(
        torch.isfinite(lens.shapiro_time_delay(thx, thy, z_s, lens.pack(x)))
    )


if __name__ == "__main__":
    test()
