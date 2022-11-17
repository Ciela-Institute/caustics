import lenstronomy.Util.param_util as param_util
import numpy as np
import torch
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.LightModel.light_model import LightModel

from caustic.sources import Sersic
from caustic.utils import get_meshgrid


def test_sersic():
    # Caustic setup
    res = 0.05
    nx = 200
    ny = 200
    thx, thy = get_meshgrid(res, nx, ny)
    sersic = Sersic(use_lenstronomy_k=True)

    # Lenstronomy setup
    ra_at_xy_0, dec_at_xy_0 = (-5 + res / 2, -5 + res / 2)
    transform_pix2angle = np.array([[1, 0], [0, 1]]) * res
    kwargs_pixel = {
        "nx": nx,
        "ny": ny,  # number of pixels per axis
        "ra_at_xy_0": ra_at_xy_0,
        "dec_at_xy_0": dec_at_xy_0,
        "transform_pix2angle": transform_pix2angle,
    }
    pixel_grid = PixelGrid(**kwargs_pixel)
    source_light_model_list = ["SERSIC_ELLIPSE"]
    sersic_ls = LightModel(light_model_list=source_light_model_list)

    # Parameters
    thx0_src = torch.tensor(0.05)
    thy0_src = torch.tensor(0.01)
    phi_src = torch.tensor(0.8)
    q_src = torch.tensor(0.5)
    index_src = torch.tensor(1.5)
    th_e_src = torch.tensor(0.1)
    I_e_src = torch.tensor(100)
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi_src.item(), q=q_src.item())
    kwargs_light_source = [
        {
            "amp": I_e_src.item(),
            "R_sersic": th_e_src.item(),
            "n_sersic": index_src.item(),
            "e1": e1,
            "e2": e2,
            "center_x": thx0_src.item(),
            "center_y": thy0_src.item(),
        }
    ]

    brightness = sersic.brightness(
        thx, thy, thx0_src, thy0_src, q_src, phi_src, index_src, th_e_src, I_e_src
    )
    brightness_ls = sersic_ls.surface_brightness(
        *pixel_grid.coordinate_grid(nx, ny), kwargs_light_source
    )

    assert np.allclose(brightness.numpy(), brightness_ls)
