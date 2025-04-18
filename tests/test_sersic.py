from io import StringIO

import lenstronomy.Util.param_util as param_util
import numpy as np
import torch
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.LightModel.light_model import LightModel

from caustics.light import Sersic
from caustics.utils import meshgrid
from caustics.sims import build_simulator

import pytest


@pytest.mark.parametrize("q", [0.2, 0.7])
@pytest.mark.parametrize("n", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("th_e", [1.0, 10.0])
def test_sersic(sim_source, device, q, n, th_e):
    # Caustics setup
    res = 0.05
    nx = 200
    ny = 200
    thx, thy = meshgrid(res, nx, ny, device=device)

    if sim_source == "yaml":
        yaml_str = """\
        light:
            name: sersic
            kind: Sersic
            init_kwargs:
                use_lenstronomy_k: true
        """
        with StringIO(yaml_str) as f:
            sersic = build_simulator(f)
    else:
        sersic = Sersic(name="sersic", use_lenstronomy_k=True)
    sersic.to(device=device)
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
    thx0_src = 0.05
    thy0_src = 0.01
    phi_src = 0.0
    q_src = q
    index_src = n
    th_e_src = th_e
    I_e_src = 100
    # NOTE: in several places we use np.sqrt(q_src) in order to match
    # the definition used by lenstronomy. This only works when phi = 0.
    # any other angle will not give the same results between the
    # two codes.
    x = torch.tensor(
        [
            thx0_src * np.sqrt(q_src),
            thy0_src,
            np.sqrt(q_src),
            phi_src,
            index_src,
            th_e_src,
            I_e_src,
        ],
        device=device,
    )
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi_src, q=q_src)
    kwargs_light_source = [
        {
            "amp": I_e_src,
            "R_sersic": th_e_src,
            "n_sersic": index_src,
            "e1": e1,
            "e2": e2,
            "center_x": thx0_src,
            "center_y": thy0_src,
        }
    ]

    brightness = sersic.brightness(thx * np.sqrt(q_src), thy, x)
    x_ls, y_ls = pixel_grid.coordinate_grid(nx, ny)
    brightness_ls = sersic_ls.surface_brightness(x_ls, y_ls, kwargs_light_source)

    assert np.allclose(brightness.cpu().numpy(), brightness_ls)
