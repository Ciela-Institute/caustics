from io import StringIO

import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import Point
from caustics.sims import build_simulator

import pytest


@pytest.mark.parametrize("Rein", [0.1, 1.0, 2.0])
def test_point_lens(sim_source, device, Rein):
    atol = 1e-5
    rtol = 1e-5
    z_l = torch.tensor(0.9)
    z_s = torch.tensor(1.2)

    if sim_source == "yaml":
        yaml_str = f"""\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: point
            kind: Point
            init_kwargs:
                z_l: {float(z_l)}
                z_s: {float(z_s)}
                cosmology: *cosmology
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = Point(name="point", cosmology=cosmology, z_l=z_l, z_s=z_s)
    lens_model_list = ["POINT_MASS"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    x = torch.tensor([0.912, -0.442, Rein])
    kwargs_ls = [{"center_x": x[0].item(), "center_y": x[1].item(), "theta_E": Rein}]

    lens_test_helper(lens, lens_ls, x, kwargs_ls, rtol, atol, device=device)
