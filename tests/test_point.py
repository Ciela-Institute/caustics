import torch
import yaml
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import Point

import pytest


@pytest.mark.parametrize("th_ein", [0.1, 1.0, 2.0])
def test_point_lens(sim_source, device, lens_models, th_ein):
    atol = 1e-5
    rtol = 1e-5
    z_l = torch.tensor(0.9)

    if sim_source == "yaml":
        yaml_str = f"""\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: point
            kind: Point
            params:
                z_l: {float(z_l)}
            init_kwargs:
                cosmology: *cosmology
        """
        yaml_dict = yaml.safe_load(yaml_str.encode("utf-8"))
        mod = lens_models.get("Point")
        lens = mod(**yaml_dict["lens"]).model_obj()
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = Point(name="point", cosmology=cosmology, z_l=z_l)
    lens_model_list = ["POINT_MASS"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.912, -0.442, th_ein])
    kwargs_ls = [{"center_x": x[0].item(), "center_y": x[1].item(), "theta_E": th_ein}]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol, device=device)
