from io import StringIO

import torch
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import Point
from caustics.sims import build_simulator

import pytest


@pytest.mark.parametrize("th_ein", [0.1, 1.0, 2.0])
def test_point_lens(sim_source, device, th_ein):
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
            init_kwargs:
                z_l: {float(z_l)}
                cosmology: *cosmology
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
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


def test_point_parametrization():

    cosmology = FlatLambdaCDM(name="cosmo")
    lens = Point(name="point", cosmology=cosmology, z_l=0.5)

    # Check default
    assert lens.parametrization == "Rein"

    # Check set to mass
    lens.parametrization = "mass"
    assert lens.parametrization == "mass"
    # Check setting mass and z_l, and z_s to get mass
    lens.mass = 1e10
    lens.z_s = 1.0
    assert np.allclose(lens.th_ein.value.item(), 0.1637, atol=1e-3)

    # Check reset to cartesian
    lens.parametrization = "Rein"
    assert lens.parametrization == "Rein"
    assert lens.th_ein.value is None
    assert not hasattr(lens, "mass")
    assert not hasattr(lens, "z_s")

    with pytest.raises(ValueError):
        lens.parametrization = "weird"
