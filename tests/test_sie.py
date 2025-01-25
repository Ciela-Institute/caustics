from math import pi
from io import StringIO

import torch
import numpy as np
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE
from caustics.utils import meshgrid
from caustics.sims import build_simulator

import pytest


@pytest.mark.parametrize("q", [0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize("phi", [pi / 3, -pi / 4, pi / 6])
@pytest.mark.parametrize("Rein", [0.1, 1.0, 2.5])
def test_sie(sim_source, device, q, phi, Rein):
    atol = 1e-5
    rtol = 1e-3
    z_s = torch.tensor(1.2)

    if sim_source == "yaml":
        yaml_str = f"""\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: sie
            kind: SIE
            init_kwargs:
                z_s: {float(z_s)}
                cosmology: *cosmology
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = SIE(name="sie", cosmology=cosmology, z_s=z_s)
    lens_model_list = ["SIE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    x = torch.tensor([0.5, 0.912, -0.442, q, phi, Rein])
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)
    kwargs_ls = [
        {
            "theta_E": Rein,
            "e1": e1,
            "e2": e2,
            "center_x": x[1].item(),
            "center_y": x[2].item(),
        }
    ]

    lens_test_helper(
        lens, lens_ls, x, kwargs_ls, rtol, atol, test_shear=q < 0.99, device=device
    )


def test_sie_time_delay():
    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    z_s = torch.tensor(1.2)
    lens = SIE(name="sie", cosmology=cosmology, z_s=z_s)

    # Parameters
    x = torch.tensor([0.5, 0.912, -0.442, 0.7, pi / 3, 1.4])

    n_pix = 10
    res = 0.05
    upsample_factor = 2
    thx, thy = meshgrid(
        res / upsample_factor,
        upsample_factor * n_pix,
        upsample_factor * n_pix,
        dtype=torch.float32,
    )

    assert torch.all(torch.isfinite(lens.time_delay(thx, thy, x)))
    assert torch.all(
        torch.isfinite(
            lens.time_delay(
                thx,
                thy,
                x,
                geometric_time_delay=True,
                shapiro_time_delay=False,
            )
        )
    )
    assert torch.all(
        torch.isfinite(
            lens.time_delay(
                thx,
                thy,
                x,
                geometric_time_delay=False,
                shapiro_time_delay=True,
            )
        )
    )


def test_sie_parametrization():

    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIE(
        name="sie",
        cosmology=cosmology,
        z_l=0.5,
        z_s=1.0,
        x0=0.0,
        y0=0.0,
        q=0.5,
        phi=pi / 4,
        Rein=1.0,
    )

    # Check default
    assert lens.parametrization == "Rein"

    # Check set to angular
    lens.parametrization = "velocity_dispersion"
    assert lens.parametrization == "velocity_dispersion"
    # Check setting sigma_v to get Rein
    lens.sigma_v = 250.0
    assert np.isfinite(lens.Rein.value.item())

    # Check reset to cartesian
    lens.parametrization = "Rein"
    assert lens.parametrization == "Rein"
    assert lens.Rein.value is None
    assert not hasattr(lens, "sigma_v")

    with pytest.raises(ValueError):
        lens.parametrization = "weird"

    # check init velocity_dispersion
    lens = SIE(
        cosmology=cosmology,
        parametrization="velocity_dispersion",
        sigma_v=250.0,
    )
    assert np.allclose(lens.sigma_v.value.item(), 250, rtol=1e-5)
