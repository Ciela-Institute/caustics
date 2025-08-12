from io import StringIO

import torch
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel

from utils import lens_test_helper
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import ExternalShear
from caustics.sims import build_simulator

import pytest


def test(sim_source, device):
    atol = 1e-5
    rtol = 1e-5
    z_s = torch.tensor(2.0, device=device)

    if sim_source == "yaml":
        yaml_str = f"""\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: shear
            kind: ExternalShear
            init_kwargs:
                z_s: {float(z_s)}
                cosmology: *cosmology
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = ExternalShear(name="shear", cosmology=cosmology, z_s=z_s)
    lens.to(device=device)
    lens_model_list = ["SHEAR"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    x = torch.tensor([0.7, 0.12, -0.52, -0.1, 0.1], device=device)
    kwargs_ls = [
        {
            "ra_0": x[1].item(),
            "dec_0": x[2].item(),
            "gamma1": x[3].item(),
            "gamma2": x[4].item(),
        }
    ]

    lens_test_helper(
        lens, lens_ls, x, kwargs_ls, rtol, atol, test_kappa=False, device=device
    )


def test_external_shear_parametrization():

    cosmology = FlatLambdaCDM(name="cosmo")
    lens = ExternalShear(name="shear", cosmology=cosmology, gamma_1=0.1, gamma_2=0.1)

    # Check default
    assert lens.parametrization == "cartesian"

    # Check set to angular
    lens.parametrization = "angular"
    assert lens.parametrization == "angular"
    # Check setting gamma theta to get gamma1 and gamma2
    lens.gamma = 1.0
    lens.phi = np.pi / 4
    assert np.allclose(lens.gamma_1.value.item(), 0.0, atol=1e-5)
    assert np.allclose(lens.gamma_2.value.item(), 1.0, atol=1e-5)

    # Check reset to cartesian
    del lens.phi
    lens.parametrization = "cartesian"
    lens.parametrization = "angular"
    del lens.gamma
    lens.parametrization = "cartesian"
    assert lens.parametrization == "cartesian"
    assert lens.gamma_1.value is None
    assert lens.gamma_2.value is None
    assert not hasattr(lens, "gamma")
    assert not hasattr(lens, "phi")

    with pytest.raises(ValueError):
        lens.parametrization = "weird"

    # check init angular
    lens = ExternalShear(
        cosmology=cosmology, parametrization="angular", gamma=1.0, phi=np.pi / 4
    )
    assert np.allclose(lens.gamma.value.item(), 1.0, atol=1e-5)
    assert np.allclose(lens.phi.value.item(), np.pi / 4, atol=1e-5)
