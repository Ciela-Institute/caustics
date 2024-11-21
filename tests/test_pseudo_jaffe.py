from io import StringIO

import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import PseudoJaffe
from caustics.sims import build_simulator

import pytest


@pytest.mark.parametrize("mass", [1e8, 1e10, 1e12])
@pytest.mark.parametrize("Rc,Rs", [[1.0, 10.0], [1e-2, 1.0], [0.5, 1.0]])
def test_pseudo_jaffe(sim_source, device, mass, Rc, Rs):
    atol = 1e-5
    rtol = 1e-5

    if sim_source == "yaml":
        yaml_str = """\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens:
            name: pj
            kind: PseudoJaffe
            init_kwargs:
                cosmology: *cosmology
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
        cosmology = lens.cosmology
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = PseudoJaffe(name="pj", cosmology=cosmology)
    lens_model_list = ["PJAFFE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters, computing kappa_0 with a helper function
    z_s = torch.tensor(2.1)
    x = torch.tensor([0.5, 0.071, 0.023, mass, Rc, Rs])
    kappa_0 = lens.get_convergence_0(z_s, x)

    kwargs_ls = [
        {
            "sigma0": kappa_0.item(),
            "Ra": x[4].item(),
            "Rs": x[5].item(),
            "center_x": x[1].item(),
            "center_y": x[2].item(),
        }
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol, device=device)


def test_massenclosed(device):
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = PseudoJaffe(name="pj", cosmology=cosmology)
    lens.to(device=device)
    z_s = torch.tensor(2.1)
    x = torch.tensor([0.5, 0.071, 0.023, -1e100, 0.5, 1.5])
    d_l = cosmology.angular_diameter_distance(x[0])
    arcsec_to_rad = 1 / (180 / torch.pi * 60**2)
    kappa_0 = lens.central_convergence(
        x[0],
        z_s,
        torch.tensor(2e11),
        x[4] * d_l * arcsec_to_rad,
        x[5] * d_l * arcsec_to_rad,
        cosmology.critical_surface_density(x[0], z_s),
    )
    x[3] = (
        2
        * torch.pi
        * kappa_0
        * cosmology.critical_surface_density(x[0], z_s)
        * x[4]
        * x[5]
        * (d_l * arcsec_to_rad) ** 2
    )
    xx = torch.linspace(0, 10, 10, device=device)
    masses = lens.mass_enclosed_2d(xx, z_s, x)

    assert torch.all(masses < x[3])
