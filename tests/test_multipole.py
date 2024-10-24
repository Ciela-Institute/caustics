from io import StringIO

import torch
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from utils import (
    alpha_test_helper,
    kappa_test_helper,
    Psi_test_helper,
)

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import Multipole
from caustics.sims import build_simulator

import pytest


@pytest.mark.parametrize("m_order", [2, 3, 4, 5, 6])
def test_multipole_lenstronomy(sim_source, device, m_order):
    atol = 1e-5
    rtol = 1e-5
    z_l = torch.tensor(0.5)

    if sim_source == "yaml":
        yaml_str = f"""\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: multipole
            kind: Multipole
            init_kwargs:
                z_l: {float(z_l)}
                cosmology: *cosmology
                m: {m_order}
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = Multipole(name="multipole", cosmology=cosmology, z_l=z_l, m=m_order)
    lens_model_list = ["MULTIPOLE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters  m, a_m ,phi_m
    z_s = torch.tensor(1.2)
    x = torch.tensor([-0.342, 0.51, 0.1, 3.14 / 4])
    kwargs_ls = [
        {
            "center_x": x[0].item(),
            "center_y": x[1].item(),
            "a_m": x[2].item(),
            "phi_m": x[3].item(),
            "m": m_order,
        }
    ]

    # Different tolerances for difference quantities
    alpha_test_helper(
        lens, lens_ls, z_s, x, kwargs_ls, rtol=rtol, atol=atol, device=device
    )
    kappa_test_helper(
        lens, lens_ls, z_s, x, kwargs_ls, rtol=rtol, atol=atol, device=device
    )
    Psi_test_helper(
        lens, lens_ls, z_s, x, kwargs_ls, rtol=rtol, atol=atol, device=device
    )


@pytest.mark.parametrize("m_order", [[2], [3, 4], [2, 6, 8]])
def test_multipole_stack(device, m_order):

    cosmology = FlatLambdaCDM(name="cosmo")
    z_l = torch.tensor(0.5)
    z_s = torch.tensor(1.2)
    lens = Multipole(
        name="multipole",
        cosmology=cosmology,
        z_l=z_l,
        x0=0,
        y0=0,
        a_m=np.ones(len(m_order)),
        phi_m=np.zeros(len(m_order)),
        m=m_order,
    )
    lens.to(device=device)

    x = torch.linspace(-1, 1, 10, device=device)
    y = torch.linspace(-1, 1, 10, device=device)
    ax, ay = lens.reduced_deflection_angle(x, y, z_s)
    assert torch.all(torch.isfinite(ax)).item(), "multipole ax is not finite"
    assert torch.all(torch.isfinite(ay)).item(), "multipole ay is not finite"

    p = lens.potential(x, y, z_s)
    assert torch.all(torch.isfinite(p)).item(), "multipole potential is not finite"

    k = lens.convergence(x, y, z_s)
    assert torch.all(torch.isfinite(k)).item(), "multipole convergence is not finite"
