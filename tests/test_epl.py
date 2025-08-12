from math import pi
from io import StringIO

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import Psi_test_helper, alpha_test_helper, kappa_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import EPL
from caustics.sims import build_simulator
from caustics.utils import meshgrid

import pytest


@pytest.mark.parametrize("q", [0.4, 0.7])
@pytest.mark.parametrize("phi", [pi / 3, -pi / 4])
@pytest.mark.parametrize("Rein", [0.1, 1.0])
@pytest.mark.parametrize("t", [0.1, 1.0, 1.9])
def test_lenstronomy_epl(sim_source, device, q, phi, Rein, t):
    z_s = torch.tensor(1.0, device=device)
    if sim_source == "yaml":
        yaml_str = f"""\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: epl
            kind: EPL
            init_kwargs:
                z_s: {float(z_s)}
                cosmology: *cosmology
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = EPL(name="epl", cosmology=cosmology, z_s=z_s)
    lens = lens.to(device=device)
    # There is also an EPL_NUMBA class lenstronomy, but it shouldn't matter much
    lens_model_list = ["EPL"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    x = torch.tensor([0.7, 0.912, -0.442, q, phi, Rein, t], device=device)

    e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)
    theta_E = Rein
    kwargs_ls = [
        {
            "theta_E": theta_E,
            "e1": e1,
            "e2": e2,
            "center_x": x[1].item(),
            "center_y": x[2].item(),
            "gamma": t + 1,  # important: add +1
        }
    ]

    # Different tolerances for difference quantities
    alpha_test_helper(
        lens, lens_ls, x, kwargs_ls, rtol=1e-100, atol=1e-3, device=device
    )
    kappa_test_helper(
        lens, lens_ls, x, kwargs_ls, rtol=3e-5, atol=1e-100, device=device
    )
    Psi_test_helper(lens, lens_ls, x, kwargs_ls, rtol=1e-3, atol=1e-100, device=device)


def test_special_case_sie(device):
    """
    Checks that the deflection field matches an SIE for `t=1`.
    """
    cosmology = FlatLambdaCDM(name="cosmo")
    z_s = torch.tensor(1.9, device=device)
    lens = EPL(name="epl", cosmology=cosmology, z_s=z_s)
    lens.to(device=device)
    lens_model_list = ["SIE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    x = torch.tensor([0.7, 0.912, -0.442, 0.7, pi / 3, 1.4, 1.0], device=device)
    e1, e2 = param_util.phi_q2_ellipticity(phi=x[4].item(), q=x[3].item())
    theta_E = x[5].item()  # (x[5] / x[3].sqrt()).item()
    kwargs_ls = [
        {
            "theta_E": theta_E,
            "e1": e1,
            "e2": e2,
            "center_x": x[1].item(),
            "center_y": x[2].item(),
        }
    ]

    # Different tolerances for difference quantities
    alpha_test_helper(
        lens, lens_ls, x, kwargs_ls, rtol=1e-100, atol=6e-5, device=device
    )
    kappa_test_helper(
        lens, lens_ls, x, kwargs_ls, rtol=6e-5, atol=1e-100, device=device
    )
    Psi_test_helper(lens, lens_ls, x, kwargs_ls, rtol=3e-5, atol=1e-100, device=device)


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 4, 18])
def test_epl_n_chunks_consistency(chunk_size, device):
    """
    Compare results of the EPL lens model for different values of chunk_size,
    ensuring they produce consistent results within a tolerance.

    Parameters:
        chunk_size:
            Size chunks for the iterative solver.
    """
    # Setup cosmology and lens model
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = EPL(
        name="epl",
        cosmology=cosmology,
        z_s=1.0,
        z_l=0.5,
        chunk_size=chunk_size,
        **EPL._null_params,
    ).to(device=device)

    # Input Parameters
    x, y = meshgrid(0.2, 10, 10, device=device)

    # Compute results for chunk_size=1 (baseline)
    lens.chunk_size = 1
    alpha_x_1, alpha_y_1 = lens.reduced_deflection_angle(x, y)
    potential_1 = lens.potential(
        x,
        y,
    )
    convergence_1 = lens.convergence(x, y)

    # Compute results for current chunk_size
    lens.chunk_size = chunk_size
    alpha_x_n, alpha_y_n = lens.reduced_deflection_angle(x, y)
    potential_n = lens.potential(x, y)
    convergence_n = lens.convergence(x, y)

    # Validate against baseline results
    assert torch.allclose(
        alpha_x_1, alpha_x_n, rtol=1e-5, atol=1e-7
    ), "alpha_x mismatch"
    assert torch.allclose(
        alpha_y_1, alpha_y_n, rtol=1e-5, atol=1e-7
    ), "alpha_y mismatch"
    assert torch.allclose(
        potential_1, potential_n, rtol=1e-5, atol=1e-7
    ), "Potential mismatch"
    assert torch.allclose(
        convergence_1, convergence_n, rtol=1e-5, atol=1e-7
    ), "Convergence mismatch"
