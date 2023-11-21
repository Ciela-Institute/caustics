from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import Psi_test_helper, alpha_test_helper, kappa_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import EPL


def test_lenstronomy():
    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = EPL(name="epl", cosmology=cosmology)
    # There is also an EPL_NUMBA class lenstronomy, but it shouldn't matter much
    lens_model_list = ["EPL"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    z_s = torch.tensor(1.0)
    x = torch.tensor([0.7, 0.912, -0.442, 0.7, pi / 3, 1.4, 1.35])

    e1, e2 = param_util.phi_q2_ellipticity(phi=x[4].item(), q=x[3].item())
    theta_E = (x[5] / x[3].sqrt()).item()
    kwargs_ls = [
        {
            "theta_E": theta_E,
            "e1": e1,
            "e2": e2,
            "center_x": x[1].item(),
            "center_y": x[2].item(),
            "gamma": x[6].item() + 1,  # important: add +1
        }
    ]

    # Different tolerances for difference quantities
    alpha_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol=1e-100, atol=6e-5)
    kappa_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol=3e-5, atol=1e-100)
    Psi_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol=3e-5, atol=1e-100)


def test_special_case_sie():
    """
    Checks that the deflection field matches an SIE for `t=1`.
    """
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = EPL(name="epl", cosmology=cosmology)
    lens_model_list = ["SIE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    z_s = torch.tensor(1.9)
    x = torch.tensor([0.7, 0.912, -0.442, 0.7, pi / 3, 1.4, 1.0])
    e1, e2 = param_util.phi_q2_ellipticity(phi=x[4].item(), q=x[3].item())
    theta_E = (x[5] / x[3].sqrt()).item()
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
    alpha_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol=1e-100, atol=6e-5)
    kappa_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol=6e-5, atol=1e-100)
    Psi_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol=3e-5, atol=1e-100)


if __name__ == "__main__":
    test_lenstronomy()
    test_special_case_sie()
