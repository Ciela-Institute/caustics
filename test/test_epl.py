from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import alpha_test_helper, kappa_test_helper, Psi_test_helper

from caustic.lenses import EPL


def test_lenstronomy():
    # Models
    lens = EPL()
    lens_model_list = [
        "EPL"
    ]  # there's also EPL_NUMBA in lenstronomy, but shouldn't make a difference
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    thx0 = torch.tensor(0.912)
    thy0 = torch.tensor(-0.442)
    q = torch.tensor(0.7)
    phi = torch.tensor(pi / 3)
    b = torch.tensor(1.4)
    t = torch.tensor(1.35)  # TODO: choose slope
    s = torch.tensor(0.0)
    args = (None, None, None, thx0, thy0, q, phi, b, t, s)
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi.item(), q=q.item())
    theta_E = (b / q.sqrt()).item()
    kwargs_ls = [
        {
            "theta_E": theta_E,
            "e1": e1,
            "e2": e2,
            "center_x": thx0.item(),
            "center_y": thy0.item(),
            "gamma": t.item() + 1,  # important: add +1
        }
    ]

    # Different tolerances for difference quantities
    alpha_test_helper(lens, lens_ls, args, kwargs_ls, rtol=1e-100, atol=4.5e-5)
    kappa_test_helper(lens, lens_ls, args, kwargs_ls, rtol=1.5e-5, atol=1e-100)
    Psi_test_helper(lens, lens_ls, args, kwargs_ls, rtol=2e-5, atol=1e-100)


def test_special_case_sie():
    """
    Checks that the deflection field matches an SIE for `t=1`.
    """
    lens = EPL()
    lens_model_list = ["SIE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    thx0 = torch.tensor(0.912)
    thy0 = torch.tensor(-0.442)
    q = torch.tensor(0.7)
    phi = torch.tensor(pi / 3)
    b = torch.tensor(1.4)
    t = torch.tensor(1.0)  # special case
    s = torch.tensor(0.0)
    args = (None, None, None, thx0, thy0, q, phi, b, t, s)
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi.item(), q=q.item())
    theta_E = (b / q.sqrt()).item()
    kwargs_ls = [
        {
            "theta_E": theta_E,
            "e1": e1,
            "e2": e2,
            "center_x": thx0.item(),
            "center_y": thy0.item(),
        }
    ]

    # Different tolerances for difference quantities
    alpha_test_helper(lens, lens_ls, args, kwargs_ls, rtol=1e-100, atol=4.5e-5)
    kappa_test_helper(lens, lens_ls, args, kwargs_ls, rtol=6e-5, atol=1e-100)
    Psi_test_helper(lens, lens_ls, args, kwargs_ls, rtol=2e-5, atol=1e-100)

if __name__ == "__main__":
    test_special_case_sie()
    # test_lenstronomy()
