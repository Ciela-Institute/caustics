from math import pi

import lenstronomy.Util.param_util as param_util
import numpy as np
import torch
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.LensModel.lens_model import LensModel

from caustic.lenses import SIE
from caustic.utils import get_meshgrid


def setup():
    # Caustic setup
    res = 0.05
    nx = 200
    ny = 200
    thx, thy = get_meshgrid(res, nx, ny)
    sie = SIE()

    # Lenstronomy setup
    ra_at_xy_0, dec_at_xy_0 = (-5 + res / 2, -5 + res / 2)
    transform_pix2angle = np.array([[1, 0], [0, 1]]) * res
    kwargs_pixel = {
        "nx": nx,
        "ny": ny,  # number of pixels per axis
        "ra_at_xy_0": ra_at_xy_0,
        "dec_at_xy_0": dec_at_xy_0,
        "transform_pix2angle": transform_pix2angle,
    }
    pixel_grid = PixelGrid(**kwargs_pixel)
    thx_ls, thy_ls = pixel_grid.coordinate_grid(nx, ny)
    lens_model_list = ["SIE"]
    sie_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    thx0_sie = torch.tensor(0.912)
    thy0_sie = torch.tensor(-0.442)
    q_sie = torch.tensor(0.7)
    phi_sie = torch.tensor(pi / 3)
    b_sie = torch.tensor(1.4)
    s_sie = torch.tensor(0.0)
    args = (thx0_sie, thy0_sie, q_sie, phi_sie, b_sie, s_sie)
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi_sie.item(), q=q_sie.item())
    kwargs_ls = [
        {
            "theta_E": b_sie.item(),
            "e1": e1,
            "e2": e2,
            "center_x": thx0_sie.item(),
            "center_y": thy0_sie.item(),
        }
    ]

    return sie, sie_ls, args, kwargs_ls, thx, thy, thx_ls, thy_ls


def test_alpha():
    sie, sie_ls, args, kwargs_ls, thx, thy, thx_ls, thy_ls = setup()

    # Cosmologies and redshifts don't matter since SIE is not a physical model
    alpha_x, alpha_y = sie.alpha(thx, thy, None, None, None, *args)
    alpha_x_ls, alpha_y_ls = sie_ls.alpha(thx_ls, thy_ls, kwargs_ls)

    assert np.allclose(alpha_x.numpy(), alpha_x_ls, atol=1e-5)
    assert np.allclose(alpha_y.numpy(), alpha_y_ls, atol=1e-5)


def test_Psi():
    sie, sie_ls, args, kwargs_ls, thx, thy, thx_ls, thy_ls = setup()

    # Cosmologies and redshifts don't matter since SIE is not a physical model
    Psi = sie.Psi(thx, thy, None, None, None, *args)
    Psi_ls = sie_ls.potential(thx_ls, thy_ls, kwargs_ls)

    assert np.allclose(Psi.numpy(), Psi_ls)


def test_kappa():
    sie, sie_ls, args, kwargs_ls, thx, thy, thx_ls, thy_ls = setup()

    # Cosmologies and redshifts don't matter since SIE is not a physical model
    kappa = sie.kappa(thx, thy, None, None, None, *args)
    kappa_ls = sie_ls.kappa(thx_ls, thy_ls, kwargs_ls)

    assert np.allclose(kappa.numpy(), kappa_ls, atol=1e-5)
