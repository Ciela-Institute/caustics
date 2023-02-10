from typing import Any, Dict, List, Tuple, Union

import numpy as np
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.LensModel.lens_model import LensModel

from caustic.lenses import ThinLens
from caustic.lenses.base import ThickLens
from caustic.utils import get_meshgrid


def setup_grids(res=0.05, n_pix=200):
    # Caustic setup
    thx, thy = get_meshgrid(res, n_pix, n_pix)

    # Lenstronomy setup
    ra_at_xy_0, dec_at_xy_0 = (-5 + res / 2, -5 + res / 2)
    transform_pix2angle = np.array([[1, 0], [0, 1]]) * res
    kwargs_pixel = {
        "nx": n_pix,
        "ny": n_pix,  # number of pixels per axis
        "ra_at_xy_0": ra_at_xy_0,
        "dec_at_xy_0": dec_at_xy_0,
        "transform_pix2angle": transform_pix2angle,
    }
    pixel_grid = PixelGrid(**kwargs_pixel)
    thx_ls, thy_ls = pixel_grid.coordinate_grid(n_pix, n_pix)
    return thx, thy, thx_ls, thy_ls


def alpha_test_helper(lens, lens_ls, args, kwargs_ls, atol, rtol):
    thx, thy, thx_ls, thy_ls = setup_grids()
    alpha_x, alpha_y = lens.alpha(thx, thy, *args)
    alpha_x_ls, alpha_y_ls = lens_ls.alpha(thx_ls, thy_ls, kwargs_ls)
    assert np.allclose(alpha_x.numpy(), alpha_x_ls, rtol, atol)
    assert np.allclose(alpha_y.numpy(), alpha_y_ls, rtol, atol)


def Psi_test_helper(lens, lens_ls, args, kwargs_ls, atol, rtol):
    thx, thy, thx_ls, thy_ls = setup_grids()
    Psi = lens.Psi(thx, thy, *args)
    Psi_ls = lens_ls.potential(thx_ls, thy_ls, kwargs_ls)
    # Potential is only defined up to a constant
    Psi -= Psi.min()
    Psi_ls -= Psi_ls.min()
    assert np.allclose(Psi.numpy(), Psi_ls, rtol, atol)


def kappa_test_helper(lens, lens_ls, args, kwargs_ls, atol, rtol):
    thx, thy, thx_ls, thy_ls = setup_grids()
    kappa = lens.kappa(thx, thy, *args)
    kappa_ls = lens_ls.kappa(thx_ls, thy_ls, kwargs_ls)
    assert np.allclose(kappa.numpy(), kappa_ls, rtol, atol)


def lens_test_helper(
    lens: Union[ThinLens, ThickLens],
    lens_ls: LensModel,
    args: Tuple[Any, ...],
    kwargs_ls: List[Dict[str, Any]],
    rtol,
    atol,
    test_alpha=True,
    test_Psi=True,
    test_kappa=True,
):
    if test_alpha:
        alpha_test_helper(lens, lens_ls, args, kwargs_ls, atol, rtol)

    if test_Psi:
        Psi_test_helper(lens, lens_ls, args, kwargs_ls, atol, rtol)

    if test_kappa:
        kappa_test_helper(lens, lens_ls, args, kwargs_ls, atol, rtol)
