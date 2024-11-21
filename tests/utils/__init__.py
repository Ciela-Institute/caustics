"""
Utilities for testing
"""

from typing import Any, Dict, List, Union

import numpy as np
from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM_AP
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.LensModel.lens_model import LensModel

import caustics

import pytest
import textwrap

__all__ = ("mock_from_file",)


@pytest.fixture
def sim_yaml():
    return textwrap.dedent(
        """\
    cosmology: &cosmo
        name: cosmo
        kind: FlatLambdaCDM

    lens: &lens
        name: lens
        kind: SIE
        init_kwargs:
            cosmology: *cosmo

    src: &src
        name: source
        kind: Sersic

    lnslt: &lnslt
        name: lenslight
        kind: Sersic

    simulator:
        name: minisim
        kind: LensSource
        init_kwargs:
            # Single lense
            lens: *lens
            source: *src
            lens_light: *lnslt
            pixelscale: 0.05
            pixels_x: 100
    """
    )


@pytest.fixture
def sim_obj():
    cosmology = caustics.FlatLambdaCDM()
    sie = caustics.SIE(cosmology=cosmology, name="lens")
    src = caustics.Sersic(name="source")
    lnslt = caustics.Sersic(name="lenslight")
    return caustics.LensSource(
        lens=sie, source=src, lens_light=lnslt, pixelscale=0.05, pixels_x=100
    )


def get_default_cosmologies(device=None):
    cosmology = caustics.FlatLambdaCDM("cosmo")
    cosmology_ap = FlatLambdaCDM_AP(100 * cosmology.h0.value, cosmology.Om0.value, Tcmb0=0)

    if device is not None:
        cosmology = cosmology.to(device=device)
    return cosmology, cosmology_ap


def setup_grids(res=0.05, n_pix=100, device=None):
    # Caustics setup
    thx, thy = caustics.utils.meshgrid(res, n_pix, device=device)
    if device is not None:
        thx = thx.to(device=device)
        thy = thy.to(device=device)

    # Lenstronomy setup
    fov = res * n_pix
    ra_at_xy_0, dec_at_xy_0 = ((-fov + res) / 2, (-fov + res) / 2)
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


def alpha_test_helper(lens, lens_ls, z_s, x, kwargs_ls, atol, rtol, device=None):
    thx, thy, thx_ls, thy_ls = setup_grids(device=device)
    alpha_x, alpha_y = lens.reduced_deflection_angle(thx, thy, z_s, x)
    alpha_x_ls, alpha_y_ls = lens_ls.alpha(thx_ls, thy_ls, kwargs_ls)
    assert np.allclose(alpha_x.cpu().numpy(), alpha_x_ls, rtol, atol)
    assert np.allclose(alpha_y.cpu().numpy(), alpha_y_ls, rtol, atol)


def Psi_test_helper(lens, lens_ls, z_s, x, kwargs_ls, atol, rtol, device=None):
    thx, thy, thx_ls, thy_ls = setup_grids(device=device)
    Psi = lens.potential(thx, thy, z_s, x)
    Psi_ls = lens_ls.potential(thx_ls, thy_ls, kwargs_ls)
    # Potential is only defined up to a constant
    Psi -= Psi.min()
    Psi_ls -= Psi_ls.min()
    assert np.allclose(Psi.cpu().numpy(), Psi_ls, rtol, atol)


def kappa_test_helper(lens, lens_ls, z_s, x, kwargs_ls, atol, rtol, device=None):
    thx, thy, thx_ls, thy_ls = setup_grids(device=device)
    kappa = lens.convergence(thx, thy, z_s, x)
    kappa_ls = lens_ls.kappa(thx_ls, thy_ls, kwargs_ls)
    assert np.allclose(kappa.cpu().numpy(), kappa_ls, rtol, atol)


def shear_test_helper(
    lens, lens_ls, z_s, x, kwargs_ls, atol, rtol, just_egregious=False, device=None
):
    thx, thy, thx_ls, thy_ls = setup_grids(device=device, n_pix=1000 if just_egregious else 100)
    gamma1, gamma2 = lens.shear(
        thx,
        thy,
        z_s,
        x,
        method="finitediff" if just_egregious else "autograd",
        pixelscale=thx[0][1] - thx[0][0],
    )
    gamma1_ls, gamma2_ls = lens_ls.gamma(thx_ls, thy_ls, kwargs_ls)
    if just_egregious:  # only for NFW and TNFW, this needs more attention
        print(np.sum(np.abs(np.log10(np.abs(1 - gamma1.cpu().numpy() / gamma1_ls))) < 1))
        assert np.sum(np.abs(np.log10(np.abs(1 - gamma1.cpu().numpy() / gamma1_ls))) < 1) < 100000
        assert np.sum(np.abs(np.log10(np.abs(1 - gamma2.cpu().numpy() / gamma2_ls))) < 1) < 100000
    else:
        assert np.allclose(gamma1.cpu().numpy(), gamma1_ls, rtol, atol)
        assert np.allclose(gamma2.cpu().numpy(), gamma2_ls, rtol, atol)


def lens_test_helper(
    lens: Union[caustics.ThinLens, caustics.ThickLens],
    lens_ls: LensModel,
    z_s,
    x,
    kwargs_ls: List[Dict[str, Any]],
    rtol,
    atol,
    test_alpha=True,
    test_Psi=True,
    test_kappa=True,
    test_shear=True,
    shear_egregious=False,
    device=None,
):
    if device is not None:
        lens = lens.to(device=device)
        z_s = z_s.to(device=device)
        x = x.to(device=device)

    if test_alpha:
        alpha_test_helper(lens, lens_ls, z_s, x, kwargs_ls, atol, rtol, device=device)

    if test_Psi:
        Psi_test_helper(lens, lens_ls, z_s, x, kwargs_ls, atol, rtol, device=device)

    if test_kappa:
        kappa_test_helper(lens, lens_ls, z_s, x, kwargs_ls, atol, rtol, device=device)

    if test_shear:
        shear_test_helper(
            lens,
            lens_ls,
            z_s,
            x,
            kwargs_ls,
            atol,
            rtol * 10,
            just_egregious=shear_egregious,
            device=device,
        )  # shear seems less precise than other measurements
