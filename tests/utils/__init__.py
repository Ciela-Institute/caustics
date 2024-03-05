"""
Utilities for testing
"""
from typing import Any, Dict, List, Union

import torch
import numpy as np
from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM_AP
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.LensModel.lens_model import LensModel

from caustics.lenses import ThinLens, EPL, NFW, ThickLens, PixelatedConvergence
from caustics.light import Sersic, Pixelated
from caustics.utils import get_meshgrid
from caustics.sims import Simulator
from caustics.cosmology import FlatLambdaCDM

def mock_from_file(mocker, yaml_str):
    # Mock the from_file function
    # this way, we don't need to use a real file
    mocker.patch(
        'caustics.models.api.from_file',
        return_value=yaml_str.encode('utf-8')
    )


def setup_simulator(cosmo_static=False, use_nfw=True, simulator_static=False, batched_params=False, device=None):
    n_pix = 20

    class Sim(Simulator):
        def __init__(self, name="simulator"):
            super().__init__(name)
            if simulator_static:
                self.add_param("z_s", 1.0)
            else:
                self.add_param("z_s", None)
            z_l = 0.5
            self.cosmo = FlatLambdaCDM(h0=0.7 if cosmo_static else None, name="cosmo")
            if use_nfw:
                self.lens = NFW(
                    self.cosmo, z_l=z_l, name="lens"
                )  # NFW  wactually depend on cosmology, so a better test for Parametrized
            else:
                self.lens = EPL(self.cosmo, z_l=z_l, name="lens")
            self.sersic = Sersic(name="source")
            self.thx, self.thy = get_meshgrid(0.04, n_pix, n_pix, device=device)
            self.n_pix = n_pix
            self.to(device=device)

        def forward(self, params):
            (z_s,) = self.unpack(params)
            alphax, alphay = self.lens.reduced_deflection_angle(
                x=self.thx, y=self.thy, z_s=z_s, params=params
            )
            bx = self.thx - alphax
            by = self.thy - alphay
            return self.sersic.brightness(bx, by, params)

    # default simulator params
    z_s = torch.tensor([1.0, 1.5])
    sim_params = [z_s]
    # default cosmo params
    h0 = torch.tensor([0.68, 0.75])
    cosmo_params = [h0]
    # default lens params
    if use_nfw:
        x0 = torch.tensor([0.0, 0.1])
        y0 = torch.tensor([0.0, 0.1])
        m = torch.tensor([1e12, 1e13])
        c = torch.tensor([10, 5])
        lens_params = [x0, y0, m, c]
    else:
        x0 = torch.tensor([0, 0.1])
        y0 = torch.tensor([0, 0.1])
        q = torch.tensor([0.9, 0.8])
        phi = torch.tensor([-0.56, 0.8])
        b = torch.tensor([1.5, 1.2])
        t = torch.tensor([1.2, 1.0])
        lens_params = [x0, y0, q, phi, b, t]
    # default source params
    x0s = torch.tensor([0, 0.1])
    y0s = torch.tensor([0, 0.1])
    qs = torch.tensor([0.9, 0.8])
    phis = torch.tensor([-0.56, 0.8])
    n = torch.tensor([1.0, 4.0])
    Re = torch.tensor([0.2, 0.5])
    Ie = torch.tensor([1.2, 10.0])
    source_params = [x0s, y0s, qs, phis, n, Re, Ie]

    if not batched_params:
        sim_params = [_x[0] for _x in sim_params]
        cosmo_params = [_x[0] for _x in cosmo_params]
        lens_params = [_x[0] for _x in lens_params]
        source_params = [_x[0] for _x in source_params]
    
    sim = Sim()
    # Set device when not None
    if device is not None:
        sim = sim.to(device=device)
        sim_params = [_p.to(device=device) for _p in sim_params]
        cosmo_params = [_p.to(device=device) for _p in cosmo_params]
        lens_params = [_p.to(device=device) for _p in lens_params]
        source_params = [_p.to(device=device) for _p in source_params]
    
    return sim, (sim_params, cosmo_params, lens_params, source_params)


def setup_image_simulator(cosmo_static=False, batched_params=False, device=None):
    n_pix = 20

    class Sim(Simulator):
        def __init__(self, name="test"):
            super().__init__(name)
            pixel_scale = 0.04
            z_l = 0.5
            self.z_s = torch.tensor(1.0)
            self.cosmo = FlatLambdaCDM(h0=0.7 if cosmo_static else None, name="cosmo")
            self.epl = EPL(self.cosmo, z_l=z_l, name="lens")
            self.kappa = PixelatedConvergence(
                pixel_scale,
                n_pix,
                self.cosmo,
                z_l=z_l,
                shape=(n_pix, n_pix),
                name="kappa",
            )
            self.source = Pixelated(
                x0=0.0,
                y0=0.0,
                pixelscale=pixel_scale / 2,
                shape=(n_pix, n_pix),
                name="source",
            )
            self.thx, self.thy = get_meshgrid(pixel_scale, n_pix, n_pix, device=device)
            self.n_pix = n_pix
            self.to(device=device)

        def forward(self, params):
            alphax, alphay = self.epl.reduced_deflection_angle(
                x=self.thx, y=self.thy, z_s=self.z_s, params=params
            )
            alphax_h, alphay_h = self.kappa.reduced_deflection_angle(
                x=self.thx, y=self.thy, z_s=self.z_s, params=params
            )
            bx = self.thx - alphax - alphax_h
            by = self.thy - alphay - alphay_h
            return self.source.brightness(bx, by, params)

    # default cosmo params
    h0 = torch.tensor([0.68, 0.75])
    # default lens params
    x0 = torch.tensor([0, 0.1])
    y0 = torch.tensor([0, 0.1])
    q = torch.tensor([0.9, 0.8])
    phi = torch.tensor([-0.56, 0.8])
    b = torch.tensor([1.5, 1.2])
    t = torch.tensor([1.2, 1.0])
    # default kappa params
    kappa = torch.randn([2, n_pix, n_pix])
    source = torch.randn([2, n_pix, n_pix])

    cosmo_params = [h0]
    lens_params = [x0, y0, q, phi, b, t]
    if not batched_params:
        cosmo_params = [_x[0] for _x in cosmo_params]
        lens_params = [_x[0] for _x in lens_params]
        kappa = kappa[0]
        source = source[0]
    
    sim = Sim()
    # Set device when not None
    if device is not None:
        sim = sim.to(device=device)
        cosmo_params = [_p.to(device=device) for _p in cosmo_params]
        lens_params = [_p.to(device=device) for _p in lens_params]
        kappa = kappa.to(device=device)
        source = source.to(device=device)

    return sim, (cosmo_params, lens_params, [kappa], [source])


def get_default_cosmologies(device=None):
    cosmology = FlatLambdaCDM("cosmo")
    cosmology_ap = FlatLambdaCDM_AP(100 * cosmology.h0.value, cosmology.Om0.value, Tcmb0=0)
    
    if device is not None:
        cosmology = cosmology.to(device=device)
    return cosmology, cosmology_ap


def setup_grids(res=0.05, n_pix=100, device=None):
    # Caustics setup
    thx, thy = get_meshgrid(res, n_pix, n_pix, device=device)
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


def lens_test_helper(
    lens: Union[ThinLens, ThickLens],
    lens_ls: LensModel,
    z_s,
    x,
    kwargs_ls: List[Dict[str, Any]],
    rtol,
    atol,
    test_alpha=True,
    test_Psi=True,
    test_kappa=True,
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
