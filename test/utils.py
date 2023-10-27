from typing import Any, Dict, List, Union

import torch
import numpy as np
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.LensModel.lens_model import LensModel

from caustic.lenses import ThinLens, EPL, NFW, ThickLens, PixelatedConvergence
from caustic.sources import Sersic, Pixelated
from caustic.utils import get_meshgrid
from caustic.sims import Simulator
from caustic.cosmology import FlatLambdaCDM

def setup_simulator(cosmo_static=False, use_nfw=True, simulator_static=False, batched_params=False):
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
                self.lens = NFW(self.cosmo, z_l=z_l, name="lens") # NFW  wactually depend on cosmology, so a better test for Parametrized
            else:
                self.lens = EPL(self.cosmo, z_l=z_l, name="lens")
            self.sersic = Sersic(name="source")
            self.thx, self.thy = get_meshgrid(0.04, n_pix, n_pix)
            self.n_pix = n_pix

        def forward(self, params):
            z_s, = self.unpack(params)
            alphax, alphay = self.lens.reduced_deflection_angle(x=self.thx, y=self.thy, z_s=z_s, params=params) 
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
        x0 = torch.tensor([0., 0.1])
        y0 = torch.tensor([0., 0.1])
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
    n = torch.tensor([1., 4.])
    Re = torch.tensor([.2, .5])
    Ie = torch.tensor([1.2, 10.])
    source_params = [x0s, y0s, qs, phis, n, Re, Ie]
   
    if not batched_params:
        sim_params = [_x[0] for _x in sim_params]
        cosmo_params = [_x[0] for _x in cosmo_params]
        lens_params = [_x[0] for _x in lens_params]
        source_params = [_x[0] for _x in source_params]
    return Sim(), (sim_params, cosmo_params, lens_params, source_params)


def setup_image_simulator(cosmo_static=False, batched_params=False):
    n_pix = 20
    class Sim(Simulator):
        def __init__(self, name="test"):
            super().__init__(name)
            pixel_scale = 0.04
            z_l = 0.5
            self.z_s = torch.tensor(1.0)
            self.cosmo = FlatLambdaCDM(h0=0.7 if cosmo_static else None, name="cosmo")
            self.epl = EPL(self.cosmo, z_l=z_l, name="lens")
            self.kappa = PixelatedConvergence(pixel_scale, n_pix, self.cosmo, z_l=z_l, shape=(n_pix, n_pix), name="kappa")
            self.source = Pixelated(x0=0., y0=0., pixelscale=pixel_scale/2, shape=(n_pix, n_pix), name="source")
            self.thx, self.thy = get_meshgrid(pixel_scale, n_pix, n_pix)
            self.n_pix = n_pix

        def forward(self, params):
            alphax, alphay = self.epl.reduced_deflection_angle(x=self.thx, y=self.thy, z_s=self.z_s, params=params) 
            alphax_h, alphay_h = self.kappa.reduced_deflection_angle(x=self.thx, y=self.thy, z_s=self.z_s, params=params)
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
    return Sim(), (cosmo_params, lens_params, [kappa], [source])


def setup_grids(res=0.05, n_pix=100):
    # Caustic setup
    thx, thy = get_meshgrid(res, n_pix, n_pix)

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


def alpha_test_helper(lens, lens_ls, z_s, x, kwargs_ls, atol, rtol):
    thx, thy, thx_ls, thy_ls = setup_grids()
    alpha_x, alpha_y = lens.reduced_deflection_angle(thx, thy, z_s, lens.pack(x))
    alpha_x_ls, alpha_y_ls = lens_ls.alpha(thx_ls, thy_ls, kwargs_ls)
    assert np.allclose(alpha_x.numpy(), alpha_x_ls, rtol, atol)
    assert np.allclose(alpha_y.numpy(), alpha_y_ls, rtol, atol)


def Psi_test_helper(lens, lens_ls, z_s, x, kwargs_ls, atol, rtol):
    thx, thy, thx_ls, thy_ls = setup_grids()
    Psi = lens.potential(thx, thy, z_s, lens.pack(x))
    Psi_ls = lens_ls.potential(thx_ls, thy_ls, kwargs_ls)
    # Potential is only defined up to a constant
    Psi -= Psi.min()
    Psi_ls -= Psi_ls.min()
    assert np.allclose(Psi.numpy(), Psi_ls, rtol, atol)


def kappa_test_helper(lens, lens_ls, z_s, x, kwargs_ls, atol, rtol):
    thx, thy, thx_ls, thy_ls = setup_grids()
    kappa = lens.convergence(thx, thy, z_s, lens.pack(x))
    kappa_ls = lens_ls.kappa(thx_ls, thy_ls, kwargs_ls)
    assert np.allclose(kappa.numpy(), kappa_ls, rtol, atol)


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
):
    if test_alpha:
        alpha_test_helper(lens, lens_ls, z_s, lens.pack(x), kwargs_ls, atol, rtol)

    if test_Psi:
        Psi_test_helper(lens, lens_ls, z_s, lens.pack(x), kwargs_ls, atol, rtol)

    if test_kappa:
        kappa_test_helper(lens, lens_ls, z_s, lens.pack(x), kwargs_ls, atol, rtol)
