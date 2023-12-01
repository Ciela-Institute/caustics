from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM_ap
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper
import numpy as np

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, Multiplane, PixelatedConvergence
from caustics.utils import get_meshgrid


def test():
    rtol = 0
    atol = 5e-3

    # Setup
    z_s = torch.tensor(1.5, dtype=torch.float32)
    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=torch.float32)

    # Parameters
    xs = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    x = torch.tensor([p for _xs in xs for p in _xs], dtype=torch.float32)

    lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=[SIE(name=f"sie_{i}", cosmology=cosmology) for i in range(len(xs))],
    )
    # lens.effective_reduced_deflection_angle = lens.raytrace

    # lenstronomy
    kwargs_ls = []
    for _xs in xs:
        e1, e2 = param_util.phi_q2_ellipticity(phi=_xs[4], q=_xs[3])
        kwargs_ls.append(
            {
                "theta_E": _xs[5],
                "e1": e1,
                "e2": e2,
                "center_x": _xs[1],
                "center_y": _xs[2],
            }
        )

    # Use same cosmology
    cosmo_ap = FlatLambdaCDM_ap(cosmology.h0.value, cosmology.Om0.value, Tcmb0=0)
    lens_ls = LensModel(
        lens_model_list=["SIE" for _ in range(len(xs))],
        z_source=z_s.item(),
        lens_redshift_list=[_xs[0] for _xs in xs],
        cosmo=cosmo_ap,
        multi_plane=True,
    )

    lens_test_helper(
        lens, lens_ls, z_s, x, kwargs_ls, rtol, atol, test_Psi=False, test_kappa=False
    )


def test_params():
    z_s = 1
    n_planes = 10
    cosmology = FlatLambdaCDM()
    pixel_size = 0.04
    pixels = 16
    z = np.linspace(1e-2, 1, n_planes)
    planes = []
    for p in range(n_planes):
        lens = PixelatedConvergence(
            name=f"plane_{p}",
            pixelscale=pixel_size,
            n_pix=pixels,
            cosmology=cosmology,
            z_l=z[p],
            x0=0.0,
            y0=0.0,
            shape=(pixels, pixels),
            padding="tile",
        )
        planes.append(lens)
    multiplane_lens = Multiplane(cosmology=cosmology, lenses=planes)
    z_s = torch.tensor(z_s)
    x, y = get_meshgrid(pixel_size, 32, 32)
    params = [torch.randn(pixels, pixels) for i in range(10)]

    # Test out the computation of a few quantities to make sure params are passed correctly

    # First case, params as list of tensors
    kappa_eff = multiplane_lens.effective_convergence_div(x, y, z_s, params)
    assert kappa_eff.shape == torch.Size([32, 32])
    alphax, alphay = multiplane_lens.effective_reduced_deflection_angle(
        x, y, z_s, params
    )

    # Second case, params given as a kwargs
    kappa_eff = multiplane_lens.effective_convergence_div(x, y, z_s, params=params)
    assert kappa_eff.shape == torch.Size([32, 32])
    alphax, alphay = multiplane_lens.effective_reduced_deflection_angle(
        x, y, z_s, params=params
    )

    # Test that we can pass a dictionary
    params = {f"plane_{p}": torch.randn(pixels, pixels) for p in range(n_planes)}
    kappa_eff = multiplane_lens.effective_convergence_div(x, y, z_s, params)
    assert kappa_eff.shape == torch.Size([32, 32])
    alphax, alphay = multiplane_lens.effective_reduced_deflection_angle(
        x, y, z_s, params
    )


if __name__ == "__main__":
    test()
