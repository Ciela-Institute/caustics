from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM_ap
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper, setup_grids

from caustic.cosmology import FlatLambdaCDM
from caustic.lenses import SIE, Multiplane_ls, PixelatedConvergence

def test():
    rtol = 0
    atol = 5e-3

    res = 0.05
    n_pix = 100
    fov = res * n_pix

    # Setup
    z_s = torch.tensor(1.5, dtype=torch.float32)
    cosmology = FlatLambdaCDM("cosmo")
    cosmology.to(dtype=torch.float32)

    # Parameters
    # xs = [
    #     [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
    #     [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
    #     [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    # ]
    # xs = [
    #     [0.7, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
    #     [0.5, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
    #     [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    # ]
    xs_sie = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.5, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [0.5, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    lenses_sie = [SIE(f"sie-{i}", cosmology) for i in range(len(xs_sie))]

    thx, thy, thx_ls, thy_ls = setup_grids(res, n_pix)

    x_pc = []
    lenses_pc = []
    for i in range(len(lenses_sie)):
        x_sie_i = torch.tensor(xs_sie[i], dtype=torch.float32)
        kappa = lenses_sie[i].potential(thx, thy, z_s, lenses_sie[i].pack(x_sie_i))

        x_kap = kappa.flatten()
        PC = PixelatedConvergence(f"pc-{i}", fov, n_pix, cosmology,
                                  z_l=x_sie_i[0],
                                  x0=x_sie_i[1],
                                  y0=x_sie_i[2],
                                  convergence_map_shape=(n_pix, n_pix))

        x_pc.append(x_kap)
        lenses_pc.append(PC)

    lens = Multiplane_ls(
        "multiplane", cosmology, [SIE(f"sie-{i}", cosmology) for i in range(len(xs_sie))]
    )
    #lens.effective_reduced_deflection_angle = lens.raytrace

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

if __name__ == "__main__":
    test()
