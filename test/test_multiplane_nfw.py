from math import pi

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import torch
from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM_ap
from astropy.cosmology import default_cosmology

from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.cosmology import FlatLambdaCDM
from caustic.lenses import NFW, Multiplane_ls

h0_default = float(default_cosmology.get().h)
Om0_default = float(default_cosmology.get().Om0)
Ob0_default = float(default_cosmology.get().Ob0)


def test():
    rtol = 0
    atol = 5e-3

    # Setup
    z_s = torch.tensor(1.5, dtype=torch.float32)
    cosmology = FlatLambdaCDM("cosmo")
    cosmology.to(dtype=torch.float32)

    # Parameters
    xs = [
        [0.4, 0.457, 0.141, 1e13, 8.0],
        [0.7, -1.0, 0.5, 1e13, 8],
        [1.1, -0.4, -1.3, 1e13, 8],
    ]
    x = torch.tensor([p for _xs in xs for p in _xs], dtype=torch.float32)

    lens = Multiplane_ls(
        "multiplane", cosmology, [NFW(f"nfw-{i}", cosmology) for i in range(len(xs))]
    )
    #lens.effective_reduced_deflection_angle = lens.raytrace

    # lenstronomy

    # Use same cosmology
    # cosmo_ap = FlatLambdaCDM_ap(cosmology.h0.value, cosmology.Om0.value, Tcmb0=0)
    cosmo_ap = FlatLambdaCDM_ap(H0=h0_default * 100, Om0=Om0_default, Ob0=Ob0_default)


    kwargs_ls = []
    for _xs in xs:
        lens_cosmo = LensCosmo(z_lens=_xs[0], z_source=z_s.item(), cosmo=cosmo_ap)
        Rs, alpha_Rs = lens_cosmo.nfw_physical2angle(M=_xs[3], c=_xs[4])
        kwargs_ls.append(
            {
                "Rs": Rs,
                "alpha_Rs": alpha_Rs,
                "center_x": _xs[1],
                "center_y": _xs[2],
            }
        )

    lens_ls = LensModel(
        lens_model_list=["NFW" for _ in range(len(xs))],
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
