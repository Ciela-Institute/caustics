# from math import pi

# import lenstronomy.Util.param_util as param_util
import torch
from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM_AP
from astropy.cosmology import default_cosmology

# next three imports to get Rs_angle and alpha_Rs in arcsec for lenstronomy
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper, setup_grids

from caustic.cosmology import FlatLambdaCDM as CausticFlatLambdaCDM
from caustic.lenses import NFW

h0_default = float(default_cosmology.get().h)
Om0_default = float(default_cosmology.get().Om0)
Ob0_default = float(default_cosmology.get().Ob0)


def test():
    atol = 1e-5
    rtol = 3e-2

    # Models
    cosmology = CausticFlatLambdaCDM(name="cosmo")
    z_l = torch.tensor(0.1)
    lens = NFW(name="nfw", cosmology=cosmology, z_l=z_l)
    lens_model_list = ["NFW"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    print(lens)

    # Parameters
    z_s = torch.tensor(0.5)

    thx0 = 0.457
    thy0 = 0.141
    m = 1e12
    c = 8.0
    x = torch.tensor([thx0, thy0, m, c])
    
    # Lenstronomy
    cosmo = FlatLambdaCDM_AP(H0=h0_default * 100, Om0=Om0_default, Ob0=Ob0_default)
    lens_cosmo = LensCosmo(z_lens=z_l.item(), z_source=z_s.item(), cosmo=cosmo)
    Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=m, c=c)

    # lenstronomy params ['Rs', 'alpha_Rs', 'center_x', 'center_y']
    kwargs_ls = [
        {"Rs": Rs_angle, "alpha_Rs": alpha_Rs, "center_x": thx0, "center_y": thy0}
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, atol, rtol)

def test_runs():
    cosmology = CausticFlatLambdaCDM(name="cosmo")
    z_l = torch.tensor(0.1)
    lens = NFW(name="nfw", cosmology=cosmology, z_l=z_l, use_case = "differentiable")
    
    # Parameters
    z_s = torch.tensor(0.5)

    thx0 = 0.457
    thy0 = 0.141
    m = 1e12
    rs = 8.0
    x = torch.tensor([thx0, thy0, m, rs])
    
    thx, thy, thx_ls, thy_ls = setup_grids()
    
    Psi = lens.potential(thx, thy, z_s, lens.pack(x))
    assert torch.all(torch.isfinite(Psi))
    alpha = lens.reduced_deflection_angle(thx, thy, z_s, lens.pack(x))
    assert torch.all(torch.isfinite(alpha[0]))
    assert torch.all(torch.isfinite(alpha[1]))
    kappa = lens.convergence(thx, thy, z_s, lens.pack(x))
    assert torch.all(torch.isfinite(kappa))
    

if __name__ == "__main__":
    test()
