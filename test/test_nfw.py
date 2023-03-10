import torch
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from utils import get_default_cosmologies, lens_test_helper

from caustic.lenses import NFW


def test():
    atol = 1e-5
    rtol = 3e-2

    # Models
    cosmology, cosmology_ap = get_default_cosmologies()
    z_l = torch.tensor(0.1)
    lens = NFW("nfw", cosmology, z_l=z_l)
    lens_ls = LensModel(lens_model_list=["NFW"], cosmo=cosmology_ap)

    # Parameters
    z_s = torch.tensor(0.5)

    thx0 = 0.457
    thy0 = 0.141
    m = 1e12
    c = 8.0
    x = torch.tensor([thx0, thy0, m, c])

    # Lenstronomy
    lens_cosmo = LensCosmo(z_lens=z_l.item(), z_source=z_s.item(), cosmo=cosmology_ap)
    Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=m, c=c)
    kwargs_ls = [
        {"Rs": Rs_angle, "alpha_Rs": alpha_Rs, "center_x": thx0, "center_y": thy0}
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, atol, rtol)


if __name__ == "__main__":
    test()
