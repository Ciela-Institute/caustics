from collections import defaultdict

import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.cosmology import FlatLambdaCDM
from caustic.lenses import PseudoJaffe


def test():
    atol = 1e-5
    rtol = 1e-5

    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = PseudoJaffe(name="pj", cosmology=cosmology)
    lens_model_list = ["PJAFFE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters, computing kappa_0 with a helper function
    z_s = torch.tensor(2.1)
    x = torch.tensor([0.5, 0.071, 0.023, -1e100, 0.5, 1.5])
    d_l = cosmology.angular_diameter_distance(x[0])
    arcsec_to_rad = 1 / (180 / torch.pi * 60 ** 2)
    kappa_0 = lens.central_convergence(
        x[0], z_s, torch.tensor(2e11), x[4] * d_l * arcsec_to_rad, x[5] * d_l * arcsec_to_rad, cosmology.critical_surface_density(x[0], z_s)
    )
    x[3] = 2 * torch.pi * kappa_0 * cosmology.critical_surface_density(x[0], z_s) * x[4] * x[5] * (d_l * arcsec_to_rad)**2
    kwargs_ls = [
        {
            "sigma0": kappa_0.item(),
            "Ra": x[4].item(),
            "Rs": x[5].item(),
            "center_x": x[1].item(),
            "center_y": x[2].item(),
        }
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol)

def test_massenclosed():
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = PseudoJaffe(name="pj", cosmology=cosmology)
    z_s = torch.tensor(2.1)
    x = torch.tensor([0.5, 0.071, 0.023, -1e100, 0.5, 1.5])
    d_l = cosmology.angular_diameter_distance(x[0])
    arcsec_to_rad = 1 / (180 / torch.pi * 60 ** 2)
    kappa_0 = lens.central_convergence(
        x[0], z_s, torch.tensor(2e11), x[4] * d_l * arcsec_to_rad, x[5] * d_l * arcsec_to_rad, cosmology.critical_surface_density(x[0], z_s)
    )
    x[3] = 2 * torch.pi * kappa_0 * cosmology.critical_surface_density(x[0], z_s) * x[4] * x[5] * (d_l * arcsec_to_rad)**2
    xx = torch.linspace(0,10,10)
    masses = lens.mass_enclosed_2d(xx, z_s, lens.pack(x))

    assert torch.all(masses < x[3])

    
if __name__ == "__main__":
    test()
