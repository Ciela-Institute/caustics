from math import pi

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.lenses import SIE, MultiplaneLens
from caustic.cosmology import FlatLambdaCDMCosmology

def test():
    atol = 1e-2
    rtol = 1e-2

    # Setup
    redshift_list = [0.1, 0.5]
    z_source = 1.
    cosmology = FlatLambdaCDMCosmology()
    
    # Models
    lens = MultiplaneLens()
    lenses = [SIE(), SIE()]
    lens_model_list = ["SIE", "SIE"]
    lens_ls = LensModel(
        lens_model_list=lens_model_list,
        z_source = z_source,
        lens_redshift_list = redshift_list,
        multi_plane = True,
    )

    # Parameters
    thx0 = torch.tensor(0.912)
    thy0 = torch.tensor(-0.442)
    q = torch.tensor(0.7)
    phi = torch.tensor(pi / 3)
    b = torch.tensor(1.4)
    s = torch.tensor(0.0)
    args = (
        z_source, # source redshift
        cosmology, # cosmology object
        lenses, # list of lens planes
        torch.tensor(redshift_list), # list of lens plane redshifts
        ((thx0, thy0, q, phi, b, s),(thx0, thy0, q, phi, b, s)), # list of lens plane arguments,
    )
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi.item(), q=q.item())
    kwargs_ls = [
        {
            "theta_E": b.item(),
            "e1": e1,
            "e2": e2,
            "center_x": thx0.item(),
            "center_y": thy0.item(),
        },
        {
            "theta_E": b.item(),
            "e1": e1,
            "e2": e2,
            "center_x": thx0.item(),
            "center_y": thy0.item(),
        },
    ]

    lens_test_helper(lens, lens_ls, args, kwargs_ls, rtol, atol, test_Psi = False, test_kappa = False)
