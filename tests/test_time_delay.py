import torch
import numpy as np
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel

import caustics

import pytest


@pytest.mark.parametrize("q", [0.5, 0.7, 0.9])
@pytest.mark.parametrize("phi", [0.0, np.pi / 3, np.pi / 2])
@pytest.mark.parametrize("bx,by", [(0.1, -0.05), (0.2, 0.1), (0.0, 0.0)])
def test_time_delay_pointsource(q, phi, bx, by):

    # configuration parameters
    bx = torch.tensor(bx)
    by = torch.tensor(by)
    z_l = torch.tensor(0.5)
    z_s = torch.tensor(1.0)

    # Define caustics lens
    cosmo = caustics.FlatLambdaCDM(name="cosmo")
    lens = caustics.SIE(cosmology=cosmo, z_l=z_l, x0=0.0, y0=0.0, q=q, phi=phi, b=1.0)
    x, y = lens.forward_raytrace(bx, by, z_s)

    # Define lenstronomy lens
    lens_model_list = ["SIE"]
    lens_ls = LensModel(
        lens_model_list=lens_model_list, z_lens=z_l.item(), z_source=z_s.item()
    )
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)
    kwargs_ls = [{"theta_E": 1.0, "e1": e1, "e2": e2, "center_x": 0.0, "center_y": 0.0}]

    # Compute time delay caustics
    tdc = lens.time_delay(x, y, z_s).detach().cpu().numpy()
    tdc = tdc - np.min(tdc)
    np.sort(tdc)

    # Compute time delay lenstronomy
    time_delays = lens_ls.arrival_time(
        x.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        kwargs_ls,
    )
    time_delays = time_delays - np.min(time_delays)
    np.sort(time_delays)

    # Compare time delays
    assert np.allclose(tdc, time_delays, atol=1e-3)
