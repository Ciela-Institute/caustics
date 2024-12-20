from io import StringIO

# import lenstronomy.Util.param_util as param_util
import torch
from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM_AP
from astropy.cosmology import default_cosmology

# next three imports to get Rs_angle and alpha_Rs in arcsec for lenstronomy
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper, setup_grids

from caustics.cosmology import FlatLambdaCDM as CausticFlatLambdaCDM
from caustics.lenses import NFW
from caustics.sims import build_simulator

import pytest

h0_default = float(default_cosmology.get().h)
Om0_default = float(default_cosmology.get().Om0)
Ob0_default = float(default_cosmology.get().Ob0)


@pytest.mark.parametrize("m", [1e8, 1e10, 1e12])
@pytest.mark.parametrize("c", [1.0, 8.0, 20.0])
def test_nfw(sim_source, device, m, c):
    atol = 1e-5
    rtol = 3e-2
    z_l = torch.tensor(0.1)
    z_s = torch.tensor(0.5)

    if sim_source == "yaml":
        yaml_str = f"""\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: nfw
            kind: NFW
            init_kwargs:
                z_l: {float(z_l)}
                z_s: {float(z_s)}
                cosmology: *cosmology
                use_case: differentiable
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
    else:
        # Models
        cosmology = CausticFlatLambdaCDM(name="cosmo")
        lens = NFW(
            name="nfw", cosmology=cosmology, z_l=z_l, use_case="differentiable", z_s=z_s
        )
    lens_model_list = ["NFW"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    print(lens)

    # Parameters

    thx0 = 0.457
    thy0 = 0.141
    # m = 1e12
    # c = 8.0
    x = torch.tensor([thx0, thy0, m, c])

    # Lenstronomy
    cosmo = FlatLambdaCDM_AP(H0=h0_default * 100, Om0=Om0_default, Ob0=Ob0_default)
    lens_cosmo = LensCosmo(z_lens=z_l.item(), z_source=z_s.item(), cosmo=cosmo)
    Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=m, c=c)

    # lenstronomy params ['Rs', 'alpha_Rs', 'center_x', 'center_y']
    kwargs_ls = [
        {"Rs": Rs_angle, "alpha_Rs": alpha_Rs, "center_x": thx0, "center_y": thy0}
    ]

    lens_test_helper(
        lens,
        lens_ls,
        x,
        kwargs_ls,
        atol,
        rtol,
        shear_egregious=True,  # not why match is so bad
        device=device,
    )


def test_runs(sim_source, device):
    z_l = torch.tensor(0.1)
    z_s = torch.tensor(0.5)
    if sim_source == "yaml":
        yaml_str = f"""\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: nfw
            kind: NFW
            init_kwargs:
                z_l: {float(z_l)}
                z_s: {float(z_s)}
                cosmology: *cosmology
                use_case: differentiable
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
    else:
        # Models
        cosmology = CausticFlatLambdaCDM(name="cosmo")
        lens = NFW(
            name="nfw", cosmology=cosmology, z_l=z_l, use_case="differentiable", z_s=z_s
        )
    lens.to(device=device)
    # Parameters

    thx0 = 0.457
    thy0 = 0.141
    m = 1e12
    rs = 8.0
    x = torch.tensor([thx0, thy0, m, rs])

    thx, thy, thx_ls, thy_ls = setup_grids(device=device)

    Psi = lens.potential(thx, thy, x)
    assert torch.all(torch.isfinite(Psi))
    alpha = lens.reduced_deflection_angle(thx, thy, x)
    assert torch.all(torch.isfinite(alpha[0]))
    assert torch.all(torch.isfinite(alpha[1]))
    kappa = lens.convergence(thx, thy, x)
    assert torch.all(torch.isfinite(kappa))
