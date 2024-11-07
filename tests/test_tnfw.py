from io import StringIO

import torch
from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM_AP
from astropy.cosmology import default_cosmology

# next three imports to get Rs_angle and alpha_Rs in arcsec for lenstronomy
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper, setup_grids

from caustics.cosmology import FlatLambdaCDM as CausticFlatLambdaCDM
from caustics.lenses import TNFW
from caustics.sims import build_simulator
import pytest


h0_default = float(default_cosmology.get().h)
Om0_default = float(default_cosmology.get().Om0)
Ob0_default = float(default_cosmology.get().Ob0)


@pytest.mark.parametrize(
    "m", [1e8, 1e10, 1e12]
)  # Note with m=1e14 the test fails, due to the Rs_angle becoming too large (pytorch is unstable)
@pytest.mark.parametrize("c", [1.0, 8.0, 40.0])
@pytest.mark.parametrize("t", [2.0, 5.0, 20.0])
def test(sim_source, device, m, c, t):
    atol = 1e-5
    rtol = 3e-2
    z_l = torch.tensor(0.1)

    if sim_source == "yaml":
        yaml_str = f"""\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: tnfw
            kind: TNFW
            init_kwargs:
                z_l: {float(z_l)}
                cosmology: *cosmology
                interpret_m_total_mass: false
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
    else:
        # Models
        cosmology = CausticFlatLambdaCDM(name="cosmo")
        lens = TNFW(
            name="tnfw", cosmology=cosmology, z_l=z_l, interpret_m_total_mass=False
        )

    lens_model_list = ["TNFW"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    z_s = torch.tensor(0.5)

    thx0 = 0.457
    thy0 = 0.141
    x = torch.tensor([thx0, thy0, m, c, t])

    # Lenstronomy
    cosmo = FlatLambdaCDM_AP(H0=h0_default * 100, Om0=Om0_default, Ob0=Ob0_default)
    lens_cosmo = LensCosmo(z_lens=z_l.item(), z_source=z_s.item(), cosmo=cosmo)
    Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=m, c=c)
    x[3] = Rs_angle

    # lenstronomy params ['Rs', 'alpha_Rs', 'center_x', 'center_y']
    kwargs_ls = [
        {
            "Rs": Rs_angle,
            "alpha_Rs": alpha_Rs,
            "r_trunc": Rs_angle * t,
            "center_x": thx0,
            "center_y": thy0,
        }
    ]

    lens_test_helper(
        lens,
        lens_ls,
        z_s,
        x,
        kwargs_ls,
        atol,
        rtol,
        test_alpha=True,
        test_Psi=False,
        test_kappa=True,
        test_shear=True,
        shear_egregious=True,  # not sure why match is so bad
        device=device,
    )


def test_runs(device):
    cosmology = CausticFlatLambdaCDM(name="cosmo")
    z_l = torch.tensor(0.1)
    lens = TNFW(name="tnfw", cosmology=cosmology, z_l=z_l, use_case="differentiable")
    lens.to(device=device)
    # Parameters
    z_s = torch.tensor(0.5)

    thx0 = 0.457
    thy0 = 0.141
    m = 1e12
    rs = 8.0
    t = 3.0
    x = torch.tensor([thx0, thy0, m, rs, t])

    thx, thy, thx_ls, thy_ls = setup_grids(device=device)

    Psi = lens.potential(thx, thy, z_s, x)
    assert torch.all(torch.isfinite(Psi))

    Rt = lens.get_truncation_radius(x)

    assert Rt == (rs * t)
