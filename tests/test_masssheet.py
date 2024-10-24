from io import StringIO

import torch

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import MassSheet
from caustics.utils import meshgrid
from caustics.sims import build_simulator

import pytest


@pytest.mark.parametrize("convergence", [-1.0, 0.0, 1.0])
def test_masssheet(sim_source, device, convergence):
    if sim_source == "yaml":
        yaml_str = """\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: sheet
            kind: MassSheet
            init_kwargs:
                cosmology: *cosmology
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = MassSheet(name="sheet", cosmology=cosmology)

    lens.to(device=device)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.5, 0.0, 0.0, convergence])

    thx, thy = meshgrid(0.01, 10, device=device)

    ax, ay = lens.reduced_deflection_angle(thx, thy, z_s, x)

    p = lens.potential(thx, thy, z_s, x)

    c = lens.convergence(thx, thy, z_s, x)

    assert torch.all(torch.isfinite(ax))
    assert torch.all(torch.isfinite(ay))
    assert torch.all(torch.isfinite(p))
    assert torch.all(torch.isfinite(c))
