import torch
import yaml

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import MassSheet
from caustics.utils import meshgrid

import pytest


@pytest.mark.parametrize("convergence", [-1.0, 0.0, 1.0])
def test_masssheet(sim_source, device, lens_models, convergence):
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
        yaml_dict = yaml.safe_load(yaml_str.encode("utf-8"))
        mod = lens_models.get("MassSheet")
        lens = mod(**yaml_dict["lens"]).model_obj()
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
