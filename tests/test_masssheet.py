import torch
import yaml

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import MassSheet
from caustics.utils import get_meshgrid


def test(sim_source, device, lens_models):
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
    x = torch.tensor([0.5, 0.0, 0.0, 0.7])

    thx, thy = get_meshgrid(0.01, 10, 10, device=device)

    lens.reduced_deflection_angle(thx, thy, z_s, x)

    lens.potential(thx, thy, z_s, x)

    lens.convergence(thx, thy, z_s, x)
