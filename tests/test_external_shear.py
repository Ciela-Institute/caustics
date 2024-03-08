import torch
import yaml
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import ExternalShear


def test(sim_source, device, lens_models):
    atol = 1e-5
    rtol = 1e-5

    if sim_source == "yaml":
        yaml_str = """\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: shear
            kind: ExternalShear
            init_kwargs:
                cosmology: *cosmology
        """
        yaml_dict = yaml.safe_load(yaml_str.encode("utf-8"))
        mod = lens_models.get("ExternalShear")
        lens = mod(**yaml_dict["lens"]).model_obj()
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = ExternalShear(name="shear", cosmology=cosmology)
    lens.to(device=device)
    lens_model_list = ["SHEAR"]
    lens_ls = LensModel(lens_model_list=lens_model_list)
    print(lens)

    # Parameters
    z_s = torch.tensor(2.0, device=device)
    x = torch.tensor([0.7, 0.12, -0.52, -0.1, 0.1], device=device)
    kwargs_ls = [
        {
            "ra_0": x[1].item(),
            "dec_0": x[2].item(),
            "gamma1": x[3].item(),
            "gamma2": x[4].item(),
        }
    ]

    lens_test_helper(
        lens, lens_ls, z_s, x, kwargs_ls, rtol, atol, test_kappa=False, device=device
    )


if __name__ == "__main__":
    test(None)
