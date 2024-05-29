import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper, alpha_test_helper, kappa_test_helper, Psi_test_helper
import yaml

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import Multipole


def test(sim_source, device, lens_models):
    atol = 1e-5
    rtol = 1e-5
    z_l = torch.tensor(0.5)

    if sim_source == "yaml":
        yaml_str = f"""\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: multipole
            kind: Multipole
            params:
                z_l: {float(z_l)}
            init_kwargs:
                cosmology: *cosmology
        """
        yaml_dict = yaml.safe_load(yaml_str.encode("utf-8"))
        mod = lens_models.get("Multipole")
        lens = mod(**yaml_dict["lens"]).model_obj()
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = Multipole(name="multipole", cosmology=cosmology, z_l=z_l)
    lens_model_list = ["MULTIPOLE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters  m, a_m ,phi_m
    z_s = torch.tensor(1.2)
    x = torch.tensor([-0.342, 0.51, 0.1, 3.14/4, 3])
    kwargs_ls = [
        {"center_x": x[0].item(), "center_y": x[1].item(), "a_m": x[2].item(), "phi_m": x[3].item(), "m": x[4].item()}
    ]

    #lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol, device=device)
    # Different tolerances for difference quantities
    alpha_test_helper(
        lens, lens_ls, z_s, x, kwargs_ls, rtol=rtol, atol=atol, device=device
    )
    #kappa_test_helper(
    #    lens, lens_ls, z_s, x, kwargs_ls, rtol=rtol, atol=atol, device=device
    #)
    Psi_test_helper(
        lens, lens_ls, z_s, x, kwargs_ls, rtol=rtol, atol=atol, device=device
    )


if __name__ == "__main__":
    test(None)
