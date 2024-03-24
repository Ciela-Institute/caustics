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
    x = torch.tensor([-0.52, -0.1, 0.1], device=device)
    kwargs_ls = [
        {
            "gamma1": x[1].item(),
            "gamma2": x[2].item(),
        }
    ]

    lens_test_helper(
        lens, lens_ls, z_s, x, kwargs_ls, rtol, atol, test_kappa=False, device=device
    )


def test_parametrization():
    cosmo = FlatLambdaCDM(name="cosmo")
    lens = ExternalShear(cosmology=cosmo, z_l=0.5, parametrization="cartesian")
    lens_polar = ExternalShear(cosmology=cosmo, z_l=0.5, parametrization="polar")

    gamma_1 = torch.tensor(0.1)
    gamma_2 = torch.tensor(0.2)
    gamma = torch.sqrt(gamma_1**2 + gamma_2**2)
    phi = 0.5 * torch.atan2(gamma_2, gamma_1)

    # Check that the conversion yields the same results in the deflection angle
    x = torch.tensor([0.1, 0.2])
    y = torch.tensor([0.2, 0.1])
    z_s = torch.tensor(2.0)

    a1, a2 = lens.reduced_deflection_angle(x, y, z_s, gamma_1=gamma_1, gamma_2=gamma_2)
    a1_p, a2_p = lens_polar.reduced_deflection_angle(x, y, z_s, gamma=gamma, phi=phi)
    assert torch.allclose(a1, a1_p)
    assert torch.allclose(a2, a2_p)


if __name__ == "__main__":
    test(None)
