from math import pi
import yaml

import torch
import numpy as np

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, Multiplane, PixelatedConvergence
from caustics.utils import meshgrid
from utils import setup_grids


def test(sim_source, device, lens_models):
    # Setup
    z_s = torch.tensor(1.5, dtype=torch.float32)

    # Parameters
    xs = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    x = torch.tensor([p for _xs in xs for p in _xs], dtype=torch.float32, device=device)

    if sim_source == "yaml":
        yaml_str = """\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        sie1: &sie1
            name: sie_1
            kind: SIE
            init_kwargs:
                cosmology: *cosmology
        sie2: &sie2
            name: sie_2
            kind: SIE
            init_kwargs:
                cosmology: *cosmology
        sie3: &sie3
            name: sie_3
            kind: SIE
            init_kwargs:
                cosmology: *cosmology

        lens: &lens
            name: multiplane
            kind: Multiplane
            init_kwargs:
                cosmology: *cosmology
                lenses:
                    - *sie1
                    - *sie2
                    - *sie3
        """
        yaml_dict = yaml.safe_load(yaml_str.encode("utf-8"))
        mod = lens_models.get("Multiplane")
        lens = mod(**yaml_dict["lens"]).model_obj()
        lens.to(dtype=torch.float32, device=device)
        cosmology = lens.cosmology
    else:
        cosmology = FlatLambdaCDM(name="cosmo")
        cosmology.to(dtype=torch.float32, device=device)
        lens = Multiplane(
            name="multiplane",
            cosmology=cosmology,
            lenses=[SIE(name=f"sie_{i}", cosmology=cosmology) for i in range(len(xs))],
        )

    # Check raytracing doesn't blow up
    thx, thy, _, _ = setup_grids(device=device)
    beta_x, beta_y = lens.raytrace(thx, thy, z_s, x)
    assert torch.isfinite(beta_x).all()
    assert torch.isfinite(beta_y).all()

    td = lens.time_delay(thx, thy, z_s, x)
    assert torch.isfinite(td).all()

    # lenstronomy is wrong, can't compare


def test_multiplane_time_delay(device):
    # Setup
    z_s = torch.tensor(1.5, dtype=torch.float32, device=device)
    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=torch.float32, device=device)

    n_pix = 10
    res = 0.05
    upsample_factor = 2
    thx, thy = meshgrid(
        res / upsample_factor,
        upsample_factor * n_pix,
        dtype=torch.float32,
        device=device,
    )

    # Parameters
    xs = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    x = torch.tensor([p for _xs in xs for p in _xs], dtype=torch.float32, device=device)

    lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=[SIE(name=f"sie_{i}", cosmology=cosmology) for i in range(len(xs))],
    )
    lens.to(device=device)

    assert torch.isfinite(lens.time_delay(thx, thy, z_s, x)).all()
    assert torch.isfinite(
        lens.shapiro_time_delay(
            thx,
            thy,
            z_s,
            x,
        )
    ).all()
    assert torch.isfinite(lens.geometric_time_delay(thx, thy, z_s, x)).all()


def test_params(device):
    z_s = 1
    n_planes = 10
    cosmology = FlatLambdaCDM()
    pixel_size = 0.04
    pixels = 16
    z = np.linspace(1e-2, 1, n_planes)
    planes = []
    for p in range(n_planes):
        lens = PixelatedConvergence(
            name=f"plane_{p}",
            pixelscale=pixel_size,
            cosmology=cosmology,
            z_l=z[p],
            x0=0.0,
            y0=0.0,
            shape=(pixels, pixels),
            padding="tile",
        )
        lens.to(device=device)
        planes.append(lens)
    multiplane_lens = Multiplane(cosmology=cosmology, lenses=planes)
    multiplane_lens.to(device=device)
    z_s = torch.tensor(z_s)
    x, y = meshgrid(pixel_size, 32, device=device)
    params = [torch.randn(pixels, pixels, device=device) for i in range(10)]

    # Test out the computation of a few quantities to make sure params are passed correctly

    # First case, params as list of tensors
    kappa_eff = multiplane_lens.effective_convergence_div(x, y, z_s, params)
    assert kappa_eff.shape == torch.Size([32, 32])
    alphax, alphay = multiplane_lens.effective_reduced_deflection_angle(
        x, y, z_s, params
    )

    # Second case, params given as a kwargs
    kappa_eff = multiplane_lens.effective_convergence_div(x, y, z_s, params=params)
    assert kappa_eff.shape == torch.Size([32, 32])
    alphax, alphay = multiplane_lens.effective_reduced_deflection_angle(
        x, y, z_s, params=params
    )

    # Test that we can pass a dictionary
    params = {
        f"plane_{p}": torch.randn(pixels, pixels, device=device)
        for p in range(n_planes)
    }

    kappa_eff = multiplane_lens.effective_convergence_div(x, y, z_s, params)
    assert kappa_eff.shape == torch.Size([32, 32])
    alphax, alphay = multiplane_lens.effective_reduced_deflection_angle(
        x, y, z_s, params
    )


if __name__ == "__main__":
    test(None)
