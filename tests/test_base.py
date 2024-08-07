import torch
import numpy as np

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE
from caustics import test as mini_test


def test(device):
    z_l = torch.tensor(0.5, dtype=torch.float32, device=device)
    z_s = torch.tensor(1.5, dtype=torch.float32, device=device)

    # Model
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIE(
        name="sie",
        cosmology=cosmology,
        z_l=z_l,
        x0=torch.tensor(0.0),
        y0=torch.tensor(0.0),
        q=torch.tensor(0.4),
        phi=torch.tensor(np.pi / 5),
        b=torch.tensor(1.0),
    )
    # Send to device
    lens = lens.to(device)

    # Point in the source plane
    sp_x = torch.tensor(0.2, device=device)
    sp_y = torch.tensor(0.2, device=device)

    # Points in image plane
    x, y = lens.forward_raytrace(sp_x, sp_y, z_s)

    # Raytrace to check
    bx, by = lens.raytrace(x, y, z_s)

    assert torch.all((sp_x - bx).abs() < 1e-3)
    assert torch.all((sp_y - by).abs() < 1e-3)


def test_magnification(device):
    z_l = torch.tensor(0.5, dtype=torch.float32, device=device)
    z_s = torch.tensor(1.5, dtype=torch.float32, device=device)

    # Model
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIE(
        name="sie",
        cosmology=cosmology,
        z_l=z_l,
        x0=torch.tensor(0.0),
        y0=torch.tensor(0.0),
        q=torch.tensor(0.4),
        phi=torch.tensor(np.pi / 5),
        b=torch.tensor(1.0),
    )
    # Send to device
    lens = lens.to(device)

    # Point in image plane
    x = torch.tensor(0.1, device=device)
    y = torch.tensor(0.1, device=device)

    mag = lens.magnification(x, y, z_s)

    assert np.isfinite(mag.item())
    assert mag.item() > 0

    # grid in image plane
    x = torch.linspace(-0.1, 0.1, 10, device=device)
    y = torch.linspace(-0.1, 0.1, 10, device=device)
    x, y = torch.meshgrid(x, y, indexing="ij")

    mag = lens.magnification(x, y, z_s)

    assert np.all(np.isfinite(mag.detach().cpu().numpy()))
    assert np.all(mag.detach().cpu().numpy() > 0)


def test_quicktest(device):
    """
    Quick test to check that the built-in `test` module is working
    """
    mini_test(device=device)
