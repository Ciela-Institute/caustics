import torch
import numpy as np

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE


def test():
    z_l = torch.tensor(0.5, dtype=torch.float32)
    z_s = torch.tensor(1.5, dtype=torch.float32)

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

    # Point in the source plane
    sp_x = torch.tensor(0.2)
    sp_y = torch.tensor(0.2)

    # Points in image plane
    x, y = lens.forward_raytrace(sp_x, sp_y, z_s)

    # Raytrace to check
    bx, by = lens.raytrace(x, y, z_s)

    assert torch.all((sp_x - bx).abs() < 1e-3)
    assert torch.all((sp_y - by).abs() < 1e-3)
