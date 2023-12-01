import torch

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import MassSheet
from caustics.utils import get_meshgrid


def test():
    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = MassSheet(name="sheet", cosmology=cosmology)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.5, 0.0, 0.0, 0.7])

    thx, thy = get_meshgrid(0.01, 10, 10)

    ax, ay = lens.reduced_deflection_angle(thx, thy, z_s, *x)

    lens.potential(thx, thy, z_s, *x)

    lens.convergence(thx, thy, z_s, *x)
