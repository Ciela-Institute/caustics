from math import pi

import torch

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, BatchedPlane
from caustics.utils import meshgrid


def test_batchedplane():

    z_s = torch.tensor(1.2)
    z_l = torch.tensor(0.5)
    cosmology = FlatLambdaCDM(name="cosmo")
    internallens = SIE(
        name="sie",
        cosmology=cosmology,
        x0=0.912 * torch.ones(10),  # batched
        y0=-0.442,
        q=0.5,
        phi=pi / 3,
        Rein=torch.ones(10),  # batched
    )
    internallens.to_dynamic()

    lens = BatchedPlane(
        name="lens", lens=internallens, cosmology=cosmology, z_l=z_l, z_s=z_s
    )

    n_pix = 10
    res = 0.05
    upsample_factor = 2
    thx, thy = meshgrid(
        res / upsample_factor,
        upsample_factor * n_pix,
        upsample_factor * n_pix,
        dtype=torch.float32,
    )

    x = lens.get_values()  # batched internal lens params rolled into 1d x
    ax, ay = lens.reduced_deflection_angle(thx, thy, x)
    kappa = lens.convergence(thx, thy, x)
    potential = lens.potential(thx, thy, x)

    x = internallens.get_values()  # Batched params expanded on batch dimension
    in_ax, in_ay = internallens.reduced_deflection_angle(thx, thy, x[0])
    in_kappa = internallens.convergence(thx, thy, x[0])
    in_potential = internallens.potential(thx, thy, x[0])

    assert torch.allclose(ax, 10 * in_ax)
    assert torch.allclose(ay, 10 * in_ay)
    assert torch.allclose(kappa, 10 * in_kappa)
    assert torch.allclose(potential, 10 * in_potential)
