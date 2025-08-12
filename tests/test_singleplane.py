from math import pi

import torch

from caustics import FlatLambdaCDM, SIE, SinglePlane
from caustics.utils import meshgrid


def test_singleplane():

    z_l = 0.5
    z_s = 1.0
    cosmology = FlatLambdaCDM(name="cosmo")
    sie1 = SIE(
        name="sie1",
        cosmology=cosmology,
        z_l=z_l,
        z_s=z_s,
        x0=0.2,
        y0=0.1,
        q=0.7,
        phi=pi / 3,
        Rein=1.0,
    )
    sie2 = SIE(
        name="sie2",
        cosmology=cosmology,
        z_l=z_l,
        z_s=z_s,
        x0=-0.2,
        y0=-0.3,
        q=0.3,
        phi=pi / 2,
        Rein=1.3,
    )

    sp = SinglePlane(
        name="singleplane", cosmology=cosmology, z_l=z_l, z_s=z_s, lenses=[sie1, sie2]
    )

    thx, thy = meshgrid(0.01, 10, device=torch.device("cpu"))

    n_pix = 50
    res = 0.05
    upsample_factor = 2
    thx, thy = meshgrid(
        res / upsample_factor,
        upsample_factor * n_pix,
        upsample_factor * n_pix,
        dtype=torch.float32,
    )

    sp_ax, sp_ay = sp.reduced_deflection_angle(thx, thy)
    sp_p = sp.potential(thx, thy)
    sp_c = sp.convergence(thx, thy)

    assert torch.all(torch.isfinite(sp_ax))
    assert torch.all(torch.isfinite(sp_ay))
    assert torch.all(torch.isfinite(sp_p))
    assert torch.all(torch.isfinite(sp_c))

    check_ax, check_ay = sie1.reduced_deflection_angle(thx, thy)
    check_ax += sie2.reduced_deflection_angle(thx, thy)[0]
    check_ay += sie2.reduced_deflection_angle(thx, thy)[1]
    check_p = sie1.potential(thx, thy) + sie2.potential(thx, thy)
    check_c = sie1.convergence(thx, thy) + sie2.convergence(thx, thy)

    assert torch.allclose(sp_ax, check_ax, atol=1e-5)
    assert torch.allclose(sp_ay, check_ay, atol=1e-5)
    assert torch.allclose(sp_p, check_p, atol=1e-5)
    assert torch.allclose(sp_c, check_c, atol=1e-5)
