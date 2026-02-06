from math import pi

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, BatchedPlane
from caustics.utils import meshgrid
from caustics.backend_obj import backend


def test_batchedplane():

    z_s = backend.as_array(1.2)
    z_l = backend.as_array(0.5)
    cosmology = FlatLambdaCDM(name="cosmo")
    internallens = SIE(name="sie", cosmology=cosmology, z_l=z_l)

    lens = BatchedPlane(
        name="lens", lens=internallens, cosmology=cosmology, z_l=z_l, z_s=z_s
    )
    x = backend.tile(
        backend.as_array([0.912, -0.442, 0.5, pi / 3, 1.0]).reshape(1, 5), (10, 1)
    )

    n_pix = 10
    res = 0.05
    upsample_factor = 2
    thx, thy = meshgrid(
        res / upsample_factor,
        upsample_factor * n_pix,
        upsample_factor * n_pix,
        dtype=backend.float32,
    )

    ax, ay = lens.reduced_deflection_angle(thx, thy, x)

    in_ax, in_ay = internallens.reduced_deflection_angle(thx, thy, x[0])

    assert backend.allclose(ax, 10 * in_ax)
    assert backend.allclose(ay, 10 * in_ay)

    kappa = lens.convergence(thx, thy, x)

    in_kappa = internallens.convergence(thx, thy, x[0])

    assert backend.allclose(kappa, 10 * in_kappa)

    potential = lens.potential(thx, thy, x)

    in_potential = internallens.potential(thx, thy, x[0])

    assert backend.allclose(potential, 10 * in_potential)
