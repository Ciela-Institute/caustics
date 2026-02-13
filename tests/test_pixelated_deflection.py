import caustics
from caustics.backend_obj import backend

import pytest


def test_pixelated_deflection():

    n = 65

    # deflect all rays to the center
    dX, dY = backend.meshgrid(
        -backend.linspace(1, -1, n), -backend.linspace(1, -1, n), indexing="xy"
    )

    cosmology = caustics.FlatLambdaCDM(name="cosmo")
    model = caustics.PixelatedDeflection(
        deflection_map=backend.stack((dX, dY)),
        pixelscale=2 / (n - 1),
        cosmology=cosmology,
        x0=0.0,
        y0=0.0,
        z_s=1.0,
        z_l=0.5,
    )

    X, Y = backend.meshgrid(
        backend.linspace(-0.9, 0.9, 32), backend.linspace(-0.9, 0.9, 32), indexing="xy"
    )

    bx, by = model.raytrace(X, Y)

    assert backend.all(backend.isfinite(bx))
    assert backend.allclose(bx, backend.zeros_like(bx), atol=1e-5)
    assert backend.all(backend.isfinite(by))
    assert backend.allclose(by, backend.zeros_like(by), atol=1e-5)

    with pytest.raises(NotImplementedError):
        model.potential(X, Y)

    with pytest.raises(NotImplementedError):
        model.convergence(X, Y)

    with pytest.raises(ValueError):
        caustics.PixelatedDeflection(
            cosmology=cosmology,
            pixelscale=2 / (n - 1),
            deflection_map=backend.ones((n, n)),
        )

    with pytest.raises(ValueError):
        caustics.PixelatedDeflection(
            cosmology=cosmology,
            pixelscale=2 / (n - 1),
            deflection_map=None,
            shape=(n, n),
        )
