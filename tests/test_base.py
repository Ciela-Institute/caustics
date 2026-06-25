import numpy as np

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE
from caustics import test as mini_test
from caustics.backend_obj import backend


def test_forward_raytrace(device):

    z_l = backend.as_array(0.5, dtype=backend.float32, device=device)
    z_s = backend.as_array(1.5, dtype=backend.float32, device=device)

    # Model
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIE(
        name="sie",
        cosmology=cosmology,
        z_l=z_l,
        z_s=z_s,
        x0=0.0,
        y0=0.0,
        q=0.4,
        phi=np.pi / 5,
        Rein=1.0,
        s=1e-3,
    )
    # Send to device
    lens = lens.to(device)

    # Point in the source plane
    sp_x = backend.as_array(0.2, device=device)
    sp_y = backend.as_array(0.2, device=device)

    # Points in image plane
    x, y = lens.forward_raytrace(sp_x, sp_y)

    # Raytrace to check
    bx, by = lens.raytrace(x, y)

    assert bx.shape[0] == 5
    assert backend.all(backend.abs(sp_x - bx) < 1e-3)
    assert backend.all(backend.abs(sp_y - by) < 1e-3)


def test_magnification(device):
    z_l = backend.as_array(0.5, dtype=backend.float32, device=device)
    z_s = backend.as_array(1.5, dtype=backend.float32, device=device)

    # Model
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIE(
        name="sie",
        cosmology=cosmology,
        z_l=z_l,
        z_s=z_s,
        x0=backend.as_array(0.0),
        y0=backend.as_array(0.0),
        q=backend.as_array(0.4),
        phi=backend.as_array(np.pi / 5),
        Rein=backend.as_array(1.0),
    )
    # Send to device
    lens = lens.to(device)

    # Point in image plane
    x = backend.as_array(0.1, device=device)
    y = backend.as_array(0.1, device=device)

    mag = lens.magnification(x, y)

    assert np.isfinite(mag.item())
    assert mag.item() > 0

    # grid in image plane
    x = backend.linspace(-0.1, 0.1, 10, device=device)
    y = backend.linspace(-0.1, 0.1, 10, device=device)
    x, y = backend.meshgrid(x, y, indexing="ij")

    mag = lens.magnification(x, y)

    assert np.all(np.isfinite(backend.to_numpy(mag)))
    assert np.all(backend.to_numpy(mag) > 0)


def test_quicktest(device):
    """
    Quick test to check that the built-in `test` module is working
    """
    mini_test(device=device)
