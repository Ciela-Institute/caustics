import numpy as np

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, func
from caustics import test as mini_test
from caustics.backend_obj import backend


def _sie(device):
    z_l = backend.as_array(0.5, dtype=backend.float32, device=device)
    z_s = backend.as_array(1.5, dtype=backend.float32, device=device)
    lens = SIE(
        name="sie",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=z_l,
        z_s=z_s,
        x0=0.0,
        y0=0.0,
        q=0.4,
        phi=np.pi / 5,
        Rein=1.0,
        s=1e-3,
    )
    return lens.to(device)


def test_forward_raytrace(device):
    lens = _sie(device)

    # Point in the source plane
    sp_x = backend.as_array(0.2, device=device)
    sp_y = backend.as_array(0.2, device=device)

    # Points in image plane
    x, y, counts = lens.forward_raytrace(
        sp_x, sp_y, fov=5.0, init_res=32, min_img_sep=1e-2, device=device
    )

    # Raytrace to check
    bx, by = lens.raytrace(x, y)

    assert backend.to_numpy(counts).tolist() == [5]
    assert backend.all(backend.abs(sp_x - bx) < 1e-3)
    assert backend.all(backend.abs(sp_y - by) < 1e-3)


def test_build_closed_lens_mesh_traces_the_region_only_given_mu_min(device):
    lens = _sie(device)
    mesh, curves, region = lens.build_closed_lens_mesh(
        5.0, 16, 2e-2, mu_max=50.0, device=device
    )
    assert isinstance(mesh, func.LensMesh)
    assert isinstance(curves, func.CriticalCurvesAndCaustics)
    assert region is None
    _, _, region = lens.build_closed_lens_mesh(
        5.0, 16, 2e-2, mu_min=10.0, mu_max=50.0, device=device
    )
    assert isinstance(region, func.MagnificationRegion)
    assert (region.mu_min, region.mu_max) == (10.0, 50.0)
    assert np.isfinite(region.area) and region.area > 0


def test_forward_raytrace_on_a_given_mesh_builds_none(device, monkeypatch):
    lens = _sie(device)
    mesh, _, _ = lens.build_closed_lens_mesh(5.0, 16, 2e-2, device=device)

    def build(*args, **kwargs):
        raise AssertionError("forward_raytrace built a mesh though one was given")

    monkeypatch.setattr(func, "build_closed_lens_mesh", build)
    sp_x = backend.as_array(0.2, device=device)
    sp_y = backend.as_array(0.2, device=device)
    x, y, counts = lens.forward_raytrace(sp_x, sp_y, mesh=mesh)
    want_x, want_y, want_counts = func.forward_raytrace(sp_x, sp_y, lens.raytrace, mesh)
    assert backend.to_numpy(counts).tolist() == backend.to_numpy(want_counts).tolist()
    assert np.allclose(backend.to_numpy(x), backend.to_numpy(want_x))
    assert np.allclose(backend.to_numpy(y), backend.to_numpy(want_y))


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
