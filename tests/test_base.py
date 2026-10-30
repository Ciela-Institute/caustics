import numpy as np
import pytest

from jax_compiles import compile_count

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import (
    SIE,
    SIS,
    BatchedPlane,
    ExternalShear,
    Multiplane,
    SinglePlane,
    func,
)
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


def _hole_centers(lens, device, **kwargs):
    """The centers of the holes in ``lens``'s closed mesh, in lexicographic ``(x, y)`` order."""
    mesh, _, _ = lens.build_closed_lens_mesh(5.0, 16, 2e-2, device=device, **kwargs)
    return backend.to_numpy(mesh.holes.centers)


def _sie_and_shear(cosmology, **kwargs):
    """A singular SIE at (0.3, -0.2) and a shear about (-0.5, 0.4), in one plane."""
    return SinglePlane(
        cosmology=cosmology,
        lenses=[
            SIE(cosmology=cosmology, x0=0.3, y0=-0.2, q=0.6, phi=0.4, Rein=1.0),
            ExternalShear(
                cosmology=cosmology, x0=-0.5, y0=0.4, gamma_1=0.05, gamma_2=-0.02
            ),
        ],
        **kwargs,
    )


def test_build_closed_lens_mesh_cuts_a_hole_at_every_component_center(device):
    lens = _sie_and_shear(FlatLambdaCDM(name="cosmo"), z_l=0.5, z_s=1.5).to(device)
    np.testing.assert_allclose(
        _hole_centers(lens, device), [[-0.5, 0.4], [0.3, -0.2]], atol=1e-6
    )


def test_build_closed_lens_mesh_adds_given_centers_to_the_lens_own(device):
    lens = _sie(device)
    centers = _hole_centers(lens, device, centers=[[1.8, -1.5]])
    np.testing.assert_allclose(centers, [[0.0, 0.0], [1.8, -1.5]], atol=1e-6)


def test_build_closed_lens_mesh_cuts_a_hole_at_every_batched_lens_center(device):
    cosmology = FlatLambdaCDM(name="cosmo")
    sis = SIS(
        cosmology=cosmology, x0=backend.as_array([0.9, -0.7, 0.2]), y0=0.4, Rein=0.3
    )
    # BatchedPlane vmaps over the batched lens's dynamic parameters.
    sis.to_dynamic()
    lens = BatchedPlane(cosmology=cosmology, lens=sis, z_l=0.5, z_s=1.5).to(device)
    centers = _hole_centers(lens, device, params=lens.get_values())
    np.testing.assert_allclose(
        centers, [[-0.7, 0.4], [0.2, 0.4], [0.9, 0.4]], atol=1e-6
    )


def test_build_closed_lens_mesh_cuts_a_hole_at_every_plane_component_center(device):
    cosmology = FlatLambdaCDM(name="cosmo")
    first = _sie_and_shear(cosmology, z_l=0.5)
    second = SIS(cosmology=cosmology, x0=1.2, y0=0.9, Rein=0.2, z_l=1.0)
    lens = Multiplane(cosmology=cosmology, lenses=[first, second], z_s=2.0)
    lens = lens.to(device)
    np.testing.assert_allclose(
        _hole_centers(lens, device), [[-0.5, 0.4], [0.3, -0.2], [1.2, 0.9]], atol=1e-6
    )


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


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_build_closed_lens_mesh_reuses_its_programs_at_a_new_source_redshift(
    device, monkeypatch
):
    """An SIE's map does not depend on z_s, so a build at a new z_s samples the
    same batch sizes: it traces the lens nowhere and compiles nothing."""
    calls = []
    kernel = func.reduced_deflection_angle_sie
    monkeypatch.setattr(
        func, "reduced_deflection_angle_sie", lambda *a: calls.append(1) or kernel(*a)
    )
    lens = _sie(device)
    lens.z_s.to_dynamic()
    first, second = (
        backend.to(np.asarray(z, dtype=np.float32), device=device) for z in (1.5, 2.0)
    )
    lens.build_closed_lens_mesh(5.0, 16, 2e-2, device=device, params=[first])
    calls.clear()
    before = compile_count()
    lens.build_closed_lens_mesh(5.0, 16, 2e-2, device=device, params=[second])
    assert calls == [] and compile_count() == before


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_forward_raytrace_reuses_its_programs_at_a_new_source_redshift(
    device, monkeypatch
):
    """The mesh serves both redshifts: an SIE's map does not depend on z_s."""
    lens = _sie(device)
    lens.z_s.to_dynamic()
    first, second = (
        backend.to(np.asarray(z, dtype=np.float32), device=device) for z in (1.5, 2.0)
    )
    mesh, _, _ = lens.build_closed_lens_mesh(
        5.0, 16, 2e-2, device=device, params=[first]
    )
    sp_x = backend.to(np.asarray([0.2]), device=device)
    sp_y = backend.to(np.asarray([0.2]), device=device)
    calls = []
    kernel = func.reduced_deflection_angle_sie
    monkeypatch.setattr(
        func, "reduced_deflection_angle_sie", lambda *a: calls.append(1) or kernel(*a)
    )
    _, _, want = lens.forward_raytrace(sp_x, sp_y, mesh=mesh, params=[first])
    calls.clear()
    before = compile_count()
    _, _, counts = lens.forward_raytrace(sp_x, sp_y, mesh=mesh, params=[second])
    assert calls == [] and compile_count() == before
    assert backend.to_numpy(counts).tolist() == backend.to_numpy(want).tolist()


def test_forward_raytrace_follows_new_rein_and_q(device):
    """Unlike z_s above, Rein and q change an SIE's map, so a build and solve
    at new values must match a fresh lens built at those values -- not
    silently keep the first build's or solve's. Runs under both backends:
    the main path (not just jax's caches) must track new parameter values."""
    sx = backend.as_array(np.asarray([0.1, -0.05], dtype=np.float32), device=device)
    sy = backend.as_array(np.asarray([0.05, 0.1], dtype=np.float32), device=device)
    lens = _sie(device)
    lens.Rein.to_dynamic()
    lens.q.to_dynamic()
    for Rein, q in ((1.0, 0.4), (1.3, 0.6)):
        lens.Rein, lens.q = Rein, q
        params = lens.get_values()
        mesh, _, _ = lens.build_closed_lens_mesh(
            5.0, 16, 2e-2, device=device, params=params
        )
        x, y, n = lens.forward_raytrace(sx, sy, mesh=mesh, params=params)

        fresh = _sie(device)
        fresh.Rein.to_dynamic()
        fresh.q.to_dynamic()
        fresh.Rein, fresh.q = Rein, q
        fresh_params = fresh.get_values()
        fresh_mesh, _, _ = fresh.build_closed_lens_mesh(
            5.0, 16, 2e-2, device=device, params=fresh_params
        )
        fx, fy, fn = fresh.forward_raytrace(
            sx, sy, mesh=fresh_mesh, params=fresh_params
        )

        assert backend.to_numpy(n).tolist() == backend.to_numpy(fn).tolist()
        np.testing.assert_allclose(
            np.sort(backend.to_numpy(x)),
            np.sort(backend.to_numpy(fx)),
            atol=1e-8,
        )
