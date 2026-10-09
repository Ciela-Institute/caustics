"""Total magnification read off a lens mesh."""

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive import build_lens_mesh, total_magnification
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend
from caustics.lenses.func.adaptive.geometry import contains_half_open
from caustics.lenses.func.adaptive.index import index_hits
from caustics.lenses.func.adaptive.lens_mesh import inside_fov_image
from caustics.lenses.func.adaptive.magnification import (
    LeafTables,
    _total_magnification,
    hit_magnification,
    leaf_tables,
)
from adaptive_maps import (
    AFFINE,
    affine,
    affine_jacobian,
    build,
    f64,
    i64,
    numpy_lens,
    point_jacobian,
    point_raytrace,
    sis_jacobian,
    sis_raytrace,
    to_np,
)

RNG = np.random.default_rng(20261009)


def sis_mu(b):
    """An SIS of unit Einstein radius: two images inside it, one outside."""
    return np.where(b < 1.0, 2.0 / b, 1.0 + 1.0 / b)


def point_mu(u):
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


def _mu(mesh, points):
    return to_np(_total_magnification(mesh, leaf_tables(mesh), f64(points))[0])


def _sis_user_mesh(min_img_sep=0.02):
    lens_, _ = numpy_lens(sis_raytrace, sis_jacobian)
    return build_lens_mesh(
        lens_.raytrace, lens_.jacobian_lens_equation, 6.0, 16, min_img_sep
    )


def test_an_affine_lens_has_one_over_det_a_everywhere():
    mesh, _ = build(affine, affine_jacobian, fov=4.0, init_res=4, min_img_sep=0.25)
    mu = _mu(mesh, RNG.uniform(-1.0, 1.0, (200, 2)))
    np.testing.assert_allclose(mu, 1.0 / abs(np.linalg.det(AFFINE)), rtol=1e-12)


@pytest.mark.parametrize(
    "fn, jac, want",
    [(sis_raytrace, sis_jacobian, sis_mu), (point_raytrace, point_jacobian, point_mu)],
    ids=["sis", "point"],
)
def test_mu_tot_matches_the_analytic_lens(fn, jac, want):
    """Away from the SIS pseudo-caustic and from the singular center."""
    mesh, _ = build(fn, jac, fov=6.0, init_res=16, min_img_sep=0.02)
    beta = RNG.uniform(-1.5, 1.5, (20000, 2))
    r = np.linalg.norm(beta, axis=1)
    keep = (
        to_np(inside_fov_image(mesh, f64(beta))) & (r > 0.1) & (np.abs(r - 1.0) > 0.05)
    )
    err = np.abs(_mu(mesh, beta[keep]) / want(r[keep]) - 1.0)
    assert np.median(err) < 2e-3
    assert err.max() < 5e-2


def test_hits_interpolate_det_a_where_it_keeps_its_sign_and_take_the_area_ratio_elsewhere():
    tables = LeafTables(
        area_ratio=f64([5.0, 7.0]),
        same_sign=mesh_backend.as_array([True, False]),
        size=f64([1.0, 1.0]),
    )
    w = f64([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]])  # barycentric (1/4, 1/4, 1/2)
    det = f64([[0.5, 0.5, 0.2], [0.5, -0.5, 0.2]])
    mu = to_np(hit_magnification(tables, i64([0, 1]), w, det))
    np.testing.assert_allclose(mu, [1.0 / (0.125 + 0.125 + 0.1), 7.0], rtol=1e-15)


def test_the_local_resolution_is_the_smallest_image_among_the_leaves_holding_a_point():
    mesh, _ = build(sis_raytrace, sis_jacobian, fov=6.0, init_res=16, min_img_sep=0.04)
    tables = leaf_tables(mesh)
    beta = f64(RNG.uniform(-1.5, 1.5, (300, 2)))
    _, size = _total_magnification(mesh, tables, beta)
    qidx, leaf, w = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, beta)
    held = to_np(contains_half_open(mesh.vertices_source[mesh.leaves[leaf]], w))
    want = np.full(300, np.inf)
    np.minimum.at(want, to_np(qidx)[held], to_np(tables.size[leaf])[held])
    np.testing.assert_array_equal(to_np(size), want)


def test_mu_tot_on_a_symmetry_axis_matches_mu_tot_just_off_it():
    """The images of the lens-plane axes lie on lattice lines: a point there is on shared edges."""
    mesh, _ = build(sis_raytrace, sis_jacobian, fov=6.0, init_res=16, min_img_sep=0.04)
    x = np.linspace(0.2, 1.8, 41)
    on = _mu(mesh, np.stack((x, 0.0 * x), axis=-1))
    off = _mu(mesh, np.stack((x, 0.0 * x + 1e-9), axis=-1))
    np.testing.assert_allclose(on, off, rtol=1e-6)


def test_total_magnification_is_nan_outside_the_fov_boundary_image_and_takes_any_shape():
    mesh = _sis_user_mesh()
    bx = backend.as_array([[0.3, 10.0], [0.5, -0.2]], dtype=backend.float64)
    by = backend.as_array([[0.1, 10.0], [0.0, 0.4]], dtype=backend.float64)
    mu = to_np(total_magnification(bx, by, mesh))
    assert mu.shape == (4,)
    assert np.isnan(mu[1]) and np.isfinite(mu[[0, 2, 3]]).all()


def test_total_magnification_does_not_depend_on_the_batch_size():
    mesh = _sis_user_mesh(0.04)
    beta = RNG.uniform(-1.5, 1.5, (101, 2))
    bx = backend.as_array(beta[:, 0], dtype=backend.float64)
    by = backend.as_array(beta[:, 1], dtype=backend.float64)
    whole = to_np(total_magnification(bx, by, mesh))
    for batch_size in (1, 7):
        np.testing.assert_array_equal(
            to_np(total_magnification(bx, by, mesh, batch_size=batch_size)), whole
        )


def test_no_source_gives_no_magnification():
    mesh = _sis_user_mesh(0.04)
    empty = backend.as_array(np.zeros(0), dtype=backend.float64)
    assert to_np(total_magnification(empty, empty, mesh)).shape == (0,)
