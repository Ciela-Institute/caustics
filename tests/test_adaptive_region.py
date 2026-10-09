"""Regions between two total magnifications: the source mesh, its contours and area, the closed build."""

import math

import numpy as np
import pytest

from caustics.lenses.func.adaptive import (
    build_lens_mesh,
)
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend, to_mesh
from caustics.lenses.func.adaptive.geometry import ROOT_SHAPES
from caustics.lenses.func.adaptive.lattice import (
    lattice_fov,
    lattice_h0,
    lattice_init_res,
    lattice_xy,
    make_lattice,
    midpoint_ij,
)
from caustics.lenses.func.adaptive.curves import (
    _critical_curves_and_caustics,
    pseudo_caustics,
)
from caustics.lenses.func.adaptive.lens_mesh import (
    _fov_boundary,
    inside_fov_image,
)
from caustics.lenses.func.adaptive.magnification import (
    _total_magnification,
    leaf_tables,
)
from caustics.lenses.func.adaptive.region import (
    _caustic_marks,
    _cell_validity,
    _sampler,
    _source_lattice,
    _source_mesh,
    _source_split,
)
from adaptive_maps import (
    build,
    f64,
    i64,
    numpy_lens,
    point_jacobian,
    point_raytrace,
    sie_like,
    sie_like_jacobian,
    sis_jacobian,
    sis_raytrace,
    to_np,
)


def point_radius(mu):
    """Source radius where a unit point mass has total magnification ``mu``."""
    return math.sqrt(2.0 * mu / math.sqrt(mu * mu - 1.0) - 2.0)


@pytest.fixture(scope="module")
def point():
    """A unit point mass at fov 6: its mesh and curves, on the mesh backend."""
    mesh, _ = build(
        point_raytrace, point_jacobian, fov=6.0, init_res=16, min_img_sep=0.02
    )
    return mesh, _critical_curves_and_caustics(mesh)


@pytest.fixture(scope="module")
def sis():
    """A unit SIS at fov 6, without a hole: its mesh and curves, on the mesh backend."""
    mesh, _ = build(sis_raytrace, sis_jacobian, fov=6.0, init_res=16, min_img_sep=0.02)
    return mesh, _critical_curves_and_caustics(mesh)


# Four level-0 cells of side 1 per axis, finest level 2; cell (0, 0) is invalid.
LAT = make_lattice(4.0, 0.0, 0.0, 4, 3)


_valid = np.ones((4, 4), dtype=bool)


_valid[0, 0] = False


VALID = mesh_backend.as_array(_valid)


NO_MARKS = [i64([]), i64([])]


STRADDLE = [2.0, 2.0, 2.0, 4.0, 2.0, 2.0]


BELOW = [2.0] * 6


ABOVE = [5.0] * 6


def _cell_triangle(i, j):
    """The first level-0 triangle of cell ``(i, j)`` of ``LAT``, ``(1, 3, 2)``."""
    return i64([(np.asarray(ROOT_SHAPES[0]) + (i, j)) * (1 << LAT.level)])


def _split(ij, mu, size, marks=NO_MARKS, mu_max=None):
    """``_source_split`` at ``mu_min = 3`` on one level-0 row whose six samples share ``size``."""
    values6 = f64([np.stack((np.asarray(mu), np.full(6, size)), axis=-1)])
    split = _source_split(LAT, VALID, marks, 3.0, mu_max)
    return bool(to_np(split(ij, values6, i64([0]), i64([0])))[0])


def test_a_triangle_the_threshold_crosses_splits_while_longer_than_the_leaf_images():
    """A level-0 triangle of ``LAT`` has a longest edge of sqrt(2)."""
    assert _split(_cell_triangle(1, 1), STRADDLE, size=0.5)
    assert not _split(_cell_triangle(1, 1), STRADDLE, size=2.0)
    assert not _split(_cell_triangle(1, 1), BELOW, size=0.5)


def test_a_triangle_in_an_invalid_cell_never_splits():
    assert not _split(_cell_triangle(0, 0), STRADDLE, size=0.5)
    assert not _split(_cell_triangle(0, 0), BELOW, size=2.0, marks=[i64([0]), i64([])])


def test_a_marked_cell_splits_unless_every_sample_exceeds_the_upper_threshold():
    marks = [i64([1 * 4 + 1]), i64([])]  # cell (1, 1) at level 0
    assert _split(_cell_triangle(1, 1), BELOW, size=2.0, marks=marks)
    assert not _split(_cell_triangle(1, 1), ABOVE, size=2.0, marks=marks)
    assert _split(_cell_triangle(1, 1), ABOVE, size=2.0, marks=marks, mu_max=10.0)
    assert not _split(
        _cell_triangle(1, 1), [11.0] * 6, size=2.0, marks=marks, mu_max=10.0
    )


def test_mu_max_crossings_split_too():
    samples = [9.0, 9.0, 9.0, 11.0, 9.0, 9.0]
    assert _split(_cell_triangle(1, 1), samples, size=0.5, mu_max=10.0)
    assert not _split(_cell_triangle(1, 1), samples, size=0.5)


def test_the_source_lattice_spans_the_fov_boundary_image_down_to_the_smallest_leaf_image(
    point,
):
    mesh = point[0]
    tables = leaf_tables(mesh)
    lat, max_level = _source_lattice(mesh, tables)
    assert lattice_init_res(lat) == lattice_init_res(mesh.lattice)
    poly = to_np(mesh.vertices_source[_fov_boundary(mesh)])
    corner = to_np(lattice_xy(lat, i64([0, 0])))
    assert (poly >= corner - 1e-12).all() and (
        poly <= corner + lattice_fov(lat) + 1e-12
    ).all()
    size = to_np(tables.size)
    smallest = size[np.isfinite(size) & (size > 0)].min()
    edge = math.sqrt(2.0) * lattice_h0(lat)
    assert edge / 2**max_level <= smallest < edge / 2 ** (max_level - 1)


def test_every_point_of_a_valid_cell_lies_inside_the_fov_boundary_image(sis):
    mesh = sis[0]
    lat, _ = _source_lattice(mesh, leaf_tables(mesh))
    valid = to_np(_cell_validity(mesh, lat))
    assert 0 < valid.sum() < valid.size
    corner, h0 = to_np(lattice_xy(lat, i64([0, 0]))), lattice_h0(lat)
    t = (np.arange(6) + 0.5) / 6
    u = np.stack(np.meshgrid(t, t, indexing="ij"), axis=-1).reshape(-1, 2)
    points = corner + h0 * (np.argwhere(valid)[:, None, :] + u[None])
    assert to_np(inside_fov_image(mesh, f64(points.reshape(-1, 2)))).all()


def test_a_cell_the_fov_boundary_image_runs_through_is_invalid(sis):
    mesh = sis[0]
    lat, _ = _source_lattice(mesh, leaf_tables(mesh))
    valid = to_np(_cell_validity(mesh, lat))
    corner, h0 = to_np(lattice_xy(lat, i64([0, 0]))), lattice_h0(lat)
    poly = to_np(mesh.vertices_source[_fov_boundary(mesh)])
    cell = np.clip(np.floor((poly - corner) / h0).astype(int), 0, valid.shape[0] - 1)
    assert not valid[cell[:, 0], cell[:, 1]].any()


def test_every_caustic_segment_marks_its_cell_at_every_level_coarser_than_it():
    mesh, _ = build(sie_like, sie_like_jacobian, fov=4.0, init_res=8, min_img_sep=0.05)
    curves = _critical_curves_and_caustics(mesh)
    assert to_np(curves.closed).all()
    lat, max_level = _source_lattice(mesh, leaf_tables(mesh))
    marks = [set(to_np(m).tolist()) for m in _caustic_marks(curves, lat, max_level)]
    corner, h0, r = (
        to_np(lattice_xy(lat, i64([0, 0]))),
        lattice_h0(lat),
        lattice_init_res(lat),
    )
    source, offsets = to_np(curves.source), to_np(curves.offsets)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        p = source[lo:hi]
        q = np.roll(p, -1, axis=0)
        mid, length = 0.5 * (p + q), np.linalg.norm(q - p, axis=1)
        for level in range(max_level):
            sel = length < math.sqrt(2.0) * h0 / 2**level
            ij = np.floor((mid[sel] - corner) / (h0 / 2**level)).astype(int)
            assert set((ij[:, 0] * (r << level) + ij[:, 1]).tolist()) <= marks[level]


def test_pseudo_caustics_mark_nothing():
    lens_, _ = numpy_lens(sis_raytrace, sis_jacobian)
    mesh = to_mesh(
        build_lens_mesh(
            lens_.raytrace,
            lens_.jacobian_lens_equation,
            6.0,
            16,
            0.02,
            centers=[(0.0, 0.0)],
        )
    )
    lat, max_level = _source_lattice(mesh, leaf_tables(mesh))
    marks = _caustic_marks(pseudo_caustics(mesh.holes), lat, max_level)
    assert all(m.shape[0] == 0 for m in marks)


def test_every_source_leaf_below_the_finest_level_passes_the_split(point):
    mesh, curves = point
    src = _source_mesh(mesh, curves, 3.0, None)
    rows = i64(np.flatnonzero(to_np(src.origin_level) < src.max_level))
    ij = src.vertices_ij[src.origin_leaves[rows]]
    six = mesh_backend.concatenate((ij, midpoint_ij(ij)), dim=1)
    sample = _sampler(mesh, leaf_tables(mesh))
    values6 = sample(lattice_xy(src.lattice, six.reshape(-1, 2))).reshape(-1, 6, 2)
    marks = _caustic_marks(curves, src.lattice, src.max_level)
    split = _source_split(src.lattice, src.cell_valid, marks, 3.0, None)
    assert not to_np(
        split(ij, values6, src.origin_cls[rows], src.origin_level[rows])
    ).any()


def test_source_vertices_carry_mu_tot_and_cost_no_lens_call():
    mesh, calls = build(
        point_raytrace, point_jacobian, fov=6.0, init_res=16, min_img_sep=0.02
    )
    curves = _critical_curves_and_caustics(mesh)
    made = (len(calls.raytrace), len(calls.jacobian))
    src = _source_mesh(mesh, curves, 3.0, None)
    assert (len(calls.raytrace), len(calls.jacobian)) == made
    mu, size = _total_magnification(mesh, leaf_tables(mesh), src.vertices)
    np.testing.assert_array_equal(to_np(src.values[:, 0]), to_np(mu))
    np.testing.assert_array_equal(to_np(src.values[:, 1]), to_np(size))
