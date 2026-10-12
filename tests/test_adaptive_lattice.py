"""The dyadic lattice: keys, positions, sizes, growth, level-0 triangles and midpoints."""

import math
import numpy as np
import pytest

from caustics.lenses.func.adaptive.mesh_backend import mesh_backend
from caustics.lenses.func.adaptive.geometry import child_matrix_tables, shape_matrix
from caustics.lenses.func.adaptive.lattice import (
    check_lattice_keys,
    depth_floor,
    extend_lattice,
    initial_triangles,
    lattice_ij_from_key,
    lattice_key,
    lattice_xy,
    make_lattice,
    midpoint_ij,
    ring_triangles,
    warn_depth_limited,
)

from adaptive_maps import i64, to_np


def _i64(x):
    return i64(x)


def _grid(n):
    """Every point of an ``(n + 1) x (n + 1)`` lattice, shape ``((n + 1)**2, 2)``."""
    axis = np.arange(n + 1)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)


def _same_lattice(a, b):
    return (a.level, a.n, a.scale, a.origin) == (
        b.level,
        b.n,
        b.scale,
        b.origin,
    ) and np.array_equal(to_np(a.lo), to_np(b.lo))


def test_depth_floor_matches_the_size_criterion():
    assert depth_floor(5.0 / 100, 10.0) == 0
    d = depth_floor(5.0 / 100, 1e-3)
    l0 = np.sqrt(2) * 5.0 / 100
    assert l0 / 2**d <= 1e-3 < l0 / 2 ** (d - 1)


def test_lattice_key_roundtrip_and_geometry():
    lat = make_lattice(4.0, 0.0, 0.0, 4, 3)
    assert lat.n == 32
    ij_np = np.array([[0, 0], [32, 32], [7, 19]], dtype=np.int64)
    ij = mesh_backend.as_array(ij_np, dtype=mesh_backend.int64)
    key = lattice_key(lat, ij)
    assert to_np(lattice_ij_from_key(lat, key)).tolist() == ij_np.tolist()
    assert np.allclose(to_np(lattice_xy(lat, ij[0])), [-2.0, -2.0])
    assert np.allclose(to_np(lattice_xy(lat, ij[1])), [2.0, 2.0])


def test_widening_the_lattice_does_not_move_any_vertex():
    """Bit-identical coordinates, not merely close ones.

    `scale' = fov / (2n)` equals `fl(fov / n) / 2` exactly, because binary
    floating point is scale-invariant under powers of two, and `(2 * ij) *
    scale'` then rounds the same exact real as `ij * scale`. If this ever fails,
    the widened lattice has perturbed the frozen mesh's geometry and every
    downstream bit-exactness argument in the module is void.
    """
    fov, init_res, max_level = 4.0, 4, 3
    narrow = make_lattice(fov, 0.0, 0.0, init_res, max_level)
    wide = make_lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    assert wide.n == 2 * narrow.n
    assert wide.level == max_level + 1

    ij_np = np.stack(
        np.meshgrid(np.arange(narrow.n + 1), np.arange(narrow.n + 1), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    ij = mesh_backend.as_array(ij_np, dtype=mesh_backend.int64)
    assert np.array_equal(
        to_np(lattice_xy(narrow, ij)),
        to_np(lattice_xy(wide, 2 * ij)),
    )


def test_initial_triangles_tile_the_square_and_are_positively_oriented():
    """Ported from `test_adaptive_mesh.py`: geometric properties that the
    oracle-comparison test above does not check -- full square coverage and
    a consistent positive orientation -- rather than element-wise equality
    with the oracle.
    """
    _, _, _, _, root_class = child_matrix_tables()
    init_res, max_level = 4, 2
    ij, cls = initial_triangles(init_res, max_level, root_class)
    assert ij.shape == (2 * init_res**2, 3, 2)
    P = shape_matrix(mesh_backend.to(ij, dtype=mesh_backend.float64))
    area = P[..., 0, 0] * P[..., 1, 1] - P[..., 0, 1] * P[..., 1, 0]
    assert bool(mesh_backend.all(area > 0))
    step = 1 << max_level
    assert np.isclose(float(mesh_backend.sum(area)) / 2, (init_res * step) ** 2)
    assert set(to_np(cls).tolist()) == set(to_np(root_class).tolist())


def test_midpoints_are_exact_integers_and_opposite_their_vertex():
    ij = np.array([[[0, 0], [4, 0], [0, 4]]], dtype=np.int64)
    m = midpoint_ij(_i64(ij))
    assert to_np(m)[0].tolist() == [[2, 2], [0, 2], [2, 0]]


def test_midpoints_are_exact_at_max_level_on_the_widened_lattice():
    """The reason the lattice is one level finer than max_level.

    With the lattice at max_level a triangle's edges are one unit long and
    `midpoint_ij`'s floor division collapses each "midpoint" onto one of that
    edge's own endpoints. One level finer, every edge vector is even at every
    level up to and including max_level, so the midpoints are genuine lattice
    points -- and they are exactly the points with an odd coordinate, which is
    what guarantees they can never collide with a cached vertex.
    """
    _, _, _, _, root_class = child_matrix_tables()
    max_level = 3
    lat = make_lattice(4.0, 0.0, 0.0, 2, max_level + 1)
    ij, cls = initial_triangles(2, lat.level, root_class)
    # Descend to max_level by taking child C_4 (the middle child) each time.
    for _ in range(max_level):
        ij = midpoint_ij(ij)
    assert bool(mesh_backend.all(ij % 2 == 0)), "max_level vertices are even"

    mid = midpoint_ij(ij)
    # Exact: the floor division threw nothing away.
    assert bool(
        mesh_backend.all((ij[:, [1, 2, 0]] + ij[:, [2, 0, 1]]) % 2 == 0)
    ), "edge endpoint sums must be even for the midpoint to be exact"
    # Every midpoint has an odd coordinate, so it is not a vertex of any level.
    assert bool(mesh_backend.all(mesh_backend.any(mid % 2 == 1, dim=-1)))


def test_check_lattice_keys_rejects_a_lattice_too_fine_to_key():
    with pytest.raises(ValueError, match="lattice too fine"):
        check_lattice_keys(2**20, 21, "Raise min_img_sep.")


def test_a_fresh_lattice_places_points_exactly_as_before():
    lat = make_lattice(4.0, 0.3, -0.1, 3, 4)
    ij = _grid(lat.n)
    assert lat.origin == 0
    want = to_np(lat.lo) + ij.astype(np.float64) * lat.scale
    assert np.array_equal(to_np(lattice_xy(lat, i64(ij))), want)


def test_extend_lattice_keeps_every_old_position_bit_for_bit():
    """A non-dyadic center on purpose: nothing may be recomputed from a new fov."""
    lat = make_lattice(4.0, 0.3, -0.1, 3, 4)
    ext = extend_lattice(lat, 2)
    pad = 2 << lat.level
    ij = _grid(lat.n)
    assert np.array_equal(
        to_np(lattice_xy(ext, i64(ij + pad))),
        to_np(lattice_xy(lat, i64(ij))),
    )


def test_check_lattice_keys_rejects_exactly_the_lattices_int64_cannot_key():
    """``45 * 2**26 + 1`` points per axis still key in int64; ``46 * 2**26 + 1`` do not."""
    check_lattice_keys(45, 25, "unused")
    with pytest.raises(ValueError, match="lattice too fine.*Rebuild coarser"):
        check_lattice_keys(46, 25, "Rebuild coarser.")


def test_ring_triangles_are_the_level0_triangles_outside_the_central_block():
    level = 3
    root_class = child_matrix_tables()[4]
    every_ij, every_cls = initial_triangles(6, level, root_class)
    ring_ij, ring_cls = ring_triangles(6, 1, level, root_class)

    def cells(ij):
        return to_np(ij)[:, 0, :] // (1 << level)

    def inner(ij):
        c = cells(ij)
        return ((c >= 1) & (c < 5)).all(axis=1)

    def as_set(ij, cls):
        return {(tuple(t.reshape(-1)), c) for t, c in zip(to_np(ij), to_np(cls))}

    assert ring_ij.shape[0] == 2 * (6 * 6 - 4 * 4)
    assert not inner(ring_ij).any()
    ring = as_set(ring_ij, ring_cls)
    every = as_set(every_ij, every_cls)
    assert ring <= every
    assert len(every - ring) == int(inner(every_ij).sum())


def test_extend_lattice_grows_the_extent_and_keeps_key_order():
    lat = make_lattice(4.0, 0.3, -0.1, 3, 4)
    ext = extend_lattice(lat, 2)
    pad = 2 << lat.level
    assert (ext.level, ext.scale) == (lat.level, lat.scale)
    assert ext.n == lat.n + 2 * pad
    assert ext.origin == pad
    assert np.array_equal(to_np(ext.lo), to_np(lat.lo))
    ij = _grid(lat.n)
    old = to_np(lattice_key(lat, i64(ij)))
    got = to_np(lattice_key(ext, i64(ij + pad)))
    assert np.array_equal(np.argsort(old), np.argsort(got))


def test_an_extended_dyadic_lattice_is_the_fresh_lattice_of_the_larger_fov():
    """The premise of every equivalence test below.

    With a dyadic fov, center and cell size, a fresh lattice over the larger
    fov places every point exactly where the extended one does.
    """
    lat = make_lattice(4.0, 0.5, -0.25, 8, 3)
    ext = extend_lattice(lat, 2)
    fresh = make_lattice(6.0, 0.5, -0.25, 12, 3)
    assert (ext.n, ext.scale) == (fresh.n, fresh.scale)
    ij = i64(_grid(ext.n))
    assert np.array_equal(to_np(lattice_xy(ext, ij)), to_np(lattice_xy(fresh, ij)))


def test_extend_lattice_chains_and_accepts_zero():
    lat = make_lattice(4.0, 0.0, 0.0, 2, 3)
    assert _same_lattice(extend_lattice(lat, 0), lat)
    assert _same_lattice(
        extend_lattice(extend_lattice(lat, 1), 2), extend_lattice(lat, 3)
    )


def test_the_derived_lattice_sizes_match_the_build_arguments():
    lat = make_lattice(4.0, 0.25, -0.5, 8, 5)
    assert lat.init_res == 8
    assert lat.h0 == 0.5
    assert lat.fov == 4.0
    grown = extend_lattice(lat, 2)
    assert grown.init_res == 12 and grown.fov == 6.0
    assert grown.h0 == 0.5
    assert to_np(lat.corner).tolist() == [-1.75, -2.5]
    assert to_np(grown.corner).tolist() == [-2.75, -3.5]


def test_keys_round_trip_without_a_stride_field():
    lat = make_lattice(4.0, 0.0, 0.0, 3, 2)
    ij = i64([[0, 0], [lat.n, lat.n], [5, 7]])
    assert np.array_equal(
        to_np(lattice_ij_from_key(lat, lattice_key(lat, ij))), to_np(ij)
    )
    assert to_np(lattice_key(lat, ij)).tolist()[2] == 5 * (lat.n + 1) + 7


def test_depth_floor_reads_the_level0_hypotenuse():
    assert depth_floor(1.0, 2.0) == 0
    assert depth_floor(1.0, math.sqrt(2.0) / 4) == 2


def test_warn_depth_limited_names_the_request_and_the_needed_depth():
    with pytest.warns(
        UserWarning,
        match=r"Lens mesh is depth-limited.*min_img_sep=0.1.*max_depth >= 4",
    ):
        warn_depth_limited("Lens mesh", "min_img_sep=0.1 arcsec", 0.5, 0.05, 2)


def test_warn_depth_limited_is_silent_when_the_floor_is_reached(recwarn):
    warn_depth_limited("Lens mesh", "x", 0.5, 0.05, 4)
    assert not recwarn.list
