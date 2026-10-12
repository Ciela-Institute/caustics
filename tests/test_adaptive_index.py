"""The multi-level source-plane index and its lookups."""

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend, to_mesh
from caustics.lenses.func.adaptive.geometry import triangle_weights
from caustics.lenses.func.adaptive.index import (
    CELLS_PER_BOX,
    KEY_BITS,
    as_points,
    build_index,
    index_cells,
    index_hits,
)

from adaptive_maps import (
    assert_hits_equal,
    assert_same,
    brute_hits,
    build,
    collapse,
    collapse_jacobian,
    f64,
    finite_rows,
    i64,
    index_cell,
    index_cell_ranges,
    index_point_cells,
    SIE_LIKE_LENS,
    to_np,
)
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import Point
from caustics.lenses.func.adaptive.lens_mesh import build_lens_mesh, leaf_grow
from caustics.lenses.func.adaptive import index as index_module


def test_build_index_of_an_empty_leaf_set_has_no_levels_and_finds_nothing():
    vs, leaves = f64(np.zeros((3, 2))), i64(np.zeros((0, 3)))
    idx = build_index(vs, leaves, i64(np.zeros(0)))
    assert tuple(idx.levels.shape) == (0,) and tuple(idx.keys.shape) == (0,)
    assert to_np(idx.offsets).tolist() == [0] and tuple(idx.leaves.shape) == (0,)
    start, count = index_cells(idx, f64([[0.5, 0.5]]))
    assert tuple(start.shape) == tuple(count.shape) == (1, 0)
    hits = index_hits(idx, vs, leaves, f64([[0.5, 0.5]]))
    assert [tuple(a.shape) for a in hits] == [(0,), (0,), (0, 3)]


@pytest.mark.skipif(
    backend.backend != "torch", reason="needs torch's meta default device"
)
@pytest.mark.parametrize("n_rows", [0, 60], ids=["empty", "filled"])
def test_build_index_allocates_on_its_inputs_device_not_the_default_one(n_rows):
    """The index is built where its inputs live, which need not be the default device.

    With the default device set to ``meta``, anything ``build_index``
    allocates without its inputs' device lands there and fails to mix.
    """
    import torch

    rng = np.random.default_rng(13)
    vs = f64(rng.normal(size=(50, 2)))
    leaves = i64(rng.integers(0, 50, (60, 3)))
    rows = i64(np.arange(n_rows))
    with torch.device("meta"):
        idx = build_index(vs, leaves, rows)
    for name in ("lo", "hi", "fine", "levels", "keys", "offsets", "leaves"):
        assert getattr(idx, name).device == vs.device, name


def test_as_points_flattens_any_shape_into_float64_rows():
    got = as_points(2.0, 3.0, None)
    assert got.dtype == mesh_backend.float64 and to_np(got).tolist() == [[2.0, 3.0]]
    got = as_points(f64([[0.0, 1.0]]), f64([[2.0, 3.0]]), None)
    assert to_np(got).tolist() == [[0.0, 2.0], [1.0, 3.0]]
    assert as_points(f64([]), f64([]), None).shape == (0, 2)


def test_build_index_with_a_zero_grow_is_the_plain_index():
    rng = np.random.default_rng(11)
    vs = f64(rng.uniform(-1, 1, (90, 2)))
    leaves = i64(np.arange(90).reshape(30, 3))
    rows = i64(np.arange(30))
    assert_same(
        build_index(vs, leaves, rows, f64(np.zeros(30))), build_index(vs, leaves, rows)
    )


def test_index_hits_with_grow_returns_a_point_within_grow_of_a_triangle_and_not_beyond():
    vs = f64([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0], [5.0, 6.0]])
    leaves = i64([[0, 1, 2], [3, 4, 5]])
    grow = f64([0.2, 0.0])
    idx = build_index(vs, leaves, i64([0, 1]), grow)
    beta = f64([[0.5, -0.1], [0.5, -0.3], [0.25, 0.25], [5.5, 4.9]])
    qidx, tri, w = index_hits(idx, vs, leaves, beta, grow)
    assert list(zip(to_np(qidx).tolist(), to_np(tri).tolist())) == [(0, 0), (2, 0)]
    expected = triangle_weights(vs[leaves[tri]], beta[qidx])
    assert np.array_equal(to_np(w), to_np(expected))
    plain_q, plain_t, _ = index_hits(idx, vs, leaves, beta)
    assert list(zip(to_np(plain_q).tolist(), to_np(plain_t).tolist())) == [(2, 0)]


def _scattered_triangles(seed, n=400):
    """``n`` triangles with sides from 1e-9 to 1e2 arcsec, the first five moved out to about 1e6."""
    rng = np.random.default_rng(seed)
    size = 10.0 ** rng.uniform(-9, 2, n)
    corner = rng.uniform(-1, 1, (n, 2))
    corner[:5] *= 1e6
    return corner[:, None, :] + size[:, None, None] * rng.uniform(-1, 1, (n, 3, 2))


def _as_mesh(tri):
    """``tri`` ``(n, 3, 2)`` as vertices, triangles and all their rows."""
    n = tri.shape[0]
    return (
        f64(tri.reshape(-1, 2)),
        i64(np.arange(3 * n).reshape(n, 3)),
        i64(np.arange(n)),
    )


def test_index_hits_equal_brute_force_across_eleven_orders_of_magnitude():
    tri = _scattered_triangles(9)
    vs, leaves, rows = _as_mesh(tri)
    rng = np.random.default_rng(10)
    n = tri.shape[0]
    grow = f64(np.where(rng.uniform(size=n) < 0.2, 10.0 ** rng.uniform(-9, 0, n), 0.0))
    inner = tri[:, 0] + 0.3 * (tri[:, 1] - tri[:, 0]) + 0.3 * (tri[:, 2] - tri[:, 0])
    beta = f64(np.concatenate([inner, tri[:, 1], rng.uniform(-1, 1, (200, 2))]))
    idx = build_index(vs, leaves, rows, grow)
    assert_hits_equal(
        index_hits(idx, vs, leaves, beta, grow),
        brute_hits(vs, leaves, rows, beta, grow),
    )


def test_index_hits_skip_nan_infinite_and_far_points_and_take_empty_input():
    rng = np.random.default_rng(14)
    tri = rng.uniform(-1, 1, (40, 1, 2)) + 0.2 * rng.uniform(-1, 1, (40, 3, 2))
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows)
    beta = f64(
        [[np.nan, 0.0], [np.inf, 0.0], [0.0, -np.inf], [1e9, 1e9], tri[3].mean(axis=0)]
    )
    qidx, hit, w = index_hits(idx, vs, leaves, beta)
    assert set(to_np(qidx).tolist()) == {4}
    want_q, want_t, want_w = brute_hits(vs, leaves, rows, beta[4:])
    assert want_t.size > 0
    assert_hits_equal((qidx, hit, w), (want_q + 4, want_t, want_w))
    empty = index_hits(idx, vs, leaves, f64(np.zeros((0, 2))))
    assert [tuple(a.shape) for a in empty] == [(0,), (0,), (0, 3)]


def test_build_index_registers_each_row_at_one_level_in_exactly_the_cells_of_its_box():
    tri = _scattered_triangles(21)
    n = tri.shape[0]
    rng = np.random.default_rng(22)
    grow = np.where(rng.uniform(size=n) < 0.2, 10.0 ** rng.uniform(-9, 0, n), 0.0)
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows, f64(grow))
    level, i0, i1 = index_cell_ranges(idx, tri, grow)
    assert np.unique(level).size > 5, "fixture must spread over many levels"
    assert ((i1 - i0 + 1) <= CELLS_PER_BOX + 2).all()
    keys, offsets = to_np(idx.keys), to_np(idx.offsets)
    key_of_entry = np.repeat(keys, np.diff(offsets))
    row_of_entry = to_np(idx.leaves)
    for r in range(n):
        ix, iy = np.meshgrid(
            np.arange(i0[r, 0], i1[r, 0] + 1),
            np.arange(i0[r, 1], i1[r, 1] + 1),
            indexing="ij",
        )
        want = (level[r] << (2 * KEY_BITS)) | (ix.ravel() << KEY_BITS) | iy.ravel()
        got = key_of_entry[row_of_entry == r]
        assert np.array_equal(np.sort(got), np.sort(want)), r


def test_build_index_keys_ascend_and_give_its_levels_and_fine_side():
    vs, leaves, rows = _as_mesh(_scattered_triangles(23))
    idx = build_index(vs, leaves, rows)
    keys, offsets = to_np(idx.keys), to_np(idx.offsets)
    assert (np.diff(keys) > 0).all()
    assert offsets[0] == 0 and offsets[-1] == to_np(idx.leaves).size
    assert (np.diff(offsets) > 0).all()
    assert to_np(idx.levels).tolist() == np.unique(keys >> (2 * KEY_BITS)).tolist()
    entries = to_np(idx.leaves)
    assert all(
        (np.diff(entries[a:b]) > 0).all() for a, b in zip(offsets[:-1], offsets[1:])
    )
    assert (
        np.diff(offsets) > 1
    ).any(), "fixture must put several triangles in one cell"
    extent = (to_np(idx.hi) - to_np(idx.lo)).max()
    assert float(to_np(idx.fine)) == extent / 2**KEY_BITS


def test_index_cells_gives_the_cell_each_point_falls_in_at_every_level():
    tri = _scattered_triangles(27)
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows)
    rng = np.random.default_rng(28)
    beta = np.concatenate(
        [tri[:, 0], rng.uniform(-1, 1, (100, 2)), [[np.nan, 0.0], [1e9, 1e9]]]
    )
    start, count = index_cells(idx, f64(beta))
    start, count = to_np(start), to_np(count)
    levels, entries = to_np(idx.levels), to_np(idx.leaves)
    assert start.shape == count.shape == (beta.shape[0], levels.size)
    lo, hi = to_np(idx.lo), to_np(idx.hi)
    for b, point in enumerate(beta):
        inside = bool(np.all((point >= lo) & (point <= hi)))
        for j, level in enumerate(levels):
            want = np.zeros(0, dtype=np.int64)
            if inside:
                ix, iy = index_point_cells(idx, [level], point[None])[0]
                want = index_cell(idx, level, ix, iy)
            got = entries[start[b, j] : start[b, j] + count[b, j]]
            assert np.array_equal(got, want), (b, level)


def test_index_hits_find_points_on_the_cell_lines_of_every_level():
    """Points snapped onto each level's cell lines, and ``hi`` itself, are found as brute force finds them."""
    tri = _scattered_triangles(25)[5:]
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows)
    lo, fine = to_np(idx.lo), float(to_np(idx.fine))
    points = [to_np(idx.hi)[None]]
    for level in to_np(idx.levels):
        side = fine * 2.0**level
        snapped = lo + np.floor((tri[::20, 0] - lo) / side) * side
        points += [snapped, np.stack([snapped[:, 0], tri[::20, 0, 1]], axis=-1)]
    beta = f64(np.concatenate(points))
    assert_hits_equal(
        index_hits(idx, vs, leaves, beta), brute_hits(vs, leaves, rows, beta)
    )


def test_a_kappa_one_sheet_s_single_source_point_hits_every_leaf():
    """Every source vertex is one point, so the index has no extent; that point lies in every leaf."""
    mesh, _ = build(collapse, collapse_jacobian, min_img_sep=1.0)
    vs = mesh.vertices_source
    assert (
        np.unique(to_np(vs), axis=0).shape[0] == 1
    ), "fixture must collapse every vertex"
    _, hit, _ = index_hits(mesh.index, vs, mesh.leaves, vs[:1])
    assert to_np(hit).tolist() == list(range(mesh.leaves.shape[0]))


BUILD = dict(fov=6.0, init_res=4, min_img_sep=0.1, max_depth=6)


@pytest.fixture(scope="module")
def mesh():
    return to_mesh(
        build_lens_mesh(
            SIE_LIKE_LENS.raytrace, SIE_LIKE_LENS.jacobian_lens_equation, **BUILD
        )
    )


@pytest.fixture(scope="module")
def beta():
    rng = np.random.default_rng(21)
    return mesh_backend.as_array(
        rng.uniform(-1.5, 1.5, (64, 2)), dtype=mesh_backend.float64
    )


@pytest.fixture(scope="module")
def point_mesh():
    """A point mass of Einstein radius 1 with its center a hole: source triangles from about 1e-12 to 1e4 arcsec^2."""
    lens = Point(
        name="pt",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        Rein=1.0,
        s=0.0,
    )
    mesh = build_lens_mesh(
        lens.raytrace,
        lens.jacobian_lens_equation,
        fov=5.0,
        init_res=32,
        min_img_sep=1e-2,
        centers=[[0.0, 0.0]],
    )
    return to_mesh(mesh)


def _hits_against_brute_force(mesh, points):
    """``index_hits`` on ``mesh`` at ``points``, asserted equal to brute force over every finite leaf."""
    grow = leaf_grow(mesh.origin_status, mesh.origin_deviation, mesh.leaf_origin)
    rows = i64(finite_rows(mesh))
    got = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, points, grow)
    assert_hits_equal(
        got, brute_hits(mesh.vertices_source, mesh.leaves, rows, points, grow)
    )
    return got


def test_index_hits_equal_brute_force_on_the_sie_like_mesh(mesh, beta):
    vs = to_np(mesh.vertices_source)
    finite = vs[np.isfinite(vs).all(axis=1)]
    hi = to_np(mesh.index.hi)
    upper = np.array([hi, [hi[0], 0.0], [0.0, hi[1]]])
    points = f64(np.concatenate([to_np(beta), finite[::7][:40], upper]))
    qidx, _, _ = _hits_against_brute_force(mesh, points)
    assert to_np(qidx).size > 0


def test_index_hits_equal_brute_force_on_a_point_mass_mesh(point_mesh):
    rng = np.random.default_rng(31)
    vs = to_np(point_mesh.vertices_source)
    finite = vs[np.isfinite(vs).all(axis=1)]
    points = np.concatenate(
        [
            rng.uniform(-0.3, 0.3, (60, 2)),
            rng.uniform(-5.0, 5.0, (30, 2)),
            finite[rng.choice(len(finite), 20, replace=False)],
        ]
    )
    qidx, _, _ = _hits_against_brute_force(point_mesh, f64(points))
    assert to_np(qidx).size > 0


def test_point_mass_candidates_average_at_most_three_times_the_box_floor(point_mesh):
    """Near a point mass's caustic, candidates per point average at most three times the floor.

    The floor is the number of grown leaf boxes containing the point, which
    no bounding-box index can go below. The single uniform grid this index
    replaced read 700 to 2,700 times the floor here, and the multi-level
    grid measured 1.3 to 2.0 times on 13 lens meshes. The bound is on the
    mean: next to the point-like caustic the floor itself reaches the
    thousands, and where the floor is small, candidates have exceeded four
    times it by up to 219.
    """
    rows = finite_rows(point_mesh)
    vs, leaves = to_np(point_mesh.vertices_source), to_np(point_mesh.leaves)
    grow = to_np(
        leaf_grow(
            point_mesh.origin_status,
            point_mesh.origin_deviation,
            point_mesh.leaf_origin,
        )
    )[rows]
    tri = vs[leaves[rows]]
    box_lo, box_hi = tri.min(axis=1) - grow[:, None], tri.max(axis=1) + grow[:, None]
    beta = np.random.default_rng(41).uniform(-0.3, 0.3, (300, 2))
    floor = np.array(
        [np.all((box_lo <= p) & (p <= box_hi), axis=1).sum() for p in beta]
    )
    _, count = index_cells(point_mesh.index, f64(beta))
    candidates = to_np(count).sum(axis=1)
    assert candidates.mean() <= 3 * floor.mean()


def test_index_hits_hold_at_most_pair_cap_pairs_per_step(mesh, beta, monkeypatch):
    """Each step of a query holds at most ``PAIR_CAP`` pairs, or one point holding more, and the hits do not change."""
    grow = leaf_grow(mesh.origin_status, mesh.origin_deviation, mesh.leaf_origin)
    whole = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, beta, grow)
    steps = []
    chunk_hits = index_module._chunk_hits

    def spy(index, vertices, triangles, points, start, count, total, grow):
        steps.append((points.shape[0], total))
        return chunk_hits(index, vertices, triangles, points, start, count, total, grow)

    monkeypatch.setattr(index_module, "_chunk_hits", spy)
    over_cap = shared = False
    for cap in (7, 64):
        monkeypatch.setattr(index_module, "PAIR_CAP", cap)
        steps.clear()
        got = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, beta, grow)
        for name, a, b in zip(("qidx", "tri", "w"), got, whole):
            np.testing.assert_array_equal(
                to_np(a), to_np(b), err_msg=f"{name}, cap {cap}"
            )
        assert all(total <= cap or n == 1 for n, total in steps), steps
        over_cap |= any(n == 1 and total > cap for n, total in steps)
        shared |= any(n > 1 for n, _ in steps)
    assert over_cap, "no point held more than the cap"
    assert shared, "no step held several points"
