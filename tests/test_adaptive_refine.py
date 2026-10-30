"""The shared refinement loop -- refine, balance and close -- on toy samplers."""

import numpy as np
import pytest

from caustics.lenses.func.adaptive.mesh_backend import mesh_backend
from caustics.lenses.func.adaptive.geometry import COMPOSE, ROOT_CLASS, area2
from caustics.lenses.func.adaptive.lattice import (
    initial_triangles,
    lattice_key,
    lattice_xy,
    make_lattice,
    midpoint_ij,
)
from caustics.lenses.func.adaptive.refine import (
    activate,
    add_leaves,
    add_roots,
    cache_lookup,
    close,
    empty_cache,
    empty_store,
    evaluate,
    red_split,
    refine,
    split_rows,
)

from adaptive_maps import i64, to_np

FOV, INIT_RES, MAX_LEVEL = 4.0, 2, 4


def positions(xy):
    return mesh_backend.to(xy, dtype=mesh_backend.float64)


def run(split, sample=positions, batch_size=None, max_level=MAX_LEVEL, order=None):
    lat = make_lattice(FOV, 0.0, 0.0, INIT_RES, max_level + 1)
    ij, cls = initial_triangles(INIT_RES, lat.level, ROOT_CLASS)
    if order is not None:
        ij, cls = ij[order], cls[order]
    cache, store, rows = add_roots(
        empty_cache(2), empty_store(), lat, ij, cls, sample, batch_size
    )
    cache, store = refine(cache, store, rows, lat, sample, split, max_level, batch_size)
    return lat, cache, store


def never(ij, values6, cls, level):
    return mesh_backend.zeros((ij.shape[0],), dtype=mesh_backend.bool)


def always(ij, values6, cls, level):
    return mesh_backend.ones((ij.shape[0],), dtype=mesh_backend.bool)


def around(px, py):
    """Split every triangle whose six samples' bounding box holds ``(px, py)``."""

    def split(ij, values6, cls, level):
        x, y = values6[..., 0], values6[..., 1]
        return (
            (mesh_backend.min(x, dim=1) <= px)
            & (px <= mesh_backend.max(x, dim=1))
            & (mesh_backend.min(y, dim=1) <= py)
            & (py <= mesh_backend.max(y, dim=1))
        )

    return split


def seam_and_right_level_one(ij, values6, cls, level):
    """Left half to level 3; any level-1 triangle in the right half."""
    x = values6[..., 0]
    left = mesh_backend.all(x <= 0.0, dim=1) & (level < 3)
    right = mesh_backend.all(x >= 0.0, dim=1) & (level == 1)
    return left | right


def final_split(lat, cache, store, split, max_level=MAX_LEVEL):
    rows = mesh_backend.flatnonzero(store.valid & (store.level < max_level))
    v = store.v[rows]
    ij = cache.ij[v]
    m = cache_lookup(cache, lattice_key(lat, midpoint_ij(ij)))
    values6 = cache.values[mesh_backend.concatenate((v, m), dim=1)]
    return to_np(split(ij, values6, store.cls[rows], store.level[rows]))


def closed(lat, cache, store):
    used, leaves, leaf_origin, origin, origin_leaves = close(lat, cache, store)
    return (
        to_np(cache.ij[used]),
        to_np(leaves),
        to_np(leaf_origin),
        to_np(origin_leaves),
    )


def test_a_split_that_rejects_nothing_keeps_the_level0_triangles():
    _, _, store = run(never)
    assert to_np(store.valid).all()
    assert to_np(store.level).tolist() == [0] * (2 * INIT_RES**2)


def test_a_split_that_rejects_everything_refines_uniformly_to_max_level():
    _, _, store = run(always, max_level=3)
    level = to_np(store.level)[to_np(store.valid)]
    assert (level == 3).all() and level.size == 2 * INIT_RES**2 * 4**3


def test_every_leaf_below_max_level_passes_split_even_where_balance_forced_it():
    lat, cache, store = run(seam_and_right_level_one)
    assert not final_split(lat, cache, store, seam_and_right_level_one).any()
    valid = to_np(store.valid)
    right = (to_np(cache.values)[to_np(store.v)][..., 0] >= 0).all(axis=1)
    level = to_np(store.level)
    # The split never rejects a level-0 triangle on the right, so every
    # level-1 triangle there came from the balance; each was then tested,
    # rejected and split.
    assert (level[valid & right] >= 2).all()


def test_the_leaves_are_two_to_one_balanced():
    _, cache, store = run(around(0.31, -0.57))
    valid = to_np(store.valid)
    ij = to_np(cache.ij)[to_np(store.v)][valid]
    level = to_np(store.level)[valid]
    assert level.max() >= level.min() + 2
    vertices = {tuple(p) for p in ij.reshape(-1, 2)}
    coarse = ij[level <= MAX_LEVEL - 2]
    for e in range(3):
        a, b = coarse[:, e], coarse[:, (e + 1) % 3]
        for quarter in (a + (b - a) // 4, b - (b - a) // 4):
            assert not any(tuple(p) in vertices for p in quarter)


def test_no_point_is_sampled_twice():
    seen = []

    def sample(xy):
        seen.append(to_np(xy))
        return positions(xy)

    _, cache, _ = run(around(0.31, -0.57), sample=sample)
    points = np.concatenate(seen)
    assert len({tuple(p) for p in points}) == len(points) == cache.ij.shape[0]


@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_the_closed_mesh_does_not_depend_on_the_batch_size(batch_size):
    sizes = []

    def sample(xy):
        sizes.append(xy.shape[0])
        return positions(xy)

    want = closed(*run(around(0.31, -0.57)))
    got = closed(*run(around(0.31, -0.57), sample=sample, batch_size=batch_size))
    assert max(sizes) <= batch_size
    for a, b in zip(want, got):
        assert np.array_equal(a, b)


def test_the_closed_mesh_does_not_depend_on_the_order_of_the_roots():
    n = 2 * INIT_RES**2
    order = i64(np.random.default_rng(0).permutation(n))
    want = closed(*run(around(0.31, -0.57)))
    got = closed(*run(around(0.31, -0.57), order=order))
    for a, b in zip(want, got):
        assert np.array_equal(a, b)


def test_the_closure_is_conforming_positively_oriented_and_tiles_the_square():
    lat, cache, store = run(around(0.31, -0.57))
    used, leaves, _, _, _ = close(lat, cache, store)
    xy = lattice_xy(lat, cache.ij[used])
    a2 = to_np(area2(xy[leaves]))
    assert (a2 > 0).all()
    assert np.isclose(a2.sum() / 2, FOV**2, rtol=1e-12)
    ij = to_np(cache.ij[used])
    tri = to_np(leaves)
    a = tri.reshape(-1)
    b = tri[:, [1, 2, 0]].reshape(-1)
    keys = np.minimum(a, b) * len(ij) + np.maximum(a, b)
    edges, count = np.unique(keys, return_counts=True)
    assert set(count.tolist()) <= {1, 2}
    lo, hi = edges // len(ij), edges % len(ij)
    on_side = ((ij[lo] == 0) & (ij[hi] == 0)) | ((ij[lo] == lat.n) & (ij[hi] == lat.n))
    assert np.array_equal(count == 1, on_side.any(axis=1))


def test_every_closed_leaf_s_vertices_are_among_its_origin_s_six_samples():
    lat, cache, store = run(around(0.31, -0.57))
    used, leaves, leaf_origin, origin, _ = close(lat, cache, store)
    vij = cache.ij[store.v[origin]]
    six = np.concatenate((to_np(vij), to_np(midpoint_ij(vij))), axis=1)
    leaf_ij = to_np(cache.ij[used])[to_np(leaves)]
    allowed = six[to_np(leaf_origin)]
    match = (leaf_ij[:, :, None, :] == allowed[:, None, :, :]).all(-1).any(-1)
    assert match.all()
    assert (np.bincount(to_np(leaf_origin)) > 1).any()


def test_evaluate_keeps_the_key_index_sorted_and_the_values_by_slot():
    lat = make_lattice(FOV, 0.0, 0.0, INIT_RES, 3)
    cache = empty_cache(2)
    rng = np.random.default_rng(3)
    for _ in range(6):
        keys = i64(rng.integers(0, (lat.n + 1) ** 2, 40))
        cache = evaluate(cache, lat, keys, positions, None)
    keys = to_np(cache.keys)
    assert (np.diff(keys) > 0).all()
    ij = to_np(cache.ij)[to_np(cache.slots)]
    assert np.array_equal(ij[:, 0] * (lat.n + 1) + ij[:, 1], keys)
    assert np.array_equal(to_np(cache.values), to_np(lattice_xy(lat, cache.ij)))
    assert not to_np(cache.active).any()


def test_cache_lookup_finds_cached_keys_and_misses_the_rest():
    lat = make_lattice(FOV, 0.0, 0.0, INIT_RES, 3)
    cache = evaluate(empty_cache(2), lat, i64([5, 9, 2]), positions, None)
    assert to_np(cache_lookup(cache, i64([9, 3, 2]))).tolist() == [2, -1, 0]
    assert to_np(cache_lookup(empty_cache(2), i64([1]))).tolist() == [-1]


def test_activate_and_split_rows_never_write_into_their_inputs():
    lat = make_lattice(FOV, 0.0, 0.0, INIT_RES, 3)
    ij, cls = initial_triangles(INIT_RES, lat.level, ROOT_CLASS)
    cache, store, rows = add_roots(
        empty_cache(2), empty_store(), lat, ij, cls, positions, None
    )
    cache = evaluate(
        cache, lat, lattice_key(lat, midpoint_ij(ij)).reshape(-1), positions, None
    )
    active, valid = to_np(cache.active).copy(), to_np(store.valid).copy()
    activate(cache, i64([cache.ij.shape[0] - 1]))
    split_rows(cache, store, lat, rows[:1])
    assert np.array_equal(to_np(cache.active), active)
    assert np.array_equal(to_np(store.valid), valid)


def test_add_leaves_appends_rows_and_returns_their_indices():
    store, rows = add_leaves(empty_store(), i64([[0, 1, 2]]), i64([0]), i64([0]))
    store, more = add_leaves(
        store, i64([[1, 2, 3], [2, 3, 4]]), i64([1, 1]), i64([2, 3])
    )
    assert to_np(rows).tolist() == [0] and to_np(more).tolist() == [1, 2]
    assert to_np(store.valid).all() and to_np(store.level).tolist() == [0, 1, 1]


def test_red_split_is_triangle_major_and_advances_the_class():
    v = i64([[0, 1, 2], [3, 4, 5]])
    m = i64([[6, 7, 8], [9, 10, 11]])
    cls = i64([0, 4])
    cv, cc = red_split(v, m, cls)
    assert cv.shape == (8, 3) and cc.shape == (8,)
    cv_np, cc_np = to_np(cv), to_np(cc)
    assert cv_np[0].tolist() == [0, 8, 7]  # C_1 = (theta1, m3, m2)
    assert cv_np[3].tolist() == [6, 7, 8]  # C_4 = (m1, m2, m3)
    assert cc_np[:4].tolist() == to_np(COMPOSE)[0].tolist()
    assert cc_np[4:].tolist() == to_np(COMPOSE)[4].tolist()


def test_evaluate_is_bit_identical_under_chunking():
    lat = make_lattice(4.0, 0.0, 0.0, 4, 3)

    def curved(xy):
        x, y = xy[:, 0], xy[:, 1]
        return mesh_backend.stack((x * x - y, y * y + x), dim=-1)

    keys = i64(np.arange((lat.n + 1) ** 2))
    whole = to_np(evaluate(empty_cache(2), lat, keys, curved, None).values)
    for size in (1, 7, 33):
        got = evaluate(empty_cache(2), lat, keys, curved, size)
        assert (to_np(got.values) == whole).all()


def test_evaluate_only_samples_uncached_points():
    calls = {"n": 0}

    def sample(xy):
        calls["n"] += int(xy.shape[0])
        return positions(xy)

    lat = make_lattice(4.0, 0.0, 0.0, 2, 2)
    keys = i64([3, 7, 3, 11])
    cache = evaluate(empty_cache(2), lat, keys, sample, None)
    assert calls["n"] == 3
    cache = evaluate(cache, lat, keys, sample, None)
    assert calls["n"] == 3


def closed_parts(split=None, max_level=MAX_LEVEL):
    lat, cache, store = run(split or around(0.31, -0.57), max_level=max_level)
    used, leaves, leaf_origin, origin, origin_leaves = close(lat, cache, store)
    return lat, cache, store, used, leaves, leaf_origin, origin, origin_leaves


def tiles(lat, cache, used, leaves, leaf_origin, origin_leaves):
    """Positive leaf areas, and each origin's leaves summing to its area."""
    xy = lattice_xy(lat, cache.ij[used])
    child = to_np(area2(xy[leaves]))
    whole = to_np(area2(xy[origin_leaves]))
    summed = np.zeros_like(whole)
    np.add.at(summed, to_np(leaf_origin), child)
    return (child > 0).all() and np.allclose(summed, whole, rtol=1e-12)


def test_leaf_origin_groups_are_contiguous_and_tile_their_origin():
    lat, cache, _, used, leaves, leaf_origin, _, origin_leaves = closed_parts()
    assert (np.diff(to_np(leaf_origin)) >= 0).all()
    assert tiles(lat, cache, used, leaves, leaf_origin, origin_leaves)


def test_an_unclosed_leaf_is_its_own_origin():
    _, _, _, _, leaves, leaf_origin, _, origin_leaves = closed_parts()
    counts = np.bincount(to_np(leaf_origin))
    solo = np.flatnonzero(counts == 1)
    assert solo.size > 0
    rows = np.searchsorted(to_np(leaf_origin), solo)
    assert np.array_equal(to_np(leaves)[rows], to_np(origin_leaves)[solo])


def test_closure_uses_only_leaf_vertices():
    _, cache, _, used, _, _, _, _ = closed_parts()
    assert to_np(cache.active)[to_np(used)].all()


def test_closure_re_emits_every_origin_vertex():
    for split in (around(0.31, -0.57), seam_and_right_level_one):
        parts = closed_parts(split)
        leaves, leaf_origin, origin_leaves = parts[4], parts[5], parts[7]
        lo, lv, ol = to_np(leaf_origin), to_np(leaves), to_np(origin_leaves)
        for row in range(ol.shape[0]):
            assert set(ol[row].tolist()) <= set(lv[lo == row].reshape(-1).tolist())


def all_but_one_to_level_two(ij, values6, cls, level):
    """Every triangle to level 2 but the level-1 corner child at ``(1/3, 2/3)``."""
    cx = mesh_backend.sum(values6[:, :3, 0], dim=1) / 3
    cy = mesh_backend.sum(values6[:, :3, 1], dim=1) / 3
    spared = (level == 1) & (mesh_backend.abs(cx - 1 / 3) < 1e-9)
    spared = spared & (mesh_backend.abs(cy - 2 / 3) < 1e-9)
    return (level < 2) & ~spared


def test_close_reaches_the_count_equals_3_branch():
    lat, cache, _, used, leaves, leaf_origin, _, origin_leaves = closed_parts(
        all_but_one_to_level_two, max_level=3
    )
    assert (np.bincount(to_np(leaf_origin)) == 4).any()
    assert tiles(lat, cache, used, leaves, leaf_origin, origin_leaves)


def test_refining_the_roots_in_two_passes_gives_the_one_pass_mesh():
    split = around(0.02, -0.57)
    lat = make_lattice(FOV, 0.0, 0.0, INIT_RES, MAX_LEVEL + 1)
    ij, cls = initial_triangles(INIT_RES, lat.level, ROOT_CLASS)
    right_of_seam = mesh_backend.max(ij[:, :, 0], dim=1) > lat.n // 2
    first = mesh_backend.flatnonzero(~right_of_seam)
    second = mesh_backend.flatnonzero(right_of_seam)
    cache, store, rows = add_roots(
        empty_cache(2), empty_store(), lat, ij[first], cls[first], positions, None
    )
    cache, store = refine(cache, store, rows, lat, positions, split, MAX_LEVEL, None)
    cache, store, rows = add_roots(
        cache, store, lat, ij[second], cls[second], positions, None
    )
    cache, store = refine(cache, store, rows, lat, positions, split, MAX_LEVEL, None)
    for a, b in zip(closed(*run(split)), closed(lat, cache, store)):
        assert np.array_equal(a, b)
