"""
Refinement on the dyadic lattice, shared by the lens mesh and the source mesh of a magnification region.

A ``sample`` callable maps ``(N, 2)`` float64 positions to ``(N, C)``
float64 values. Every lattice point it is called on is kept in a
:class:`VertexCache`, so no point is sampled twice. :func:`refine` replaces
every leaf below ``max_level`` that ``split`` rejects by its four red-split
children, balances the leaves to 2:1 and tests the leaves the balance
created, until every leaf below ``max_level`` passes. :func:`close` makes
the result conforming.
"""

import math
from typing import NamedTuple

from ....backend_obj import ArrayLike
from .mesh_backend import mesh_backend
from .geometry import COMPOSE, _CHILD_VERTEX_INDEX_TABLE, expand, min_angle, sorted_find
from .lattice import lattice_ij_from_key, lattice_key, lattice_xy, midpoint_ij


class VertexCache(NamedTuple):
    """
    Every sampled lattice point, by slot, with a sorted key index.

    Parameters
    ----------
    keys: ArrayLike
        ``(N,)`` int64 lattice keys, ascending.
    slots: ArrayLike
        ``(N,)`` int64 slot of each of ``keys``.
    ij: ArrayLike
        ``(N, 2)`` int64 lattice coordinates, by slot.
    values: ArrayLike
        ``(N, C)`` float64 samples, by slot.
    active: ArrayLike
        ``(N,)`` bool, by slot: True where the point is a vertex of a leaf.
    """

    keys: ArrayLike
    slots: ArrayLike
    ij: ArrayLike
    values: ArrayLike
    active: ArrayLike


class LeafStore(NamedTuple):
    """
    Leaves by row; a split row stays, with ``valid`` False.

    Parameters
    ----------
    v: ArrayLike
        ``(R, 3)`` int64 vertex slots, positively oriented.
    level: ArrayLike
        ``(R,)`` int64 refinement level.
    cls: ArrayLike
        ``(R,)`` int64 orientation class, see
        :func:`~caustics.lenses.func.adaptive.geometry.child_matrix_tables`.
    valid: ArrayLike
        ``(R,)`` bool.
    """

    v: ArrayLike
    level: ArrayLike
    cls: ArrayLike
    valid: ArrayLike


def empty_cache(n_values):
    """A :class:`VertexCache` with no point, for ``n_values`` sample columns."""
    int64 = mesh_backend.int64
    return VertexCache(
        keys=mesh_backend.zeros((0,), dtype=int64),
        slots=mesh_backend.zeros((0,), dtype=int64),
        ij=mesh_backend.zeros((0, 2), dtype=int64),
        values=mesh_backend.zeros((0, n_values), dtype=mesh_backend.float64),
        active=mesh_backend.zeros((0,), dtype=mesh_backend.bool),
    )


def empty_store():
    """A :class:`LeafStore` with no row."""
    int64 = mesh_backend.int64
    return LeafStore(
        v=mesh_backend.zeros((0, 3), dtype=int64),
        level=mesh_backend.zeros((0,), dtype=int64),
        cls=mesh_backend.zeros((0,), dtype=int64),
        valid=mesh_backend.zeros((0,), dtype=mesh_backend.bool),
    )


def cache_lookup(cache, keys):
    """Slot of each of ``keys``, ``-1`` where it is not cached."""
    if cache.keys.shape[0] == 0:
        return mesh_backend.zeros_like(keys) - 1
    pos, found = sorted_find(cache.keys, keys)
    return mesh_backend.where(found, cache.slots[pos], -1)


def sample_points(xy, sample, batch_size):
    """
    ``sample`` on ``(N, 2)`` positions, at most ``batch_size`` rows per call.

    ``sample`` acts on each row alone, so the result does not depend on
    ``batch_size``.
    """
    if batch_size is None or xy.shape[0] <= batch_size:
        return sample(xy)
    chunks = mesh_backend.chunk(xy, math.ceil(xy.shape[0] / batch_size), dim=0)
    return mesh_backend.concatenate([sample(chunk) for chunk in chunks], dim=0)


def evaluate(cache, lat, keys, sample, batch_size):
    """
    ``cache`` with every one of ``keys`` sampled; a cached key is not sampled again.

    The new keys are merged into the sorted key index rather than re-sorted:
    all keys being distinct, a new key lands at its ``searchsorted``
    position plus its rank among the new keys.
    """
    new = mesh_backend.unique(keys)
    new = new[cache_lookup(cache, new) < 0]
    k = new.shape[0]
    if k == 0:
        return cache
    int64 = mesh_backend.int64
    ij = lattice_ij_from_key(lat, new)
    values = sample_points(lattice_xy(lat, ij), sample, batch_size)
    start = cache.ij.shape[0]
    total = cache.keys.shape[0] + k
    dest = mesh_backend.searchsorted(cache.keys, new) + mesh_backend.arange(
        k, dtype=int64
    )
    rest = mesh_backend.flatnonzero(
        mesh_backend.fill_at_indices(
            mesh_backend.ones((total,), dtype=mesh_backend.bool), dest, False
        )
    )
    keys_out = mesh_backend.fill_at_indices(
        mesh_backend.zeros((total,), dtype=int64), dest, new
    )
    slots_out = mesh_backend.fill_at_indices(
        mesh_backend.zeros((total,), dtype=int64),
        dest,
        mesh_backend.arange(start, start + k, dtype=int64),
    )
    return VertexCache(
        keys=mesh_backend.fill_at_indices(keys_out, rest, cache.keys),
        slots=mesh_backend.fill_at_indices(slots_out, rest, cache.slots),
        ij=mesh_backend.concatenate((cache.ij, ij), dim=0),
        values=mesh_backend.concatenate((cache.values, values), dim=0),
        active=mesh_backend.concatenate(
            (cache.active, mesh_backend.zeros((k,), dtype=mesh_backend.bool)), dim=0
        ),
    )


def activate(cache, slots):
    """``cache`` with ``slots`` marked as leaf vertices; ``cache`` itself is untouched."""
    return cache._replace(
        active=mesh_backend.fill_at_indices(
            mesh_backend.copy(cache.active), slots, True
        )
    )


def add_leaves(store, v, level, cls):
    """``store`` with rows ``v`` appended, and the new rows' indices."""
    start, k = store.v.shape[0], v.shape[0]
    store = LeafStore(
        v=mesh_backend.concatenate((store.v, v), dim=0),
        level=mesh_backend.concatenate((store.level, level), dim=0),
        cls=mesh_backend.concatenate((store.cls, cls), dim=0),
        valid=mesh_backend.concatenate(
            (store.valid, mesh_backend.ones((k,), dtype=mesh_backend.bool)), dim=0
        ),
    )
    return store, mesh_backend.arange(start, start + k, dtype=mesh_backend.int64)


def add_roots(cache, store, lat, ij, cls, sample, batch_size):
    """
    Level-0 triangles, their vertices sampled and active.

    Parameters
    ----------
    ij: ArrayLike
        ``(T, 3, 2)`` int64 lattice coordinates.
    cls: ArrayLike
        ``(T,)`` int64 orientation classes.

    Returns
    -------
    cache: VertexCache
    store: LeafStore
    rows: ArrayLike
        ``(T,)`` int64 rows of the new triangles.
    """
    keys = lattice_key(lat, ij)
    cache = evaluate(cache, lat, keys.reshape(-1), sample, batch_size)
    v = cache_lookup(cache, keys)
    level = mesh_backend.zeros((v.shape[0],), dtype=mesh_backend.int64)
    store, rows = add_leaves(store, v, level, cls)
    return activate(cache, v.reshape(-1)), store, rows


def resume(lat, k, vertices_ij, values, origin_leaves, origin_level, origin_cls):
    """
    The refinement state of a frozen mesh, on its lattice grown by ``k`` level-0 cells a side.

    The vertices are in lattice-key order, which a uniform shift keeps, so
    shifted by ``k << lat.level`` they are the cache as they stand, every one
    a leaf vertex; the origins are the store, row for row.

    Returns
    -------
    cache: VertexCache
    store: LeafStore
    """
    int64 = mesh_backend.int64
    ij = vertices_ij + (int(k) << lat.level)
    n_vertices, n_origins = ij.shape[0], origin_leaves.shape[0]
    cache = VertexCache(
        keys=lattice_key(lat, ij),
        slots=mesh_backend.arange(n_vertices, dtype=int64),
        ij=ij,
        values=values,
        active=mesh_backend.ones((n_vertices,), dtype=mesh_backend.bool),
    )
    store = LeafStore(
        v=origin_leaves,
        level=origin_level,
        cls=origin_cls,
        valid=mesh_backend.ones((n_origins,), dtype=mesh_backend.bool),
    )
    return cache, store


def red_split(v, m, cls):
    """
    The four children of each triangle, triangle-major.

    Parameters
    ----------
    v, m: ArrayLike
        ``(n, 3)`` int64 vertex and edge-midpoint slots, ``m_i`` opposite ``v_i``.
    cls: ArrayLike
        ``(n,)`` int64 orientation classes.

    Returns
    -------
    child_v: ArrayLike
        ``(4n, 3)`` int64 vertex slots.
    child_cls: ArrayLike
        ``(4n,)`` int64 orientation classes.
    """
    six = mesh_backend.concatenate((v, m), dim=1)
    return six[:, _CHILD_VERTEX_INDEX_TABLE].reshape(-1, 3), COMPOSE[cls].reshape(-1)


def split_rows(cache, store, lat, rows):
    """
    Replace ``rows`` by their red-split children, whose midpoints must be cached.

    Returns the cache with the children's vertices active, the store, and
    the children's rows.
    """
    v = store.v[rows]
    m = cache_lookup(cache, lattice_key(lat, midpoint_ij(cache.ij[v])))
    kid_v, kid_cls = red_split(v, m, store.cls[rows])
    kid_level = mesh_backend.repeat(store.level[rows] + 1, 4, axis=0)
    store = store._replace(
        valid=mesh_backend.fill_at_indices(mesh_backend.copy(store.valid), rows, False)
    )
    store, kids = add_leaves(store, kid_v, kid_level, kid_cls)
    return activate(cache, kid_v.reshape(-1)), store, kids


def _split_failures(cache, store, lat, rows, sample, split, batch_size):
    """Sample the midpoints of ``rows``, and split every row ``split`` rejects."""
    v = store.v[rows]
    ij = cache.ij[v]
    mid = lattice_key(lat, midpoint_ij(ij))
    cache = evaluate(cache, lat, mid.reshape(-1), sample, batch_size)
    six = mesh_backend.concatenate((v, cache_lookup(cache, mid)), dim=1)
    fail = split(ij, cache.values[six], store.cls[rows], store.level[rows])
    return split_rows(cache, store, lat, rows[mesh_backend.flatnonzero(fail)])


def unbalanced(cache, store, lat, max_level):
    """
    Valid rows with a leaf vertex at a quarter point of an edge.

    Such a vertex belongs to a neighbour two or more levels finer. Only a row
    at ``level <= max_level - 2`` can have one, and its quarter points are
    lattice points.
    """
    rows = mesh_backend.flatnonzero(store.valid & (store.level <= max_level - 2))
    if rows.shape[0] == 0:
        return rows
    ij = cache.ij[store.v[rows]]
    hit = mesh_backend.zeros((rows.shape[0],), dtype=mesh_backend.bool)
    for e in range(3):
        a, b = ij[:, e], ij[:, (e + 1) % 3]
        delta = (b - a) // 4
        for quarter in (a + delta, b - delta):
            slot = cache_lookup(cache, lattice_key(lat, quarter))
            hit = hit | (
                (slot >= 0) & cache.active[mesh_backend.where(slot >= 0, slot, 0)]
            )
    return rows[hit]


def balance(cache, store, lat, max_level, sample, batch_size):
    """
    Force-split until no leaf has a neighbour two or more levels finer.

    A split row has a leaf vertex at a quarter point, and leaf vertices are
    never removed, so every 2:1-balanced refinement of the leaves splits it
    too: the fixed point is the coarsest balanced refinement, whatever order
    rows are met in.

    Returns
    -------
    cache: VertexCache
    store: LeafStore
    created: ArrayLike
        Rows created, some of which a later round may have split again.
    """
    created = [mesh_backend.zeros((0,), dtype=mesh_backend.int64)]
    while True:
        rows = unbalanced(cache, store, lat, max_level)
        if rows.shape[0] == 0:
            return cache, store, mesh_backend.concatenate(created, dim=0)
        mid = lattice_key(lat, midpoint_ij(cache.ij[store.v[rows]]))
        cache = evaluate(cache, lat, mid.reshape(-1), sample, batch_size)
        cache, store, kids = split_rows(cache, store, lat, rows)
        created.append(kids)


def refine(cache, store, untested, lat, sample, split, max_level, batch_size):
    """
    Split until every leaf below ``max_level`` passes ``split`` and the leaves are 2:1 balanced.

    Each round samples the midpoints of the untested rows below
    ``max_level`` in one batch and replaces the rows ``split`` rejects by
    their children, which are untested in turn. Once none fails,
    :func:`balance` runs, and the rows it created are tested like any other.
    ``max_level`` rows are never tested.

    Parameters
    ----------
    cache: VertexCache
        Holding every vertex of ``store``'s rows.
    store: LeafStore
    untested: ArrayLike
        ``(K,)`` int64 rows to test.
    lat: Lattice
    sample: Callable[[ArrayLike], ArrayLike]
        ``(N, 2)`` float64 positions to ``(N, C)`` float64 values.
    split: Callable
        ``split(ij, values6, cls, level) -> (n,) bool``, True where a row
        must split. ``ij`` is ``(n, 3, 2)`` int64, the rows' vertex lattice
        coordinates; ``values6`` is ``(n, 6, C)``, the values at the
        vertices and the edge midpoints ``m_1, m_2, m_3``, ``m_i`` opposite
        vertex ``i``; ``cls`` and ``level`` are ``(n,)`` int64.
    max_level: int
    batch_size: Optional[int]
        Most rows per ``sample`` call.

    Returns
    -------
    cache: VertexCache
    store: LeafStore
    """
    while True:
        rows = untested[store.level[untested] < max_level]
        while rows.shape[0]:
            cache, store, kids = _split_failures(
                cache, store, lat, rows, sample, split, batch_size
            )
            rows = kids[store.level[kids] < max_level]
        cache, store, created = balance(
            cache, store, lat, max_level, sample, batch_size
        )
        untested = created[store.valid[created]]
        if untested.shape[0] == 0:
            return cache, store


def canonical_order(lat, cache, v):
    """Permutation sorting leaves ``v`` ``(n, 3)`` by their sorted vertex keys, which identify a triangle."""
    keys = mesh_backend.sort(lattice_key(lat, cache.ij[v]), dim=1)
    return mesh_backend.lexsort([keys[:, 2], keys[:, 1], keys[:, 0]])


def close(lat, cache, store):
    """
    The conforming closure of a balanced refinement, with its vertices compacted.

    The valid rows are put in canonical order, so the result depends on the
    refinement alone. A leaf with leaf vertices at the midpoints of ``h`` of
    its edges -- hanging nodes -- becomes ``h + 1`` triangles: bisected from
    the opposite vertex for one; for two, the corner child at the vertex
    between them, then the diagonal of the remaining quadrilateral that
    maximizes the smaller minimum angle, ties taking the first; the red
    split for three. Every triangle emitted is positively oriented, and its
    vertices are among its origin's six samples.

    Returns
    -------
    used: ArrayLike
        ``(V,)`` int64 slots of the vertices, in lattice-key order.
    leaves: ArrayLike
        ``(L, 3)`` int64 closed leaves, indices into ``used``.
    leaf_origin: ArrayLike
        ``(L,)`` int64 index into ``origin`` of each leaf's pre-closure
        leaf, non-decreasing.
    origin: ArrayLike
        ``(N,)`` int64 store rows of the pre-closure leaves, in canonical order.
    origin_leaves: ArrayLike
        ``(N, 3)`` int64 their vertices, indices into ``used``.
    """
    int64 = mesh_backend.int64
    origin = mesh_backend.flatnonzero(store.valid)
    origin = origin[canonical_order(lat, cache, store.v[origin])]
    v = store.v[origin]
    m = cache_lookup(cache, lattice_key(lat, midpoint_ij(cache.ij[v])))
    hanging = (m >= 0) & cache.active[mesh_backend.where(m >= 0, m, 0)]
    count = mesh_backend.sum(mesh_backend.long(hanging), dim=1)
    n_children = count + 1
    offsets = mesh_backend.cumsum(n_children, dim=0) - n_children
    total = int(mesh_backend.to_numpy(mesh_backend.sum(n_children)))
    leaves = mesh_backend.zeros((total, 3), dtype=int64)
    leaf_origin, _ = expand(n_children, total)

    def xy(slots):
        return lattice_xy(lat, cache.ij[slots])

    sel = mesh_backend.flatnonzero(count == 0)
    leaves = mesh_backend.fill_at_indices(leaves, offsets[sel], v[sel])
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = mesh_backend.flatnonzero((count == 1) & hanging[:, i])
        if sel.shape[0] == 0:
            continue
        o = offsets[sel]
        leaves = mesh_backend.fill_at_indices(
            leaves, o, mesh_backend.stack((v[sel, i], v[sel, j], m[sel, i]), dim=1)
        )
        leaves = mesh_backend.fill_at_indices(
            leaves, o + 1, mesh_backend.stack((v[sel, i], m[sel, i], v[sel, k]), dim=1)
        )
    for c in range(3):
        a, b = (c + 1) % 3, (c + 2) % 3
        sel = mesh_backend.flatnonzero((count == 2) & ~hanging[:, c])
        if sel.shape[0] == 0:
            continue
        o = offsets[sel]
        leaves = mesh_backend.fill_at_indices(
            leaves, o, mesh_backend.stack((v[sel, c], m[sel, b], m[sel, a]), dim=1)
        )
        a1 = mesh_backend.stack((v[sel, a], v[sel, b], m[sel, a]), dim=1)
        a2 = mesh_backend.stack((v[sel, a], m[sel, a], m[sel, b]), dim=1)
        b1 = mesh_backend.stack((v[sel, a], v[sel, b], m[sel, b]), dim=1)
        b2 = mesh_backend.stack((v[sel, b], m[sel, a], m[sel, b]), dim=1)
        score_a = mesh_backend.minimum(min_angle(xy(a1)), min_angle(xy(a2)))
        score_b = mesh_backend.minimum(min_angle(xy(b1)), min_angle(xy(b2)))
        use_b = mesh_backend.unsqueeze(score_b > score_a, -1)
        leaves = mesh_backend.fill_at_indices(
            leaves, o + 1, mesh_backend.where(use_b, b1, a1)
        )
        leaves = mesh_backend.fill_at_indices(
            leaves, o + 2, mesh_backend.where(use_b, b2, a2)
        )
    sel = mesh_backend.flatnonzero(count == 3)
    if sel.shape[0]:
        kids = mesh_backend.concatenate((v[sel], m[sel]), dim=1)[
            :, _CHILD_VERTEX_INDEX_TABLE
        ]
        for t in range(4):
            leaves = mesh_backend.fill_at_indices(leaves, offsets[sel] + t, kids[:, t])

    n_slots = cache.ij.shape[0]
    used_mask = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n_slots,), dtype=mesh_backend.bool),
        leaves.reshape(-1),
        True,
    )
    used = cache.slots[used_mask[cache.slots]]
    remap = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n_slots,), dtype=int64),
        used,
        mesh_backend.arange(used.shape[0], dtype=int64),
    )
    return used, remap[leaves], leaf_origin, origin, remap[v]
