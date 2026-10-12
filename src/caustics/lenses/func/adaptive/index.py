"""
A multi-level grid index over source-plane triangles, and lookups in it.

Cells come in power-of-two levels. Each triangle is registered at the one
level whose cells match its bounding box, grown by how far the triangle
reaches beyond itself, in every cell that box covers there. A point reads
one cell per occupied level, so the cells it reads are as fine as the
triangles registered in them wherever those crowd: along caustics, and
where outliers stretch the extent.
"""

from typing import NamedTuple

import numpy as np

from ....backend_obj import ArrayLike
from .mesh_backend import mesh_backend
from .geometry import contains, edge_nearest, expand, sorted_find, triangle_weights

# Level-0 cells divide the larger side of the index's extent into 2**KEY_BITS:
# the finest grid whose cell keys, with their level, fit in an int64.
KEY_BITS = 28
# A triangle's box spans at most about this many cells per axis at its level.
CELLS_PER_BOX = 6
# Most (point, candidate) pairs one step of a query holds.
PAIR_CAP = 1 << 20


class MeshIndex(NamedTuple):
    """
    Multi-level CSR index over triangle bounding boxes.

    Level ``l`` has square cells of side ``fine * 2**l``, counted from
    ``lo``; cell ``(ix, iy)`` of it has key ``l * 2**56 + ix * 2**28 + iy``.
    Cell ``keys[u]``'s triangles are ``leaves[offsets[u]:offsets[u + 1]]``,
    ascending.

    Parameters
    ----------
    lo, hi: ArrayLike
        ``(2,)`` float64 corners of the bounding box of every indexed
        triangle, grown.

        *Unit: arcsec*
    fine: ArrayLike
        ``()`` float64 side of the level-0 cells: the larger side of the
        bounding box over ``2**KEY_BITS``.

        *Unit: arcsec*
    levels: ArrayLike
        ``(L,)`` int64 levels holding a triangle, ascending.
    keys: ArrayLike
        ``(U,)`` int64 keys of the cells holding a triangle, ascending.
    offsets: ArrayLike
        ``(U + 1,)`` int64 CSR offsets.
    leaves: ArrayLike
        int64 triangle indices, cell by cell.
    """

    lo: ArrayLike
    hi: ArrayLike
    fine: ArrayLike
    levels: ArrayLike
    keys: ArrayLike
    offsets: ArrayLike
    leaves: ArrayLike


def _level_tables(device):
    """``2.0**l`` ``(KEY_BITS + 1,)`` float64, and the cells per axis ``2**(KEY_BITS - l)`` int64, of every level ``l``."""
    scale = mesh_backend.as_array(
        [2.0**l for l in range(KEY_BITS + 1)], dtype=mesh_backend.float64, device=device
    )
    span = mesh_backend.as_array(
        [1 << (KEY_BITS - l) for l in range(KEY_BITS + 1)],
        dtype=mesh_backend.int64,
        device=device,
    )
    return scale, span


def _cell_of(x, lo, side, top):
    """
    Cell coordinates of points ``x`` ``(..., 2)`` in cells of side ``side``
    counted from ``lo``, clamped to ``[0, top]``.

    ``side`` and ``top`` broadcast against ``x``. :func:`build_index` and
    :func:`index_cells` both locate points here, so they agree exactly.
    """
    u = mesh_backend.long(mesh_backend.floor((x - lo) / side))
    return mesh_backend.minimum(mesh_backend.clamp(u, 0, None), top)


def _cell_key(level, ix, iy):
    """The int64 key of cell ``(ix, iy)`` of ``level``."""
    return level * (1 << (2 * KEY_BITS)) + ix * (1 << KEY_BITS) + iy


def build_index(vertices, triangles, rows, grow=None):
    """
    The :class:`MeshIndex` over the bounding boxes of ``triangles[rows]``, each grown by its ``grow``.

    A box's level is the fewest doublings of ``CELLS_PER_BOX * fine`` that
    reach its longer side, so it spans at most about ``CELLS_PER_BOX``
    cells per axis there; it is registered in every one of them.

    Parameters
    ----------
    vertices: ArrayLike
        ``(V, 2)`` positions.

        *Unit: arcsec*
    triangles: ArrayLike
        ``(T, 3)`` int64 vertex indices.
    rows: ArrayLike
        ``(K,)`` int64 ascending rows of ``triangles`` to index.
    grow: Optional[ArrayLike]
        ``(K,)`` float64, how far each of ``triangles[rows]`` reaches beyond
        itself: its bounding box is grown by it on every side. None grows
        nothing.

        *Unit: arcsec*

    Returns
    -------
    MeshIndex
    """
    device = mesh_backend.device(vertices)
    f64, int64 = mesh_backend.float64, mesh_backend.int64
    tri = mesh_backend.to(vertices[triangles[rows]], dtype=f64)
    if tri.shape[0] == 0:
        return MeshIndex(
            lo=mesh_backend.zeros((2,), dtype=f64, device=device),
            hi=mesh_backend.ones((2,), dtype=f64, device=device),
            fine=mesh_backend.as_array(1.0 / (1 << KEY_BITS), dtype=f64, device=device),
            levels=mesh_backend.zeros((0,), dtype=int64, device=device),
            keys=mesh_backend.zeros((0,), dtype=int64, device=device),
            offsets=mesh_backend.zeros((1,), dtype=int64, device=device),
            leaves=mesh_backend.zeros((0,), dtype=int64, device=device),
        )
    box_lo, box_hi = mesh_backend.min(tri, dim=1), mesh_backend.max(tri, dim=1)
    if grow is not None:
        reach = mesh_backend.unsqueeze(mesh_backend.to(grow, dtype=f64), -1)
        box_lo, box_hi = box_lo - reach, box_hi + reach
    lo, hi = mesh_backend.min(box_lo, dim=0), mesh_backend.max(box_hi, dim=0)
    extent = mesh_backend.max(hi - lo)
    extent = mesh_backend.where(extent > 0, extent, 1.0)  # every box one point
    fine = mesh_backend.clamp(
        extent / (1 << KEY_BITS), float(mesh_backend.finfo(f64).tiny), None
    )

    scale, span = _level_tables(device)
    size = mesh_backend.max(box_hi - box_lo, dim=1) / (CELLS_PER_BOX * fine)
    # The number of powers of two below `size`: ceil(log2(size)), from 0 to KEY_BITS.
    level = mesh_backend.searchsorted(scale[:KEY_BITS], size)
    side = mesh_backend.unsqueeze(fine * scale[level], -1)
    top = mesh_backend.unsqueeze(span[level] - 1, -1)
    i0 = _cell_of(box_lo, lo, side, top)
    i1 = _cell_of(box_hi, lo, side, top)

    # Each box's columns of cells, keyed by their bottom cell, then each column's cells.
    column, dx = expand(i1[:, 0] - i0[:, 0] + 1)
    column_key = _cell_key(level, i0[:, 0], i0[:, 1])[column] + dx * (1 << KEY_BITS)
    cell, dy = expand((i1[:, 1] - i0[:, 1] + 1)[column])
    key = column_key[cell] + dy
    # A stable sort keeps each cell's triangles in ascending row order.
    order = mesh_backend.argsort(key)
    key = key[order]
    total = key.shape[0]
    run = mesh_backend.concatenate(
        (
            mesh_backend.ones((1,), dtype=mesh_backend.bool, device=device),
            key[1:] != key[:-1],
        ),
        dim=0,
    )
    starts = mesh_backend.flatnonzero(run)
    return MeshIndex(
        lo=lo,
        hi=hi,
        fine=fine,
        levels=mesh_backend.unique(level),
        keys=key[starts],
        offsets=mesh_backend.concatenate(
            (starts, mesh_backend.as_array([total], dtype=int64, device=device)),
            dim=0,
        ),
        leaves=rows[column[cell[order]]],
    )


def index_cells(index, beta):
    """
    The cell of each point at every level of ``index``, as CSR ranges into ``index.leaves``.

    A point outside the stored ``[lo, hi]``, NaN included, gets empty
    ranges, as does a cell that holds nothing.

    Parameters
    ----------
    index: MeshIndex
    beta: ArrayLike
        ``(B, 2)`` points.

        *Unit: arcsec*

    Returns
    -------
    start, count: ArrayLike
        ``(B, L)`` int64, one column per entry of ``index.levels``.
    """
    int64 = mesh_backend.int64
    device = mesh_backend.device(beta)
    n_levels = index.levels.shape[0]
    if n_levels == 0:
        empty = mesh_backend.zeros((beta.shape[0], 0), dtype=int64, device=device)
        return empty, empty
    scale, span = _level_tables(device)
    side = (index.fine * scale[index.levels]).reshape(1, n_levels, 1)
    top = (span[index.levels] - 1).reshape(1, n_levels, 1)
    u = _cell_of(mesh_backend.unsqueeze(beta, 1), index.lo, side, top)
    key = _cell_key(mesh_backend.unsqueeze(index.levels, 0), u[..., 0], u[..., 1])
    pos, found = sorted_find(index.keys, key)
    inside = (
        (beta[:, 0] >= index.lo[0])
        & (beta[:, 0] <= index.hi[0])
        & (beta[:, 1] >= index.lo[1])
        & (beta[:, 1] <= index.hi[1])
    )
    found = found & mesh_backend.unsqueeze(inside, -1)
    start = index.offsets[pos]
    count = mesh_backend.where(
        found, index.offsets[pos + 1] - start, mesh_backend.zeros_like(start)
    )
    return start, count


def _chunk_hits(index, vertices, triangles, beta, start, count, total, grow):
    """
    :func:`index_hits` for points ``beta`` whose cells are ``start``, ``count`` ``(B, L)``.

    ``total`` is ``count.sum()``, read on the host by the caller.
    """
    block, within = expand(count.reshape(-1), total)
    cand = index.leaves[start.reshape(-1)[block] + within]
    qidx = block // start.shape[1]
    tri = vertices[triangles[cand]]
    w = triangle_weights(tri, beta[qidx])
    hit = contains(w)
    if grow is not None:
        reach = mesh_backend.flatnonzero(~hit & (grow[cand] > 0))
        dist, _ = edge_nearest(tri[reach], beta[qidx[reach]])
        hit = mesh_backend.fill_at_indices(hit, reach, dist <= grow[cand[reach]])
    hit = mesh_backend.flatnonzero(hit)
    qidx, cand, w = qidx[hit], cand[hit], w[hit]
    # Each level's hits come in ascending order; merge them per point.
    order = mesh_backend.lexsort((cand, qidx))
    return qidx[order], cand[order], w[order]


def index_hits(index, vertices, triangles, beta, grow=None):
    """
    Triangles of ``index`` whose image contains each point, zeros counting as inside, or reaches it.

    One cell per level per point, then :func:`~.geometry.contains` on its
    triangles. With ``grow``, a triangle that does not contain the point is
    a hit when the point lies within its ``grow`` of it
    (:func:`~.geometry.edge_nearest`); the index must have been built with
    a ``grow`` at least as large. Points run in chunks of whole points
    holding at most ``PAIR_CAP`` (point, candidate) pairs, or one point
    holding more, so memory stays bounded and the result does not depend
    on the chunking.

    Parameters
    ----------
    index: MeshIndex
        Built over ``triangles`` by :func:`build_index`.
    vertices: ArrayLike
        ``(V, 2)`` positions ``triangles`` indexes.

        *Unit: arcsec*
    triangles: ArrayLike
        ``(T, 3)`` int64 vertex indices.
    beta: ArrayLike
        ``(B, 2)`` points.

        *Unit: arcsec*
    grow: Optional[ArrayLike]
        ``(T,)`` float64, how far each triangle reaches beyond itself. None
        reaches nothing.

        *Unit: arcsec*

    Returns
    -------
    qidx: ArrayLike
        ``(K,)`` int64 point of each hit, non-decreasing.
    tri: ArrayLike
        ``(K,)`` int64 triangle of each hit, ascending within each point.
    w: ArrayLike
        ``(K, 3)`` :func:`~.geometry.triangle_weights` of each hit.
    """
    int64 = mesh_backend.int64
    device = mesh_backend.device(beta)
    qidx = [mesh_backend.zeros((0,), dtype=int64, device=device)]
    tri = [mesh_backend.zeros((0,), dtype=int64, device=device)]
    w = [mesh_backend.zeros((0, 3), dtype=vertices.dtype, device=device)]
    step = max(1, PAIR_CAP // max(index.levels.shape[0], 1))
    for first in range(0, beta.shape[0], step):
        chunk = beta[first : first + step]
        start, count = index_cells(index, chunk)
        # Running candidate totals, read on the host once per chunk.
        ends = np.cumsum(mesh_backend.to_numpy(mesh_backend.sum(count, dim=1)))
        lo = 0
        while lo < chunk.shape[0]:
            before = int(ends[lo - 1]) if lo else 0
            hi = max(
                lo + 1, int(np.searchsorted(ends, before + PAIR_CAP, side="right"))
            )
            total = int(ends[hi - 1]) - before
            if total:
                found = _chunk_hits(
                    index,
                    vertices,
                    triangles,
                    chunk[lo:hi],
                    start[lo:hi],
                    count[lo:hi],
                    total,
                    grow,
                )
                qidx.append(found[0] + (first + lo))
                tri.append(found[1])
                w.append(found[2])
            lo = hi
    return (
        mesh_backend.concatenate(qidx, dim=0),
        mesh_backend.concatenate(tri, dim=0),
        mesh_backend.concatenate(w, dim=0),
    )


def as_points(bx, by, device):
    """``bx`` and ``by``, flattened, as ``(B, 2)`` float64 points on ``device``."""
    f64 = mesh_backend.float64
    x = mesh_backend.as_array(bx, dtype=f64, device=device).reshape(-1)
    y = mesh_backend.as_array(by, dtype=f64, device=device).reshape(-1)
    return mesh_backend.stack((x, y), dim=-1)
