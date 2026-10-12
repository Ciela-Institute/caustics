"""
Total magnification at source-plane points, read off a lens mesh with no lens call.

Each leaf whose source-plane triangle holds a point
(:func:`~.geometry.contains_half_open`) is one image of it, with the
magnification :func:`hit_magnification` gives; ``mu_tot`` is their sum
(:func:`total_magnification`).
"""

import math
from typing import NamedTuple

from ....backend_obj import ArrayLike, backend
from .mesh_backend import mesh_backend, to_mesh, to_user
from .geometry import area2, contains_half_open, expand, map_chunks
from .index import as_points, index_hits
from .lens_mesh import inside_fov_image


class LeafTables(NamedTuple):
    """
    What :func:`_total_magnification` reads per leaf; :func:`leaf_tables` builds it.

    Parameters
    ----------
    area_ratio: ArrayLike
        ``(L,)`` float64 ``|lens area| / |source area|``: the magnification
        of the leaf's affine map.
    same_sign: ArrayLike
        ``(L,)`` bool, True where ``det A`` at the leaf's three vertices is
        all positive or all negative.
    size: ArrayLike
        ``(L,)`` float64 longest edge of the leaf's source-plane triangle:
        how finely the lens mesh resolves the source plane there.

        *Unit: arcsec*
    """

    area_ratio: ArrayLike
    same_sign: ArrayLike
    size: ArrayLike


def leaf_tables(mesh):
    """The :class:`LeafTables` of ``mesh``'s leaves, ``O(L)``."""
    lens = mesh.vertices_lens[mesh.leaves]
    source = mesh.vertices_source[mesh.leaves]
    det = mesh.vertices_det[mesh.leaves]
    edge = source[:, [1, 2, 0]] - source
    return LeafTables(
        area_ratio=mesh_backend.abs(area2(lens)) / mesh_backend.abs(area2(source)),
        same_sign=mesh_backend.all(det > 0, dim=1) | mesh_backend.all(det < 0, dim=1),
        size=mesh_backend.max(mesh_backend.norm(edge, dim=-1), dim=1),
    )


def hit_magnification(tables, leaf, w, det):
    """
    The magnification of the image each hit stands for.

    Where ``det A`` at the leaf's vertices shares one strict sign, ``1 /
    |det A|`` with ``det A`` interpolated at the preimage, whose barycentric
    coordinates are ``w`` over its sum. Elsewhere the leaf's area ratio:
    there the interpolated ``det A`` crosses zero inside the leaf, and ``1 /
    |det A|`` would diverge like ``1 / distance``, which is not integrable,
    where the true magnification diverges like ``1 / sqrt(distance)``.

    The criterion bounds the area ratio's error: on a quadratic piece the
    affine Jacobian matches the true one at the edge midpoints and is off by
    at most ``4 delta / h`` elsewhere, and a converged leaf has ``delta <
    sigma_min * min_img_sep``, so the relative error is below about ``4
    min_img_sep / h``. It gives no bound for the interpolated part, which
    inverse ray shooting found 6 to 15 times more accurate across ten lens
    models.

    Parameters
    ----------
    tables: LeafTables
    leaf: ArrayLike
        ``(K,)`` int64 leaf of each hit.
    w: ArrayLike
        ``(K, 3)`` :func:`~.geometry.triangle_weights` of each hit.
    det: ArrayLike
        ``(K, 3)`` ``det A`` at each hit leaf's vertices.

    Returns
    -------
    ArrayLike
        ``(K,)`` float64.
    """
    bary = w / mesh_backend.unsqueeze(w[:, 0] + w[:, 1] + w[:, 2], -1)
    interpolated = 1.0 / mesh_backend.abs(mesh_backend.sum(bary * det, dim=1))
    return mesh_backend.where(
        tables.same_sign[leaf], interpolated, tables.area_ratio[leaf]
    )


def _per_point_sum(qidx, values, n):
    """
    Sum of ``values`` per point, ``(n,)``, added in hit order whatever the chunking.

    ``qidx`` is non-decreasing. Hits are added rank by rank -- every point's
    first hit, then every point's second -- so each point's sum takes its
    values in one fixed order and does not depend on which other points
    share the call.
    """
    device = mesh_backend.device(values)
    counts = mesh_backend.bincount(qidx, minlength=n)
    # qidx is non-decreasing, so a hit's place in its point's block is its rank.
    _, rank = expand(counts, qidx.shape[0])
    total = mesh_backend.zeros((n,), dtype=values.dtype, device=device)
    deepest = (
        int(mesh_backend.to_numpy(mesh_backend.max(counts))) if qidx.shape[0] else 0
    )
    for r in range(deepest):
        hit = mesh_backend.flatnonzero(rank == r)
        total = mesh_backend.add_at_indices(total, qidx[hit], values[hit])
    return total


def _total_magnification(mesh, tables, beta):
    """
    ``mu_tot`` at each point, and how finely the lens mesh resolves the source plane there, with no inside test.

    Parameters
    ----------
    mesh: LensMesh
        On ``mesh_backend``.
    tables: LeafTables
        ``leaf_tables(mesh)``.
    beta: ArrayLike
        ``(B, 2)`` float64 source-plane points.

        *Unit: arcsec*

    Returns
    -------
    mu: ArrayLike
        ``(B,)`` float64; 0 where no leaf holds the point, which inside the
        image of the fov boundary happens only where non-finite leaves were
        left out of the index.
    size: ArrayLike
        ``(B,)`` float64 smallest ``tables.size`` among the leaves holding
        each point; ``inf`` where none does.

        *Unit: arcsec*
    """
    n = beta.shape[0]
    device = mesh_backend.device(beta)
    qidx, leaf, w = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, beta)
    triangles = mesh.leaves[leaf]
    keep = mesh_backend.flatnonzero(
        contains_half_open(mesh.vertices_source[triangles], w)
    )
    qidx, leaf, w = qidx[keep], leaf[keep], w[keep]
    mu = hit_magnification(tables, leaf, w, mesh.vertices_det[mesh.leaves[leaf]])
    size = tables.size[leaf]
    order = mesh_backend.lexsort([size, qidx])
    counts = mesh_backend.bincount(qidx, minlength=n)
    first = mesh_backend.cumsum(counts, dim=0) - counts
    held = mesh_backend.flatnonzero(counts > 0)
    smallest = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n,), dtype=mesh_backend.float64, device=device) + math.inf,
        held,
        size[order][first[held]],
    )
    return _per_point_sum(qidx, mu, n), smallest


def total_magnification(bx, by, mesh, *, batch_size=None):
    """
    Total magnification of each source-plane point, from the lens mesh alone.

    The sum, over every leaf whose source-plane triangle holds the point, of
    the magnification of the image it stands for
    (:func:`hit_magnification`): ``1 / |det A|`` interpolated at the
    preimage, or the leaf's area ratio where ``det A`` changes sign across
    the leaf. A point on an edge two leaves of one sheet share counts once
    (:func:`~.geometry.contains_half_open`). No lens call is made.

    Every image of a point lies in the fov only inside the image of the fov
    boundary, of a mesh whose fov cuts no critical curve (see
    :func:`~.lens_mesh.build_closed_lens_mesh`); outside it the sum misses
    images, so it is NaN there. Leaves with a non-finite raytrace or
    Jacobian are not indexed and add nothing: next to a singular center they
    are finest-level leaves with large source images, whose area ratio is
    negligible.

    Parameters
    ----------
    bx, by: ArrayLike
        Source-plane points, any shape, flattened to ``(B,)``.

        *Unit: arcsec*
    mesh: LensMesh
    batch_size: Optional[int]
        Most source points per chunk, bounding memory; the result does not
        depend on it.

    Returns
    -------
    ArrayLike
        ``(B,)`` float64.
    """
    device = backend.device(mesh.vertices_lens)
    mu = _total_magnification_at(*to_mesh((bx, by, mesh)), batch_size=batch_size)
    return to_user(mu, device)


def _total_magnification_at(bx, by, mesh, *, batch_size):
    """:func:`total_magnification` on ``mesh_backend`` arrays."""
    device = mesh_backend.device(mesh.vertices_lens)
    beta = as_points(bx, by, device)
    tables = leaf_tables(mesh)
    mu = map_chunks(
        lambda chunk: _total_magnification(mesh, tables, chunk)[0], beta, batch_size
    )
    return mesh_backend.where(inside_fov_image(mesh, beta), mu, math.nan)
