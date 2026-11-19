"""
Total magnification at source-plane points, read off a lens mesh with no lens call.

The magnification of an image is ``1 / |det|`` of the mesh's fitted Jacobian
(``mesh.vertices_jacobian``) interpolated linearly at the image's preimage,
not of the lens's own ``det A``.

Each leaf whose source-plane triangle holds a point
(:func:`~.geometry.contains_half_open`) is one image of it, with the
magnification :func:`hit_magnification` gives; ``mu_tot`` is their sum
(:func:`total_magnification`).
"""

import math
from typing import NamedTuple

from ....backend_obj import ArrayLike, backend
from .mesh_backend import mesh_backend, to_mesh, to_user
from .geometry import area2, contains_half_open, map_chunks
from .index import as_points, index_hits
from .band import _det2, jacobian_det6
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
        ``(L,)`` bool, True where the leaf's six ``det`` (at its three
        vertices and the three edge midpoints, from the mesh's fitted vertex
        Jacobians) are finite and all positive or all negative.
    size: ArrayLike
        ``(L,)`` float64 longest edge of the leaf's source-plane triangle:
        how finely the lens mesh resolves the source plane there.

        *Unit: arcsec*
    """

    area_ratio: ArrayLike
    same_sign: ArrayLike
    size: ArrayLike


def leaf_tables(mesh):
    """
    The :class:`LeafTables` of ``mesh``'s leaves, ``O(L)``.

    ``same_sign`` reads :func:`~.band.jacobian_det6` of the leaf's fitted vertex
    Jacobians, ``mesh.vertices_jacobian``, and is False where any of the six is
    not finite (tested explicitly: six ``+inf`` are not one sign); the lens's
    own ``det A`` is not consulted, nor is :func:`~.band.resolved`. The vertex
    Jacobians themselves are interpolated by :func:`hit_magnification`.
    """
    lens = mesh.vertices_lens[mesh.leaves]
    source = mesh.vertices_source[mesh.leaves]
    det6 = jacobian_det6(mesh.vertices_jacobian[mesh.leaves])
    edge = source[:, [1, 2, 0]] - source
    return LeafTables(
        area_ratio=mesh_backend.abs(area2(lens)) / mesh_backend.abs(area2(source)),
        same_sign=mesh_backend.all(mesh_backend.isfinite(det6), dim=1)
        & (mesh_backend.all(det6 > 0, dim=1) | mesh_backend.all(det6 < 0, dim=1)),
        size=mesh_backend.max(mesh_backend.norm(edge, dim=-1), dim=1),
    )


def hit_magnification(tables, leaf, w, jacobian):
    """
    The magnification of the image each hit stands for.

    Where the six ``det`` of the leaf's fitted vertex Jacobians share one
    strict sign (``tables.same_sign``), ``1 / |det J|`` with ``J`` the
    fitted vertex Jacobians interpolated linearly at the preimage, whose
    barycentric coordinates are ``w`` over its sum, and ``det`` the explicit
    2x2 formula. Elsewhere the leaf's area ratio: there the determinant
    crosses zero inside the leaf, and ``1 / |det|`` would diverge like ``1 /
    distance``, which is not integrable, where the true magnification
    diverges like ``1 / sqrt(distance)``.

    Six ``det`` of one sign do not hold ``det J`` to that sign everywhere
    inside the leaf: ``det`` of the interpolant is quadratic in the
    barycentric coordinates and can dip through zero between its samples.
    So a hit whose ``det J`` is not finite, or is not of the strict sign of
    ``det`` at the leaf's first vertex, takes the area ratio as well. A
    ``det J`` of that sign is kept however small.

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
    jacobian: ArrayLike
        ``(K, 3, 2, 2)`` fitted Jacobian at each hit leaf's vertices
        (``mesh.vertices_jacobian[mesh.leaves[leaf]]``).

    Returns
    -------
    ArrayLike
        ``(K,)`` float64.
    """
    bary = w / mesh_backend.unsqueeze(w[:, 0] + w[:, 1] + w[:, 2], -1)
    det = _det2(mesh_backend.sum(bary[:, :, None, None] * jacobian, dim=1))
    first = _det2(jacobian[:, 0])
    interpolated = (
        tables.same_sign[leaf]
        & mesh_backend.isfinite(det)
        & (((det > 0) & (first > 0)) | ((det < 0) & (first < 0)))
    )
    return mesh_backend.where(
        interpolated, 1.0 / mesh_backend.abs(det), tables.area_ratio[leaf]
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
    # qidx is non-decreasing: a hit's rank is its offset from its point's first.
    # A gather, not expand: this runs on every total_magnification chunk.
    rank = (
        mesh_backend.arange(qidx.shape[0], dtype=mesh_backend.int64, device=device)
        - (mesh_backend.cumsum(counts, dim=0) - counts)[qidx]
    )
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
    mu = hit_magnification(tables, leaf, w, mesh.vertices_jacobian[mesh.leaves[leaf]])
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
    (:func:`hit_magnification`): ``1 / |det|`` of the mesh's fitted vertex
    Jacobians interpolated at the preimage (not the lens's own ``det A``), or
    the leaf's area ratio where the determinant changes sign across the leaf
    or is not finite. A point on an edge two leaves of one sheet share counts once
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
