"""
Every image of each source-plane point, from a lens mesh.

:func:`forward_raytrace` seeds Levenberg-Marquardt from each finite leaf
whose source-plane triangle contains the point, or reaches it
(:func:`mesh_query`, :func:`mesh_seeds`), keeps the roots that converge near
their seed and inside the fov, and merges near-coincident roots
(:func:`dedup_representatives`).
"""

from ....backend_obj import backend
from .mesh_backend import mesh_backend, to_mesh, to_user
from ....utils import batch_lm
from .criterion import affine_error
from .geometry import (
    area2,
    component_labels,
    contains,
    csr_offsets,
    edge_nearest,
    map_chunks,
    sanitize_bary,
    triangle_weights,
)
from .index import as_points, index_hits
from .lens_mesh import inside_fov, inside_fov_image, leaf_grow, make_sampler


def mesh_query(mesh, beta):
    """
    Leaves whose source-plane triangle contains each point, or reaches it.

    A leaf that did not converge, or that shares a vertex with a leaf left
    out of the index, reaches its deviation beyond its triangle
    (:func:`~.lens_mesh.leaf_grow`): near a lens center or a fold its true
    image bulges past the straight-edged one. A point on an edge two leaves
    share returns both, and leaves near a critical curve overlap, so the
    number of hits is not the image count.

    Parameters
    ----------
    mesh: LensMesh
    beta: ArrayLike
        ``(B, 2)`` source-plane points.

        *Unit: arcsec*

    Returns
    -------
    leaf_indices: ArrayLike
        ``(K,)`` int64 leaves, ascending within each point.
    offsets: ArrayLike
        ``(B + 1,)`` int64 CSR offsets into ``leaf_indices``.
    bary: ArrayLike
        ``(K, 3)`` float64 barycentric coordinates, in the simplex, of the
        point of each leaf's triangle nearest the point: the point itself
        where the triangle contains it.
    """
    device = mesh_backend.device(mesh.vertices_lens)
    beta = mesh_backend.as_array(beta, dtype=mesh_backend.float64, device=device)
    qidx, leaves, w = index_hits(
        mesh.index, mesh.vertices_source, mesh.leaves, beta, leaf_grow(mesh)
    )
    tri = mesh.vertices_source[mesh.leaves[leaves]]
    inside = contains(w)
    outside = mesh_backend.flatnonzero(~inside)
    _, found = edge_nearest(tri[outside], beta[qidx[outside]])
    nearest = mesh_backend.fill_at_indices(mesh_backend.zeros_like(w), outside, found)
    bary = mesh_backend.where(
        mesh_backend.unsqueeze(inside, -1), sanitize_bary(w, area2(tri)), nearest
    )
    counts = mesh_backend.long(mesh_backend.bincount(qidx, minlength=beta.shape[0]))
    return leaves, csr_offsets(counts), bary


def mesh_seeds(mesh, leaf_indices, bary):
    """
    Lens-plane preimage of each hit under its leaf's affine map, ``(K, 2)``.

    Inside its leaf, since ``bary`` lies in the simplex, and within about
    the leaf's :func:`~.criterion.affine_error` of the image it
    approximates: below ``min_img_sep`` on a converged leaf, by the
    refinement criterion.

    *Unit: arcsec*
    """
    tri = mesh.vertices_lens[mesh.leaves[leaf_indices]]
    return mesh_backend.sum(tri * mesh_backend.unsqueeze(bary, -1), dim=1)


def near_seed(mesh, idx, seed, root):
    """
    True where each root lies in its seed's leaf, or within ``max(r, min_img_sep)`` of its seed.

    ``r`` is the leaf's :func:`~.criterion.affine_error`, how far its affine
    model's seed can lie from the image it approximates. On a converged leaf
    it is below ``min_img_sep``, so the radius is ``min_img_sep``.

    Parameters
    ----------
    mesh: LensMesh
    idx: ArrayLike
        ``(K,)`` int64 leaf of each seed.
    seed, root: ArrayLike
        ``(K, 2)`` seeds and the roots found from them.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(K,)`` bool.
    """
    origin = mesh.leaf_origin[idx]
    radius = mesh_backend.clamp(
        affine_error(mesh.origin_deviation[origin], mesh.origin_sigma_min[origin]),
        mesh.min_img_sep,
        None,
    )
    tri = mesh.vertices_lens[mesh.leaves[idx]]
    return contains(triangle_weights(tri, root)) | (
        mesh_backend.sum((root - seed) ** 2, dim=-1) <= radius**2
    )


def dedup_block_group(points, residual2, rows, n_blocks, m, tol):
    """
    Connected-component representatives for blocks of exactly ``m`` points.

    :func:`dedup_representatives` calls it once per distinct block size, so
    the ``(n_blocks, m, m)`` intermediate is never padded to the largest block.

    Parameters
    ----------
    points: ArrayLike
        The caller's full point array, shape ``(K, 2)``.

        *Unit: arcsec*
    residual2: ArrayLike
        ``(K,)`` finite squared residual of each point.

        *Unit: arcsec^2*
    rows: ArrayLike
        ``(n_blocks * m,)`` int64 indices into ``points``, block-major. A
        host-side sequence is accepted too; see :func:`dedup_representatives`.
    n_blocks, m: int
        Block count and the common per-block point count.
    tol: float
        Separation below which two points are the same image.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(n_blocks * m,)`` bool, in the order of ``rows``.
    """
    device = mesh_backend.device(points)
    int64 = mesh_backend.int64
    rows = mesh_backend.as_array(rows, dtype=int64, device=device)
    p = points[rows].reshape(n_blocks, m, 2)
    r = residual2[rows].reshape(n_blocks, m)

    delta = mesh_backend.unsqueeze(p, 2) - mesh_backend.unsqueeze(p, 1)
    # Exact on the diagonal: every point is its own neighbour.
    labels = component_labels(mesh_backend.sum(delta * delta, dim=-1) < tol * tol)
    index = mesh_backend.unsqueeze(
        mesh_backend.arange(m, dtype=int64, device=device), 0
    )

    # A slot represents its component when no other slot of it is better: a
    # smaller residual, or the same one at an earlier slot.
    same = mesh_backend.unsqueeze(labels, 2) == mesh_backend.unsqueeze(labels, 1)
    r_i, r_j = mesh_backend.unsqueeze(r, 2), mesh_backend.unsqueeze(r, 1)
    earlier = mesh_backend.unsqueeze(index, 1) < mesh_backend.unsqueeze(index, 2)
    better = same & ((r_j < r_i) | ((r_j == r_i) & earlier))
    return (~mesh_backend.any(better, dim=2)).reshape(-1)


def dedup_representatives(points, residual2, counts, tol):
    """
    One representative per cluster of near-coincident points, within each block.

    Clusters are the connected components of the ``distance < tol`` graph,
    not greedy clusters, so they depend on the point set alone, not its
    order; a pair exactly ``tol`` apart stays distinct. Each cluster is
    represented by its point of smallest residual, ties going to the
    earliest: the earliest root of an image can be the least converged one.
    Blocks are grouped by
    size, and blocks of zero or one point skip the clustering. Block-major
    order is restored by gathering with the inverse permutation -- the
    ``argsort`` of the groups' rows -- since torch and jax resolve a scatter
    to repeated indices differently.

    Parameters
    ----------
    points: ArrayLike
        Shape ``(K, 2)``, laid out block-major: block ``b`` occupies the
        ``counts[b]`` rows following those of blocks ``0 .. b - 1``.

        *Unit: arcsec*
    residual2: ArrayLike
        ``(K,)`` finite squared residual of each point.

        *Unit: arcsec^2*
    counts: ArrayLike
        Shape ``(B,)`` int, with ``counts.sum() == K``. Zero-length blocks are
        allowed.
    tol: float
        Separation below which two points are the same image.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(K,)`` bool, True on exactly one point per cluster.
    """
    device = mesh_backend.device(points)
    counts = mesh_backend.as_array(counts, dtype=mesh_backend.int64, device=device)
    total = int(mesh_backend.to_numpy(mesh_backend.sum(counts)))
    if total == 0:
        return mesh_backend.zeros((0,), dtype=mesh_backend.bool, device=device)

    starts = mesh_backend.cumsum(counts, dim=0) - counts
    row_groups, keep_groups = [], []

    # A block of one point is its own representative.
    singles = mesh_backend.flatnonzero(counts == 1)
    if singles.shape[0] > 0:
        row_groups.append(starts[singles])
        keep_groups.append(
            mesh_backend.ones(
                (singles.shape[0],), dtype=mesh_backend.bool, device=device
            )
        )

    # The rest run grouped by size; there are a handful of distinct sizes.
    distinct = mesh_backend.to_numpy(mesh_backend.unique(counts[counts > 1])).tolist()
    for m in distinct:
        blocks = mesh_backend.flatnonzero(counts == m)
        rows = (
            mesh_backend.unsqueeze(starts[blocks], 1)
            + mesh_backend.unsqueeze(
                mesh_backend.arange(m, dtype=mesh_backend.int64, device=device), 0
            )
        ).reshape(-1)
        row_groups.append(rows)
        keep_groups.append(
            dedup_block_group(points, residual2, rows, blocks.shape[0], m, tol)
        )

    # Every row is in exactly one group, so `perm` is a permutation.
    perm = mesh_backend.concatenate(row_groups, dim=0)
    inverse = mesh_backend.argsort(perm)
    stacked = mesh_backend.concatenate(keep_groups, dim=0)
    return stacked[inverse]


def _block_sums(values, offsets):
    """Sum of ``values`` over each CSR block of ``offsets``."""
    csum = csr_offsets(values)
    return csum[offsets[1:]] - csum[offsets[:-1]]


def _images(mesh, beta, raytrace, tol, lm_map, lm_kwargs, device):
    """Images and their count per point, for one chunk of ``(B, 2)`` points; ``device`` is the lens's."""
    mesh_device = mesh_backend.device(mesh.vertices_lens)
    idx, offsets, bary = mesh_query(mesh, beta)
    seed = mesh_seeds(mesh, idx, bary)
    if seed.shape[0] == 0:
        return seed, mesh_backend.zeros(
            (beta.shape[0],), dtype=mesh_backend.int64, device=mesh_device
        )

    f, f_args = lm_map
    target = mesh_backend.repeat(beta, offsets[1:] - offsets[:-1], axis=0)
    root, _, _ = batch_lm(
        to_user(seed, device), to_user(target, device), f, f_args=f_args, **lm_kwargs
    )
    root = mesh_backend.to(to_mesh(root), device=mesh_device)
    trace = make_sampler(raytrace, None, device)
    residual2 = mesh_backend.sum((trace(root) - target) ** 2, dim=-1)
    converged = residual2 < tol * tol
    keep = converged & near_seed(mesh, idx, seed, root) & inside_fov(mesh, root)
    kept = _block_sums(mesh_backend.long(keep), offsets)
    rows = mesh_backend.flatnonzero(keep)
    root = root[rows]
    unique = dedup_representatives(root, residual2[rows], kept, mesh.min_img_sep)
    return root[unique], _block_sums(mesh_backend.long(unique), csr_offsets(kept))


def forward_raytrace(
    bx, by, raytrace, mesh, *, batch_size=None, residual_tol=1e-6, lm_kwargs=None
):
    """
    Lens-plane positions of every image of each source-plane point.

    Each leaf with a finite raytrace and Jacobian whose source-plane
    triangle contains the point, or, for a leaf that did not converge or
    that shares a vertex with a non-finite one, reaches it
    (:func:`mesh_query`), seeds Levenberg-Marquardt
    (:func:`~caustics.utils.batch_lm`) with the preimage under the leaf's
    affine map. A root is kept when it maps to within ``residual_tol`` of
    the point and lies in its seed's leaf or within ``max(r, min_img_sep)``
    of the seed, ``r`` the leaf's :func:`~.criterion.affine_error`
    (:func:`near_seed`), and lies in the fov: near a fold, a small residual
    alone admits points far from any image. Kept roots closer than
    ``min_img_sep`` are one image, the one of smallest residual.

    On a leaf that did not converge, ``r`` can be far larger than
    ``min_img_sep`` (near a fold it is unbounded), so the residual test is
    then the only filter. Near a critical curve, a stalled solve can leave a
    point within ``residual_tol`` of the source that is not an image, at a
    very large magnification.

    Every image lies in the fov only on a mesh whose fov cuts no critical
    curve (see :func:`~.lens_mesh.build_closed_lens_mesh`); elsewhere an
    image outside the fov is lost silently.

    Parameters
    ----------
    bx, by: ArrayLike
        Source-plane points, any shape, flattened to ``(B,)``.

        *Unit: arcsec*
    raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]
        The raytrace ``mesh`` was built from.
    mesh: LensMesh
    batch_size: Optional[int]
        Most source points per chunk, bounding memory. Image counts do not
        depend on it; positions can move within solver accuracy.
    residual_tol: float
        Largest ``|raytrace(x) - beta|`` of a kept root. It must exceed the
        residual the raytrace can reach: one computed in float32 resolves
        only about ``1e-7 * |x|``, coarser than ``1e-6`` beyond about 10
        arcsec.

        *Unit: arcsec*
    lm_kwargs: Optional[dict]
        Extra keyword arguments for :func:`~caustics.utils.batch_lm`,
        overriding the defaults ``jit=True`` and ``stopping=1e-10``. Its own
        ``stopping``, ``1e-4``, ends a solve with roots that can miss
        ``1e-6``, even well conditioned ones.

    Returns
    -------
    x, y: ArrayLike
        ``(K,)`` image positions, source by source: the ``counts[b]`` images
        of source ``b`` follow those of sources ``0 .. b - 1``.

        *Unit: arcsec*
    counts: ArrayLike
        ``(B,)`` int64 images per source.

    Raises
    ------
    ValueError
        If any source lies outside the image of the mesh's fov boundary
        (:func:`~.lens_mesh.inside_fov_image`), a NaN source included. Where
        the lens is non-finite on the fov boundary, the boundary's image has
        gaps, so sources near them may count as outside, and growing the fov
        does not help.
    """
    return _forward_raytrace_at_user(
        bx,
        by,
        raytrace,
        mesh,
        None,
        batch_size=batch_size,
        residual_tol=residual_tol,
        lm_kwargs=lm_kwargs,
    )


def _forward_raytrace_at_user(
    bx, by, raytrace, mesh, lm_map, *, batch_size, residual_tol, lm_kwargs
):
    """:func:`forward_raytrace` at user-backend arrays; ``lm_map`` is ``(f, f_args)``, or None to build one from ``raytrace``."""
    device = backend.device(mesh.vertices_lens)
    found = _forward_raytrace(
        *to_mesh((bx, by)),
        raytrace,
        to_mesh(mesh),
        device=device,
        batch_size=batch_size,
        residual_tol=residual_tol,
        lm_kwargs=lm_kwargs,
        lm_map=lm_map,
    )
    return to_user(found, device)


def _forward_raytrace(
    bx, by, raytrace, mesh, *, device, batch_size, residual_tol, lm_kwargs, lm_map
):
    """:func:`forward_raytrace` on ``mesh_backend`` arrays; ``device`` is the lens's."""
    mesh_device = mesh_backend.device(mesh.vertices_lens)
    beta = as_points(bx, by, mesh_device)
    outside = int(
        mesh_backend.to_numpy(mesh_backend.sum(~inside_fov_image(mesh, beta)))
    )
    if outside:
        raise ValueError(
            f"{outside} source position(s) lie outside the image of the mesh's fov "
            "boundary, so some of their images can lie outside the fov; grow the "
            "mesh with extend_lens_mesh, or build it with build_closed_lens_mesh."
        )
    tol = float(residual_tol)
    lm_kwargs = {"jit": True, "stopping": 1e-10, **(lm_kwargs or {})}
    if lm_map is None:
        # One map for the whole call, not one per chunk: `batch_lm`'s jitted
        # solve is cached on `f`'s identity, so a fresh closure per chunk
        # would recompile every chunk.
        lm_map = (
            lambda xy: backend.stack(raytrace(xy[..., 0], xy[..., 1]), dim=-1),
            (),
        )
    images, counts = map_chunks(
        lambda chunk: _images(mesh, chunk, raytrace, tol, lm_map, lm_kwargs, device),
        beta,
        batch_size,
    )
    return images[:, 0], images[:, 1], counts
