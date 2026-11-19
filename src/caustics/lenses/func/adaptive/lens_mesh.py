"""
The lens mesh: an adaptive triangulation of the lens plane, carrying the lens map at its vertices.

:func:`build_lens_mesh` refines the lens plane until each leaf's affine
model of the lens map is accurate to ``min_img_sep`` and no critical curve
runs between its samples (:func:`~.criterion.lens_status`), down to a size
floor. :func:`extend_lens_mesh` grows a mesh to a larger fov, reusing its
refinement and the images it stores, and :func:`build_closed_lens_mesh`
grows it until the fov cuts no critical curve and the image of its
boundary crosses no pseudo-caustic. Only the raytrace is called: no lens
Jacobian is evaluated.
"""

import math
from typing import NamedTuple
from warnings import warn

from ....backend_obj import ArrayLike, backend
from .mesh_backend import mesh_backend, to_mesh, to_user
from .geometry import (
    CORNER_SECTOR,
    ROOT_CLASS,
    build_device,
    is_member,
    map_chunks,
    segments_cross,
    sorted_find,
    to_device,
    winding_number,
)
from .lattice import (
    Lattice,
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
from .refine import (
    add_roots,
    cache_lookup,
    close,
    empty_cache,
    empty_store,
    refine,
    resume,
    sample_points,
)
from .index import MeshIndex, build_index
from .criterion import (
    LEAF_CONVERGED,
    LEAF_RAYTRACE_NONFINITE,
    lens_status,
)
from .band import (
    MEASURE_OFFSET,
    CriticalBand,
    build_band,
    in_band,
    jacobian_det6,
    resolved,
)
from .holes import CenterHoles, merge_centers, sample_holes
from .curves import _critical_curves_and_caustics

# Most segment pairs the closed build's crossing test holds at once.
CROSSING_PAIRS = 1 << 18
# Most rows one step of a freeze's status pass holds.
STATUS_ROWS = 1 << 16


class LensMesh(NamedTuple):
    """
    An adaptive triangulation of the lens plane, queryable from the source plane.

    Vertex ``v`` has a lens-plane position ``vertices_lens[v]`` and its image
    ``vertices_source[v]``, so the two embeddings share one topology. Every
    leaf whose origin has a finite raytrace -- not flagged
    ``LEAF_RAYTRACE_NONFINITE`` -- is in ``index``, converged or not. A leaf
    failing the criterion at the finest level straddles a fold, a critical
    curve or a lens center the mesh cannot resolve; its source triangle is
    queried grown by its deviation (:func:`leaf_grow`), and
    ``forward_raytrace`` accepts its roots further from their seeds. The
    source triangle of a leaf sharing a vertex with one left out of the
    index is grown too, converged or not. The pre-closure leaves -- the
    origins -- are kept with their level, class and status, which is what
    an extension refines from.

    No lens Jacobian is evaluated: every Jacobian on the mesh is a fit's.
    The quadratic map through an origin's six images has a Jacobian at each
    of its corners (:func:`~.criterion.fitted_jacobian`). The holders of a
    vertex are the origins with it as one of their three corners, six at an
    interior vertex, three at a hanging node and fewer on the fov boundary,
    and the vertex carries the mean of their fits there. The mean is a float
    sum, so its last bits depend on the order of its terms: two builds by
    the same route agree bit for bit, and an extension and a fresh build of
    the same mesh agree to rounding.

    Parameters
    ----------
    lattice: Lattice
        The lattice every vertex lives on. ``lattice.level - 1`` is the
        finest level, ``lattice.fov`` the fov and ``lattice.init_res`` the
        level-0 cells per axis.
    vertices_ij: ArrayLike
        ``(V, 2)`` int64 lattice coordinates, in ascending lattice-key order.
    vertices_lens: ArrayLike
        ``(V, 2)`` lens-plane positions.

        *Unit: arcsec*
    vertices_source: ArrayLike
        ``(V, 2)`` their images.

        *Unit: arcsec*
    vertices_jacobian: ArrayLike
        ``(V, 2, 2)`` float64 the mean of each vertex's holders' fitted
        Jacobians at it, as above; NaN where a holder has a non-finite
        image.
    leaves: ArrayLike
        ``(L, 3)`` int64 vertex indices of the closed, conforming leaves,
        positively oriented in the lens plane.
    leaf_origin: ArrayLike
        ``(L,)`` int64 origin of each leaf, non-decreasing.
    origin_leaves: ArrayLike
        ``(N, 3)`` int64 vertex indices of the pre-closure leaves.
    origin_level: ArrayLike
        ``(N,)`` int64 refinement level of each origin.
    origin_cls: ArrayLike
        ``(N,)`` int64 orientation class of each origin, see
        :func:`~.geometry.child_matrix_tables`.
    origin_status: ArrayLike
        ``(N,)`` int64 ``LEAF_*`` bitmask: ``LEAF_CONVERGED`` below the
        finest level, where every leaf passed the criterion, and the
        criterion's verdict at it.
    origin_deviation: ArrayLike
        ``(N,)`` float64 largest midpoint deviation of each origin, from
        :func:`~.criterion.lens_status`.

        *Unit: arcsec*
    origin_sigma_min: ArrayLike
        ``(N,)`` float64 the smaller of the smallest singular value of each
        origin's four children's affine maps and that of its fitted
        Jacobian at its six samples (:func:`~.criterion.lens_status`). With
        ``origin_deviation`` it gives :func:`~.criterion.affine_error`,
        below ``min_img_sep`` on every converged origin. That bounds how far
        a seed of the origin can lie from its image where the lens's own
        smallest singular value on the leaf is at least the fit's. Where the
        lens's Jacobian jumps inside a leaf it can be less: across the kink
        of a test map, converged leaves fit 1.0 where the lens's own falls
        to 0.6.
    index: MeshIndex
        Over the source-plane images of the leaves whose origin is finite,
        each grown by :func:`leaf_grow`.
    critical_band: CriticalBand
        The finest-level origins across which ``det`` of the mesh's
        Jacobian changes sign, with that ``det`` at their samples: from
        ``vertices_jacobian``, or measured from the raytrace at the samples
        of a leaf that is not :func:`~.band.resolved`. It also holds every
        measured sample.
    holes: CenterHoles
        Holes around the lens centers, with their hole curves.
    min_img_sep: float
        The tolerance the mesh was refined to, half the one requested.

        *Unit: arcsec*
    """

    lattice: Lattice
    vertices_ij: ArrayLike
    vertices_lens: ArrayLike
    vertices_source: ArrayLike
    vertices_jacobian: ArrayLike
    leaves: ArrayLike
    leaf_origin: ArrayLike
    origin_leaves: ArrayLike
    origin_level: ArrayLike
    origin_cls: ArrayLike
    origin_status: ArrayLike
    origin_deviation: ArrayLike
    origin_sigma_min: ArrayLike
    index: MeshIndex  # type: ignore[assignment]  # shadows tuple.index
    critical_band: CriticalBand
    holes: CenterHoles
    min_img_sep: float


def make_sampler(raytrace, device, batch_size=None):
    """
    The lens map as ``(N, 2) -> (N, 2)`` float64 images.

    ``raytrace`` receives float64 coordinates on ``device``, in the user's
    backend, padded to ``backend.padded_size(N, batch_size)`` rows by
    repeating the last. Float64 matters: the criterion compares midpoint
    deviations of ``O(fov)`` quantities, which cancel to exactly zero below
    about ``sqrt(8 * eps * fov)`` and would read as converged. The images
    are cast to float64 and come back on the positions' device, one row per
    position. A raytrace that computes in float32 still limits the mesh:
    its rounding enters the fitted Jacobian divided by the leaf's edge, and
    on small enough leaves outweighs the change of ``det A`` across them, so
    the critical curves break into open pieces. An SIS of Einstein radius 1
    computed in float32, meshed over a fov of 4 from ``init_res`` 8, traces
    one closed critical curve at a requested ``min_img_sep`` of 0.003 and
    tens of open pieces at 0.002, where its finest leaves' shorter edges
    halve from about ``1e-3`` to ``5e-4``. Raytrace in float64 to refine
    further.
    """
    f64 = backend.float64

    def sample(xy):
        n = xy.shape[0]
        size = backend.padded_size(n, batch_size)
        padded = mesh_backend.pad_rows(xy, size) if size > n else xy
        x = backend.as_array(to_user(padded[:, 0]), dtype=f64, device=device)
        y = backend.as_array(to_user(padded[:, 1]), dtype=f64, device=device)
        out = backend.stack([backend.to(c, dtype=f64) for c in raytrace(x, y)], dim=-1)
        out = mesh_backend.to(to_mesh(out), device=mesh_backend.device(xy))
        return out[:n] if size > n else out

    return sample


def _add_corner_fits(total, vertices, cls, corners):
    """
    ``total`` ``(V, 2, 2)`` with each of ``corners`` added at its vertex, one add per sector.

    Each corner covers one of six sectors around its vertex
    (:data:`~.geometry.CORNER_SECTOR`), and triangles that tile the plane
    cover each sector of a vertex at most once, so the corners of one sector
    have distinct vertices: ``add_at_indices``, which under torch does not
    accumulate a repeated index, sees none.

    Parameters
    ----------
    total: ArrayLike
        ``(V, 2, 2)`` float64, written in place.
    vertices: ArrayLike
        ``(n, 3)`` int64 vertex indices of ``n`` non-overlapping triangles.
    cls: ArrayLike
        ``(n,)`` int64 their orientation classes.
    corners: ArrayLike
        ``(n, 3, 2, 2)`` float64 the value at each corner.
    """
    sector = CORNER_SECTOR[cls]
    for s in range(6):
        at = sector == s
        total = mesh_backend.add_at_indices(total, vertices[at], corners[at])
    return total


def finite_leaves(mesh):
    """True where a leaf's origin has a finite raytrace, ``(L,)``: the leaves ``mesh.index`` holds."""
    return (mesh.origin_status[mesh.leaf_origin] & LEAF_RAYTRACE_NONFINITE) == 0


def leaf_grow(mesh):
    """
    How far each leaf's source-plane triangle reaches beyond itself when queried, ``(L,)``.

    A leaf reaches its origin's largest midpoint deviation -- about how far
    the true image of one of its edges bulges from the straight one --
    where its origin did not converge, or where it shares a vertex with a
    leaf left out of the index (:func:`finite_leaves`). Elsewhere a
    converged leaf reaches no further than its triangle: its image bulges
    past an edge into the indexed triangle across it. Beside a leaf left
    out, no indexed triangle lies there, and next to a lens center steeper
    than isothermal ``sigma_min`` is large enough that a leaf converges
    with a deviation far above ``min_img_sep``. :func:`_mesh_index` and
    :func:`~.images.mesh_query` both read it, so the index registers each
    leaf as far as a query reaches.

    *Unit: arcsec*
    """
    left_out = mesh.leaves[mesh_backend.flatnonzero(~finite_leaves(mesh))]
    at_left_out = mesh_backend.fill_at_indices(
        mesh_backend.zeros(
            (mesh.vertices_lens.shape[0],),
            dtype=mesh_backend.bool,
            device=mesh_backend.device(mesh.leaves),
        ),
        left_out.reshape(-1),
        True,
    )
    beside = mesh_backend.any(at_left_out[mesh.leaves], dim=1)
    status = mesh.origin_status[mesh.leaf_origin]
    return mesh_backend.where(
        (status == LEAF_CONVERGED) & ~beside,
        0.0,
        mesh.origin_deviation[mesh.leaf_origin],
    )


def _lens_split(h0, min_img_sep):
    """The lens criterion as :func:`~.refine.refine`'s ``split``."""

    def split(ij, values6, cls, level):
        status, _, _, _ = lens_status(values6, cls, level, h0, min_img_sep)
        return status != LEAF_CONVERGED

    return split


def _warn_depth_limited(h0, min_img_sep, max_level):
    request = f"min_img_sep={2 * min_img_sep:g} arcsec (refined internally to {min_img_sep:g})"
    warn_depth_limited("Lens mesh", request, h0, min_img_sep, max_level)


def build_lens_mesh(
    raytrace,
    fov,
    init_res,
    min_img_sep,
    max_depth=25,
    *,
    x0=0.0,
    y0=0.0,
    centers=None,
    device=None,
    batch_size=None,
):
    """
    Build an adaptive mesh of the lens plane.

    Both triangles of every cell of an ``init_res x init_res`` grid over the
    square fov are refined until each leaf passes
    :func:`~.criterion.lens_status` or reaches the finest level, where the
    longest leaf edge first falls to ``min_img_sep / 2`` -- capped at
    ``max_depth``. The leaves are then 2:1 balanced, closed and indexed. The
    criterion samples six points per triangle, so ``init_res`` must resolve
    the smallest structure of the lens.

    Only ``raytrace`` is called; no lens Jacobian is evaluated. Each
    triangle is judged on the quadratic map through its six images, each
    vertex carries the mean of the Jacobians of the fits of the origins
    holding it (:class:`LensMesh`), and the critical band is every
    finest-level leaf across which ``det`` of that Jacobian changes sign
    (:class:`~.band.CriticalBand`). Close to a singular center, or a core
    smaller than a leaf, the fit cannot tell the sign of ``det A``. A leaf
    is not :func:`~.band.resolved` where two of its vertex Jacobians differ
    by :data:`~.band.RESOLVED_SPREAD` or more in Frobenius norm, or one is
    NaN. At the six samples of such a finest-level leaf with finite images
    the band reads instead a ``det A`` measured by central differences from
    four more raytraces each, at ``x +- d`` and ``y +- d`` with ``d``
    :data:`~.band.MEASURE_OFFSET` times the mesh's ``min_img_sep``, half the
    one requested. The resolution test decides that and nothing else. A
    sample within a few ``d`` of a singular center gets a wrong ``det A``;
    the hole of a center declared in ``centers`` drops the crossings it
    makes. A singular center where ``raytrace`` returns a finite value --
    ``EPL.raytrace`` returns the center itself there -- must be declared:
    undeclared, a sample on it or within ``d`` of it is measured across the
    singularity, and the mesh can trace a spurious closed curve round it
    with a large caustic. :meth:`Lens.build_closed_lens_mesh
    <caustics.lenses.base.Lens.build_closed_lens_mesh>` declares the center
    of every component of the lens itself. The offset is limited by
    rounding at very small leaves: by estimate the measurement is sound down
    to a mesh ``min_img_sep`` of about ``1e-5`` and fails near ``1e-6``.

    A sign change of ``det A`` lying exactly along mesh lines, where the
    lens's Jacobian jumps while its map stays continuous, is not seen: each
    triangle on either side fits its own smooth branch. For a pixelated
    lens, offset the fov from the pixel grid.

    Parameters
    ----------
    raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]
        ``raytrace(x, y) -> (bx, by)`` on 1-D arrays, e.g. ``lens.raytrace``.
    fov: float
        Side of the square lens-plane domain.

        *Unit: arcsec*
    init_res: int
        Level-0 cells per axis.
    min_img_sep: float
        Requested lens-plane tolerance. The mesh refines to half of it, and
        stores that.

        *Unit: arcsec*
    max_depth: int
        Hard cap on the refinement level; a warning is raised when it binds.
    x0, y0: float
        Center of the domain.

        *Unit: arcsec*
    centers: Optional[ArrayLike]
        ``(S, 2)`` positions where the lens map may jump, such as the center
        of a singular isothermal profile or a point mass. Each gets a hole
        of radius ``min_img_sep / 2``, merged where holes overlap, whose
        boundary is traced so that critical curves can be re-joined around
        it. A singular center where ``raytrace`` returns a finite value must
        be among them, as above.

        *Unit: arcsec*
    device: Optional
        Device the lens lives on: ``raytrace`` receives coordinates there,
        and the mesh is stored there.
    batch_size: Optional[int]
        Most points per ``raytrace`` call.

    Returns
    -------
    LensMesh
    """
    mesh = _build_lens_mesh(
        raytrace,
        *to_mesh((fov, init_res, min_img_sep, max_depth)),
        x0=to_mesh(x0),
        y0=to_mesh(y0),
        centers=to_mesh(centers),
        device=device,
        batch_size=batch_size,
    )
    return to_user(mesh, device)


def _build_lens_mesh(
    raytrace,
    fov,
    init_res,
    min_img_sep,
    max_depth,
    *,
    x0,
    y0,
    centers,
    device,
    batch_size,
):
    """:func:`build_lens_mesh` on ``mesh_backend`` arrays, on the build device."""
    min_img_sep = min_img_sep / 2
    h0 = fov / init_res
    max_level = min(int(max_depth), depth_floor(h0, min_img_sep))
    check_lattice_keys(
        init_res, max_level, "Raise min_img_sep, lower max_depth, or lower init_res."
    )
    _warn_depth_limited(h0, min_img_sep, max_level)
    lat = make_lattice(fov, x0, y0, init_res, max_level + 1)
    hole_centers, hole_radius = merge_centers(centers, min_img_sep)
    sample = make_sampler(raytrace, device, batch_size)
    holes = sample_holes(
        sample,
        hole_centers,
        hole_radius,
        min_img_sep,
        batch_size,
    )
    ij, cls = initial_triangles(init_res, lat.level, ROOT_CLASS)
    cache, store, rows = add_roots(
        empty_cache(2), empty_store(), lat, ij, cls, sample, batch_size
    )
    return _freeze(lat, cache, store, rows, sample, min_img_sep, holes, batch_size)


def _six_keys(lat, ij):
    """
    The lattice keys of triangles' six samples, ``(n, 6)``: the three vertices, then the three edge midpoints.

    ``ij`` is ``(n, 3, 2)`` int64 vertex lattice coordinates; midpoint
    ``m_i`` is opposite vertex ``i`` (:func:`~.lattice.midpoint_ij`).
    """
    return mesh_backend.concatenate(
        (lattice_key(lat, ij), lattice_key(lat, midpoint_ij(ij))), dim=1
    )


def _six_values(lat, cache, store, rows, table_keys, table_values):
    """The values at store ``rows``' six samples, ``(n, 6, C)``, read from ``table_keys``, ascending, and ``table_values``."""
    keys6 = _six_keys(lat, cache.ij[store.v[rows]])
    return table_values[mesh_backend.searchsorted(table_keys, keys6)]


def _status_rows(lat, cache, store, rows, table_keys, table_values, min_img_sep):
    """
    :func:`~.criterion.lens_status` of store ``rows``, with their corner fits.

    Each row's six images are read from ``table_keys``, ascending, and
    ``table_values``, ``(T, 2)`` (:func:`_six_values`).

    Returns
    -------
    status, deviation, sigma_min: ArrayLike
        ``(n,)`` each, as :func:`~.criterion.lens_status` gives them.
    corner_jacobian: ArrayLike
        ``(n, 3, 2, 2)`` float64, the fit's Jacobian at each row's vertices,
        in the row's own order, as :func:`~.criterion.lens_status` gives it.
    """
    beta6 = _six_values(lat, cache, store, rows, table_keys, table_values)
    return lens_status(beta6, store.cls[rows], store.level[rows], lat.h0, min_img_sep)


def _freeze(lat, cache, store, rows, sample, min_img_sep, holes, batch_size, seed=None):
    """
    Refine from store ``rows``, then close and index the result, finishing its finest level.

    :func:`~.refine.refine` splits by :func:`_lens_split` down to the
    lattice's finest level. Its state is held only while :func:`_finish`
    reads it, so that it, and everything the status pass built, is freed
    before the index, the largest step, is built. ``seed`` is
    :func:`_finish`'s.
    """
    split = _lens_split(lat.h0, min_img_sep)
    mesh = _finish(
        lat,
        *refine(cache, store, rows, lat, sample, split, lat.level - 1, batch_size),
        sample,
        min_img_sep,
        holes,
        batch_size,
        seed,
    )
    return mesh._replace(index=_mesh_index(mesh))


def _mesh_index(mesh):
    """``mesh.index``: over the source-plane images of the leaves whose origin is finite, each grown by :func:`leaf_grow`."""
    indexed = mesh_backend.flatnonzero(finite_leaves(mesh))
    return build_index(
        mesh.vertices_source, mesh.leaves, indexed, leaf_grow(mesh)[indexed]
    )


def _finish(lat, cache, store, sample, min_img_sep, holes, batch_size, seed):
    """
    Close a refinement and finish its finest level: its :class:`LensMesh`, with no index yet.

    Each origin not carried over in ``seed`` gets its status, deviation and
    smallest singular value from :func:`~.criterion.lens_status` on its six
    images: below the finest level its midpoints are in the cache,
    ``refine`` having tested and passed it; at the finest level they are
    sampled here, with ``sample``, and never cached. The origins are tested
    in canonical order, in steps of at most :data:`STATUS_ROWS` rows.

    ``vertices_jacobian``, the mean of :class:`LensMesh`, is a sum by
    vertex over the count of holders. Each step of the status pass adds its
    origins' corner fits to the sum (:func:`_add_corner_fits`); an origin
    with a non-finite image adds nothing, and the vertices it holds are
    NaN. The sum is a float sum taken in canonical origin order, so a fresh
    build's mean depends on the mesh and on :data:`STATUS_ROWS` alone, not
    on the order refinement made the rows. A vertex held by no new origin
    and by no old one split here keeps the mean ``seed`` stored, bit for
    bit. At any other old vertex the sum starts as that mean times its old
    count of holders, less the fits of the old origins split here that held
    it, recomputed from the cache, where splitting put their midpoints, also
    in steps of :data:`STATUS_ROWS`.

    With every mean final, the finest-level origins, old and new, are
    taken in steps of at most :data:`STATUS_ROWS`. The six samples of each
    one that is not :func:`~.band.resolved` and has no non-finite image are
    measured (:func:`_measured_det`), each once; a sample ``seed``'s table
    holds keeps its stored value. Then each origin's six ``det A`` are
    :func:`~.band.jacobian_det6` of its vertex Jacobians, with the measured
    value at every measured sample, and the band is the origins
    :func:`~.band.in_band` holds for; status does not depend on it. Its
    images are read from the table the status pass read. An old origin new
    to the band has midpoints that table lacks: those alone are raytraced,
    with ``sample``, and read from a table of their own.

    ``sample`` is :func:`make_sampler`'s. ``seed`` is ``(status,
    deviation, sigma_min, mid_keys, mid_values, vertices_jacobian,
    measured_keys, measured_det)`` from :func:`_seed`: the statuses,
    deviations and smallest singular values of the store's first rows, the
    old band's midpoint samples and their images, read here rather than
    sampled again, the old vertices' Jacobians, by cache slot, and the old
    measured samples, by ascending key; None for a fresh build.
    """
    int64, f64 = mesh_backend.int64, mesh_backend.float64
    used, leaves, leaf_origin, origin, origin_leaves = close(lat, cache, store)
    n_rows = store.v.shape[0]
    if seed is None:
        seed = (
            mesh_backend.zeros((0,), dtype=int64),
            mesh_backend.zeros((0,), dtype=f64),
            mesh_backend.zeros((0,), dtype=f64),
            mesh_backend.zeros((0,), dtype=int64),
            mesh_backend.zeros((0, 2), dtype=f64),
            mesh_backend.zeros((0, 2, 2), dtype=f64),
            mesh_backend.zeros((0,), dtype=int64),
            mesh_backend.zeros((0,), dtype=f64),
        )
    (
        seed_status,
        seed_deviation,
        seed_sigma_min,
        mid_keys,
        mid_values,
        seed_jacobian,
        seed_measured_keys,
        seed_measured_det,
    ) = seed
    n_seed = seed_status.shape[0]
    n_new = n_rows - n_seed
    status = mesh_backend.concatenate(
        (seed_status, mesh_backend.zeros((n_new,), dtype=int64)), dim=0
    )
    deviation = mesh_backend.concatenate(
        (seed_deviation, mesh_backend.zeros((n_new,), dtype=f64)), dim=0
    )
    sigma_min = mesh_backend.concatenate(
        (seed_sigma_min, mesh_backend.zeros((n_new,), dtype=f64)), dim=0
    )
    is_young = origin >= n_seed
    young = origin[mesh_backend.flatnonzero(is_young)]
    fresh = young[mesh_backend.flatnonzero(store.level[young] == lat.level - 1)]

    mid = lattice_key(lat, midpoint_ij(cache.ij[store.v[fresh]]))
    todo = mesh_backend.unique(mid.reshape(-1))
    todo = todo[~is_member(mid_keys, todo)]
    if todo.shape[0]:
        xy = lattice_xy(lat, lattice_ij_from_key(lat, todo))
        new_values = sample_points(xy, sample, batch_size)
    else:
        new_values = mesh_backend.zeros((0, 2), dtype=f64)
    table_keys = mesh_backend.concatenate((cache.keys, mid_keys, todo), dim=0)
    table_values = mesh_backend.concatenate(
        (cache.values[cache.slots], mid_values, new_values), dim=0
    )
    order = mesh_backend.argsort(table_keys)
    table_keys, table_values = table_keys[order], table_values[order]

    origin_of_row = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n_rows,), dtype=int64) - 1,
        origin,
        mesh_backend.arange(origin.shape[0], dtype=int64),
    )
    n_vertices = used.shape[0]
    # The old vertices are the cache's first slots (resume), and the old
    # origins the store's first rows; a split one stays, not valid.
    n_old = seed_jacobian.shape[0]
    old = mesh_backend.flatnonzero(used < n_old)
    vertex_of_old_slot = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n_old,), dtype=int64), used[old], old
    )
    split = mesh_backend.flatnonzero(~store.valid[:n_seed])
    split_vertices = vertex_of_old_slot[store.v[split]]
    # Holders per vertex: the old origins', less the split ones', plus the
    # new ones'. A vertex no split or new origin holds keeps its mean.
    old_held = mesh_backend.bincount(
        vertex_of_old_slot[store.v[:n_seed]].reshape(-1), minlength=n_vertices
    )
    split_held = mesh_backend.bincount(split_vertices.reshape(-1), minlength=n_vertices)
    young_held = mesh_backend.bincount(
        origin_leaves[mesh_backend.flatnonzero(is_young)].reshape(-1),
        minlength=n_vertices,
    )
    held = old_held - split_held + young_held
    changed = (young_held > 0) | (split_held > 0)
    stored = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n_vertices, 2, 2), dtype=f64),
        old,
        seed_jacobian[used[old]],
    )
    # An old NaN mean enters the sum here. It can only be at a vertex the
    # non-finite mask below sets to NaN anyway: a non-finite origin sits at
    # the finest level and is never split, so it still holds the vertex.
    total = stored * old_held.reshape(-1, 1, 1)
    for lo in range(0, split.shape[0], STATUS_ROWS):
        rows = split[lo : lo + STATUS_ROWS]
        beta6 = _six_values(lat, cache, store, rows, table_keys, table_values)
        *_, corners = lens_status(
            beta6, store.cls[rows], store.level[rows], lat.h0, min_img_sep
        )
        total = _add_corner_fits(
            total, split_vertices[lo : lo + STATUS_ROWS], store.cls[rows], -corners
        )

    def status_step(rows):
        nonlocal total
        *verdict, corners = _status_rows(
            lat, cache, store, rows, table_keys, table_values, min_img_sep
        )
        finite = mesh_backend.flatnonzero((verdict[0] & LEAF_RAYTRACE_NONFINITE) == 0)
        adds = rows[finite]
        total = _add_corner_fits(
            total,
            origin_leaves[origin_of_row[adds]],
            store.cls[adds],
            corners[finite],
        )
        return tuple(verdict)

    young_status, young_deviation, young_sigma_min = map_chunks(
        status_step, young, STATUS_ROWS
    )
    status = mesh_backend.fill_at_indices(status, young, young_status)
    deviation = mesh_backend.fill_at_indices(deviation, young, young_deviation)
    sigma_min = mesh_backend.fill_at_indices(sigma_min, young, young_sigma_min)

    nonfinite = mesh_backend.flatnonzero(
        (status[origin] & LEAF_RAYTRACE_NONFINITE) != 0
    )
    bad = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n_vertices,), dtype=mesh_backend.bool),
        origin_leaves[nonfinite].reshape(-1),
        True,
    )
    jacobian = mesh_backend.where(
        changed.reshape(-1, 1, 1), total / held.reshape(-1, 1, 1), stored
    )
    jacobian = mesh_backend.where(bad.reshape(-1, 1, 1), math.nan, jacobian)

    def keys6_of(k):
        """Lattice keys of origins ``k``'s six samples, ``(n, 6)``."""
        return _six_keys(lat, cache.ij[store.v[origin[k]]])

    def unresolved_step(k):
        """The keys of the six samples of those of origins ``k`` measured, ``(6 n,)``."""
        finite = (status[origin[k]] & LEAF_RAYTRACE_NONFINITE) == 0
        unresolved = ~resolved(jacobian[origin_leaves[k]])
        return keys6_of(k[mesh_backend.flatnonzero(finite & unresolved)]).reshape(-1)

    # Ascending origins, so the band comes out in origin order.
    finest = mesh_backend.flatnonzero(store.level[origin] == lat.level - 1)
    measured_keys = mesh_backend.unique(
        map_chunks(unresolved_step, finest, STATUS_ROWS)
    )
    measured_det = _measured_det(
        lat,
        measured_keys,
        seed_measured_keys,
        seed_measured_det,
        sample,
        min_img_sep,
        batch_size,
    )

    def band_step(k):
        """Origins ``k`` in the band, their six det and keys, their images, and which the table holds."""
        det6 = jacobian_det6(jacobian[origin_leaves[k]])
        keys6 = keys6_of(k)
        if measured_keys.shape[0]:
            at, measured = sorted_find(measured_keys, keys6)
            det6 = mesh_backend.where(measured, measured_det[at], det6)
        keep = mesh_backend.flatnonzero(in_band(det6))
        keys6 = keys6[keep]
        pos, found = sorted_find(table_keys, keys6)
        return k[keep], det6[keep], keys6, table_values[pos], found

    band_leaves, band_det6, band_keys6, band_source6, found = map_chunks(
        band_step, finest, STATUS_ROWS
    )
    # The midpoints of an old origin new to the band, which the table lacks.
    missing = mesh_backend.flatnonzero(~found.reshape(-1))
    if missing.shape[0]:
        missing_keys = band_keys6.reshape(-1)[missing]
        side_keys = mesh_backend.unique(missing_keys)
        side_values = sample_points(
            lattice_xy(lat, lattice_ij_from_key(lat, side_keys)), sample, batch_size
        )
        band_source6 = mesh_backend.fill_at_indices(
            band_source6.reshape(-1, 2),
            missing,
            side_values[mesh_backend.searchsorted(side_keys, missing_keys)],
        ).reshape(-1, 6, 2)
    band = build_band(
        lat,
        band_leaves,
        band_keys6,
        band_source6,
        band_det6,
        lattice_ij_from_key(lat, measured_keys),
        measured_det,
    )

    mesh = LensMesh(
        lattice=lat,
        vertices_ij=cache.ij[used],
        vertices_lens=lattice_xy(lat, cache.ij[used]),
        vertices_source=cache.values[used],
        vertices_jacobian=jacobian,
        leaves=leaves,
        leaf_origin=leaf_origin,
        origin_leaves=origin_leaves,
        origin_level=store.level[origin],
        origin_cls=store.cls[origin],
        origin_status=status[origin],
        origin_deviation=deviation[origin],
        origin_sigma_min=sigma_min[origin],
        index=None,
        critical_band=band,
        holes=holes,
        min_img_sep=float(min_img_sep),
    )
    return mesh


def _measured_det(lat, keys, seed_keys, seed_det, sample, min_img_sep, batch_size):
    """
    ``det A`` measured at lattice ``keys``: read from a seed's table where it holds the key, raytraced here otherwise.

    At a sample ``x, y`` the images of ``x - d, x + d`` at ``y`` and of
    ``y - d, y + d`` at ``x`` give the Jacobian by central differences, with
    ``d = MEASURE_OFFSET * min_img_sep``, and the det its determinant; NaN
    where one of the four images is non-finite. All the samples raytraced
    go in one :func:`~.refine.sample_points` call; none when every one is
    in ``seed_keys``. A measured value depends on the sample's position
    alone.

    Parameters
    ----------
    keys: ArrayLike
        ``(K,)`` int64 lattice keys.
    seed_keys, seed_det: ArrayLike
        ``(J,)`` int64 ascending keys and ``(J,)`` float64 their measured
        ``det A``.
    sample:
        :func:`make_sampler`'s.

    Returns
    -------
    ArrayLike
        ``(K,)`` float64.
    """
    f64 = mesh_backend.float64
    pos, found = sorted_find(seed_keys, keys)
    kept = mesh_backend.flatnonzero(found)
    new = mesh_backend.flatnonzero(~found)
    det = mesh_backend.fill_at_indices(
        mesh_backend.zeros(
            (keys.shape[0],), dtype=f64, device=mesh_backend.device(keys)
        ),
        kept,
        seed_det[pos[kept]],
    )
    if new.shape[0] == 0:
        return det
    xy = lattice_xy(lat, lattice_ij_from_key(lat, keys[new]))
    d = MEASURE_OFFSET * float(min_img_sep)
    device = mesh_backend.device(xy)
    dx = mesh_backend.as_array([d, 0.0], dtype=f64, device=device)
    dy = mesh_backend.as_array([0.0, d], dtype=f64, device=device)
    n = xy.shape[0]
    beta = sample_points(
        mesh_backend.concatenate((xy - dx, xy + dx, xy - dy, xy + dy), dim=0),
        sample,
        batch_size,
    ).reshape(4, n, 2)
    jx = (beta[1] - beta[0]) / (2 * d)
    jy = (beta[3] - beta[2]) / (2 * d)
    measured = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
    finite = mesh_backend.all(
        mesh_backend.all(mesh_backend.isfinite(beta), dim=2), dim=0
    )
    return mesh_backend.fill_at_indices(
        det, new, mesh_backend.where(finite, measured, math.nan)
    )


def _seed(mesh, lat, k):
    """
    :func:`~.refine.refine`'s state, and :func:`_freeze`'s ``seed``, from a mesh on its lattice grown by ``k``.

    The cache and store are :func:`~.refine.resume`'s, with the vertices'
    images as the cache's values. The measured samples are shifted by
    ``k << lat.level``, as ``resume`` shifts the vertices, and keyed on the
    grown lattice, which keeps their order.
    """
    mesh = to_device(mesh, build_device())
    int64 = mesh_backend.int64
    cache, store = resume(
        lat,
        k,
        mesh.vertices_ij,
        mesh.vertices_source,
        mesh.origin_leaves,
        mesh.origin_level,
        mesh.origin_cls,
    )
    band = mesh.critical_band
    keys6 = _six_keys(lat, cache.ij[mesh.origin_leaves[band.leaves]])
    # Band samples are numbered in ascending key order, so these keys ascend.
    sample_keys = mesh_backend.fill_at_indices(
        mesh_backend.zeros((band.det.shape[0],), dtype=int64),
        band.samples.reshape(-1),
        keys6.reshape(-1),
    )
    mid = mesh_backend.flatnonzero(cache_lookup(cache, sample_keys) < 0)
    seed = (
        mesh.origin_status,
        mesh.origin_deviation,
        mesh.origin_sigma_min,
        sample_keys[mid],
        band.source[mid],
        mesh.vertices_jacobian,
        lattice_key(lat, band.measured_ij + (int(k) << lat.level)),
        band.measured_det,
    )
    return cache, store, seed


def extend_lens_mesh(mesh, raytrace, fov, *, batch_size=None):
    """
    Grow a lens mesh to a larger fov about the same center, reusing its refinement.

    Whole level-0 cells are added on every side: the fewest that reach
    ``fov``, to within ``1e-9`` of a cell. The ring is refined as a fresh
    build would refine it, and the balance settles the seam, so the result
    is the mesh :func:`build_lens_mesh` gives on the grown lattice, when the
    lens returns the same value at a point whatever batch it arrives in and
    the grown fov, center and cell size are dyadic. It equals that mesh bit
    for bit in every field but two, which agree to rounding:
    ``vertices_jacobian``, a float sum taken in another order, and the
    fitted entries of the band's ``det``, computed from it. Band membership
    and the set of measured samples are decided by comparisons on those
    values, so they can differ only where ``det A`` at a sample lies within
    rounding of zero, or a leaf's spread within rounding of
    :data:`~.band.RESOLVED_SPREAD`.

    No old vertex or band sample is raytraced again. Of an old leaf the
    extension brings into the band, the midpoints the old band did not hold
    are raytraced again. Only the samples the old mesh did not measure are
    measured.

    Parameters
    ----------
    mesh: LensMesh
    raytrace:
        The lens map ``mesh`` was built from.
    fov: float
        Requested side length.

        *Unit: arcsec*
    batch_size: Optional[int]
        As for :func:`build_lens_mesh`.

    Returns
    -------
    LensMesh
        ``mesh`` itself when ``fov`` needs no new cell.
    """
    device = backend.device(mesh.vertices_lens)
    on_mesh = to_mesh(mesh)
    out = _extend_lens_mesh(
        on_mesh, raytrace, to_mesh(fov), device=device, batch_size=batch_size
    )
    return mesh if out is on_mesh else to_user(out, device)


def _extend_lens_mesh(mesh, raytrace, fov, *, device, batch_size):
    """:func:`extend_lens_mesh` on ``mesh_backend`` arrays; ``device`` is the lens's."""
    h0 = mesh.lattice.h0
    k = max(0, math.ceil((float(fov) - mesh.lattice.fov) / (2.0 * h0) - 1e-9))
    if k == 0:
        return mesh
    lat = extend_lattice(to_device(mesh.lattice, build_device()), k)
    init_res, max_level = lat.init_res, lat.level - 1
    check_lattice_keys(
        init_res,
        max_level,
        "Extend by less, or rebuild with a larger min_img_sep or a lower max_depth.",
    )
    _warn_depth_limited(h0, mesh.min_img_sep, max_level)
    sample = make_sampler(raytrace, device, batch_size)
    cache, store, seed = _seed(mesh, lat, k)
    ij, cls = ring_triangles(init_res, k, lat.level, ROOT_CLASS)
    cache, store, rows = add_roots(cache, store, lat, ij, cls, sample, batch_size)
    return _freeze(
        lat,
        cache,
        store,
        rows,
        sample,
        mesh.min_img_sep,
        mesh.holes,
        batch_size,
        seed,
    )


def _hole_rings(fov, init_res, x0, y0, centers, min_img_sep):
    """
    Level-0 cells per side that put every center's hole strictly inside the fov.

    The holes are :func:`merge_centers`'s, at the halved ``min_img_sep`` the
    build gives it, so a merged hole reaches as far as its own radius. The
    smallest ``k`` with ``fov / 2 + k * h0`` beyond every hole's reach --
    its center's larger offset from ``(x0, y0)``, plus its radius.

    Returns
    -------
    k: int
        Zero when every hole already lies strictly inside.
    outside: int
        How many holes did not.
    """
    hole_centers, radius = merge_centers(centers, min_img_sep)
    if hole_centers.shape[0] == 0:
        return 0, 0
    center = mesh_backend.as_array([x0, y0], dtype=mesh_backend.float64)
    reach = mesh_backend.max(mesh_backend.abs(hole_centers - center), dim=1) + radius
    outside = int(mesh_backend.to_numpy(mesh_backend.sum(reach >= fov / 2)))
    if outside == 0:
        return 0, 0
    excess = float(mesh_backend.to_numpy(mesh_backend.max(reach))) - fov / 2
    return math.floor(excess / (fov / init_res)) + 1, outside


def _fov_boundary(mesh):
    """
    The vertices on ``mesh``'s fov boundary, counter-clockwise from its lower-left corner.

    Exact: they are the vertices at the extreme lattice coordinates, ordered
    by their integer distance along the perimeter, each corner on the side it
    starts. The mesh is conforming, so each is joined to the next by a leaf
    edge, and the last to the first.

    Returns
    -------
    ArrayLike
        ``(N,)`` int64 vertex indices.
    """
    ij = mesh.vertices_ij - mesh_backend.min(mesh.vertices_ij, dim=0)
    i, j = ij[:, 0], ij[:, 1]
    w, h = mesh_backend.max(i), mesh_backend.max(j)
    where = mesh_backend.where
    t = where(
        (j == 0) & (i < w),
        i,
        where(
            (i == w) & (j < h),
            w + j,
            where(
                (j == h) & (i > 0),
                2 * w + h - i,
                where((i == 0) & (j > 0), 2 * (w + h) - j, -1),
            ),
        ),
    )
    on = mesh_backend.flatnonzero(t >= 0)
    return on[mesh_backend.argsort(t[on])]


def _fov_bounds(mesh):
    """The lower and upper corners of ``mesh``'s fov, ``(2,)`` each: its vertices' extreme coordinates, lattice-exact."""
    return mesh_backend.min(mesh.vertices_lens, dim=0), mesh_backend.max(
        mesh.vertices_lens, dim=0
    )


def inside_fov(mesh, theta):
    """
    True where a lens-plane point lies in ``mesh``'s fov, its boundary included.

    Parameters
    ----------
    mesh: LensMesh
    theta: ArrayLike
        ``(K, 2)`` lens-plane points.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(K,)`` bool.
    """
    lo, hi = _fov_bounds(mesh)
    return mesh_backend.all((theta >= lo) & (theta <= hi), dim=1)


def inside_fov_image(mesh, beta):
    """
    True where a source-plane point lies inside the image of ``mesh``'s fov boundary.

    The image is the closed polyline through the images of
    :func:`_fov_boundary`'s vertices, as in :func:`_pseudo_caustics_crossed`,
    and a point is inside where it winds about the point
    (:func:`~.geometry.winding_number` nonzero), whichever its orientation.
    A point outside can have images outside the fov, where the mesh cannot
    find them. Where the lens is non-finite on the fov boundary, the
    boundary's image has gaps, so sources near them may count as outside,
    and growing the fov does not help. At most :data:`CROSSING_PAIRS`
    point-segment pairs are held at once.

    Parameters
    ----------
    mesh: LensMesh
    beta: ArrayLike
        ``(B, 2)`` source-plane points.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(B,)`` bool.
    """
    polygon = mesh.vertices_source[_fov_boundary(mesh)]
    step = max(1, CROSSING_PAIRS // max(1, polygon.shape[0]))
    inside = [
        mesh_backend.zeros(
            (0,), dtype=mesh_backend.bool, device=mesh_backend.device(beta)
        )
    ]
    for lo in range(0, beta.shape[0], step):
        inside.append(winding_number(polygon, beta[lo : lo + step]) != 0)
    return mesh_backend.concatenate(inside, dim=0)


def _pseudo_caustics_crossed(mesh):
    """
    How many pseudo-caustics the image of ``mesh``'s fov boundary crosses.

    The boundary's image is the closed polyline through the images of
    :func:`_fov_boundary`'s vertices, the mesh's own, affine on each leaf. A
    pseudo-caustic is the hole curve of a hole whose ``pseudo_caustic`` is
    True, closed through its wrap-around chord: the true pseudo-caustic to
    within about the hole's radius. Boundary segments outside a hole curve's
    bounding box cannot cross it; the rest are tested against it
    (:func:`~.geometry.segments_cross`), at most :data:`CROSSING_PAIRS`
    pairs at a time, until one crosses. A segment with a NaN end crosses
    nothing: every comparison on it fails.
    """
    holes = mesh.holes
    pseudo = mesh_backend.to_numpy(mesh_backend.flatnonzero(holes.pseudo_caustic))
    if pseudo.shape[0] == 0:
        return 0
    b0 = mesh.vertices_source[_fov_boundary(mesh)]
    b1 = mesh_backend.roll(b0, -1, 0)
    lo_b, hi_b = mesh_backend.minimum(b0, b1), mesh_backend.maximum(b0, b1)
    offsets = mesh_backend.to_numpy(holes.offsets).tolist()
    crossed = 0
    for h in pseudo.tolist():
        c0 = holes.source[offsets[h] : offsets[h + 1]]
        c1 = mesh_backend.roll(c0, -1, 0)
        lo, hi = mesh_backend.min(c0, dim=0), mesh_backend.max(c0, dim=0)
        near = mesh_backend.flatnonzero(
            mesh_backend.all((lo_b <= hi) & (hi_b >= lo), dim=1)
        )
        if near.shape[0] == 0:
            continue
        n0 = mesh_backend.unsqueeze(b0[near], 0)
        n1 = mesh_backend.unsqueeze(b1[near], 0)
        step = max(1, CROSSING_PAIRS // near.shape[0])
        for s in range(0, c0.shape[0], step):
            hit = segments_cross(
                mesh_backend.unsqueeze(c0[s : s + step], 1),
                mesh_backend.unsqueeze(c1[s : s + step], 1),
                n0,
                n1,
            )
            if bool(mesh_backend.any(hit)):
                crossed += 1
                break
    return crossed


def _curves_cut_by_fov(mesh, curves):
    """
    How many open curves have an end on ``mesh``'s fov boundary.

    Exact, with no tolerance: every vertex and band sample is placed by
    :func:`~.lattice.lattice_xy`, so the boundary is the vertices' own
    extreme coordinates, and a crossing on a boundary child edge, both of
    whose samples share that coordinate, reproduces it bit for bit
    (:func:`~.curves.edge_zeros`). An end at a band gap inside the fov does
    not count.
    """
    open_curves = mesh_backend.flatnonzero(~curves.closed)
    n_open = open_curves.shape[0]
    if n_open == 0:
        return 0
    ends = mesh_backend.concatenate(
        (curves.offsets[open_curves], curves.offsets[open_curves + 1] - 1)
    )
    xy = curves.lens[ends]
    lo, hi = _fov_bounds(mesh)
    on_boundary = mesh_backend.any((xy == lo) | (xy == hi), dim=1)
    cut = on_boundary[:n_open] | on_boundary[n_open:]
    return int(mesh_backend.to_numpy(mesh_backend.sum(cut)))


def build_closed_lens_mesh(
    raytrace,
    fov,
    init_res,
    min_img_sep,
    max_depth=25,
    *,
    growth=1.5,
    max_iters=10,
    x0=0.0,
    y0=0.0,
    centers=None,
    device=None,
    batch_size=None,
):
    """
    Build a lens mesh, then grow its fov until it cuts no critical curve and its boundary's image crosses no pseudo-caustic.

    When a hole around one of ``centers`` does not lie strictly inside
    ``fov``, the fov is first widened by the fewest whole level-0 cells
    that take every hole inside, with a warning. After the build, while an
    open curve ends on the fov boundary or the source-plane image of the
    boundary crosses a pseudo-caustic, and fewer than ``max_iters``
    extensions have run, :func:`extend_lens_mesh` grows the fov by
    ``growth``, rounded up to whole cells, and both are checked again.
    A curve ending inside the fov -- next to a non-finite leaf, at an
    unjoined hole, or where ``max_depth`` bound -- grows nothing.

    The pseudo-caustics are the hole curves of the holes whose
    ``pseudo_caustic`` is True, the curves ``curves`` flags in its own
    ``pseudo_caustic``, so only ``centers`` can give one. A source
    inside an isothermal center's pseudo-caustic has an image far out from
    it; while the boundary's image crosses the pseudo-caustic, some of those
    images lie outside the fov. A hole curve that is no pseudo-caustic, such
    as a point mass's loop, grows nothing.

    Parameters
    ----------
    raytrace, fov, init_res, min_img_sep, max_depth, x0, y0, centers, device, batch_size:
        As for :func:`build_lens_mesh`.
    growth: float
        Factor each extension asks to multiply the fov by.
    max_iters: int
        Most extensions to run.

    Returns
    -------
    mesh: LensMesh
    curves: CriticalCurvesAndCaustics
        The last mesh's critical curves, caustics and pseudo-caustics.
    """
    mesh, curves = _build_closed_lens_mesh(
        raytrace,
        *to_mesh((fov, init_res, min_img_sep, max_depth)),
        growth=to_mesh(growth),
        max_iters=max_iters,
        x0=to_mesh(x0),
        y0=to_mesh(y0),
        centers=to_mesh(centers),
        device=device,
        batch_size=batch_size,
    )
    return to_user(mesh, device), to_user(curves, device)


def _build_closed_lens_mesh(
    raytrace,
    fov,
    init_res,
    min_img_sep,
    max_depth,
    *,
    growth,
    max_iters,
    x0,
    y0,
    centers,
    device,
    batch_size,
):
    """:func:`build_closed_lens_mesh` on ``mesh_backend`` arrays, on the build device."""
    k, outside = _hole_rings(fov, init_res, x0, y0, centers, min_img_sep / 2)
    if k:
        widened = fov + 2 * k * (fov / init_res)
        warn(
            f"{outside} hole(s) around centers reach outside fov={fov:g}; building "
            f"at fov={widened:g}, init_res={init_res + 2 * k} so that every hole "
            "lies inside it.",
            stacklevel=3,
        )
        fov, init_res = widened, init_res + 2 * k
    mesh = _build_lens_mesh(
        raytrace,
        fov,
        init_res,
        min_img_sep,
        max_depth,
        x0=x0,
        y0=y0,
        centers=centers,
        device=device,
        batch_size=batch_size,
    )
    curves = _critical_curves_and_caustics(mesh)
    cut, crossed = _curves_cut_by_fov(mesh, curves), _pseudo_caustics_crossed(mesh)
    for _ in range(max_iters):
        if cut == 0 and crossed == 0:
            break
        mesh = _extend_lens_mesh(
            mesh,
            raytrace,
            growth * mesh.lattice.fov,
            device=device,
            batch_size=batch_size,
        )
        curves = _critical_curves_and_caustics(mesh)
        cut, crossed = _curves_cut_by_fov(mesh, curves), _pseudo_caustics_crossed(mesh)
    if cut or crossed:
        why = []
        if cut:
            why.append(f"the fov still cuts {cut} critical curve(s)")
        if crossed:
            why.append(
                f"the image of its boundary still crosses {crossed} pseudo-caustic(s)"
            )
        what = " and ".join(why)
        warn(
            f"{what[0].upper()}{what[1:]} after {max_iters} extension(s), at "
            f"fov={mesh.lattice.fov:g}; raise max_iters or growth to grow "
            "it further.",
            stacklevel=3,
        )
    return mesh, curves
