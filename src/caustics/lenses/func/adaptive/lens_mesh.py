"""
The lens mesh: an adaptive triangulation of the lens plane, carrying the lens map at its vertices.

:func:`build_lens_mesh` refines the lens plane until each leaf's affine
model of the lens map is accurate to ``min_img_sep`` and no critical curve
runs between its samples (:func:`~.criterion.lens_status`), down to a size
floor. :func:`extend_lens_mesh` grows a mesh to a larger fov, reusing every
lens evaluation, and :func:`build_closed_lens_mesh` grows it until the fov
cuts no critical curve and the image of its boundary crosses no
pseudo-caustic.
"""

import math
from typing import NamedTuple
from warnings import warn

from ....backend_obj import ArrayLike, backend
from .mesh_backend import mesh_backend, to_mesh, to_user
from .geometry import (
    ROOT_CLASS,
    build_device,
    is_member,
    segments_cross,
    sigma_min_2x2,
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
    LEAF_JACOBIAN_NONFINITE,
    LEAF_RAYTRACE_NONFINITE,
    lens_status,
)
from .band import CriticalBand, build_band, in_band
from .holes import CenterHoles, merge_centers, sample_holes
from .curves import _critical_curves_and_caustics

# Most segment pairs the closed build's crossing test holds at once.
CROSSING_PAIRS = 1 << 18


class LensMesh(NamedTuple):
    """
    An adaptive triangulation of the lens plane, queryable from the source plane.

    Vertex ``v`` has a lens-plane position ``vertices_lens[v]`` and its image
    ``vertices_source[v]``, so the two embeddings share one topology. Every
    leaf whose origin has a finite raytrace and Jacobian -- neither
    ``LEAF_RAYTRACE_NONFINITE`` nor ``LEAF_JACOBIAN_NONFINITE`` -- is in
    ``index``, converged or not. A leaf failing the criterion at the finest
    level straddles a fold, a critical curve or a lens center the mesh
    cannot resolve; its source triangle is queried grown by its deviation
    (:func:`leaf_grow`), and ``forward_raytrace`` accepts its roots further
    from their seeds. The pre-closure leaves -- the origins -- are kept with
    their level, class and status, which is what an extension refines from.

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
    vertices_det: ArrayLike
        ``(V,)`` ``det A`` of the lens Jacobian at each vertex.
    vertices_sigma_min: ArrayLike
        ``(V,)`` smallest singular value of the lens Jacobian at each vertex.
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
        ``(N,)`` float64 smallest singular value of each origin's affine
        model: the smaller of its children's and of ``A`` at its six samples.
        With ``origin_deviation`` it gives :func:`~.criterion.affine_error`,
        how far a seed of the origin can lie from its image, below
        ``min_img_sep`` on every converged origin.
    index: MeshIndex
        Over the source-plane images of the leaves whose origin is finite,
        each grown by :func:`leaf_grow`.
    critical_band: CriticalBand
        The finest-level origins ``det A`` changes sign across.
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
    vertices_det: ArrayLike
    vertices_sigma_min: ArrayLike
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


def make_sampler(raytrace, jacobian, device, batch_size=None):
    """
    The lens map as ``(N, 2) -> (N, 4)`` float64 ``(bx, by, det A, sigma_min A)``, or ``(N, 2)`` images when ``jacobian`` is None.

    ``raytrace`` and ``jacobian`` receive the same float64 coordinates on
    ``device``, in the user's backend, padded to
    ``backend.padded_size(N, batch_size)`` rows by repeating the last. Float64
    matters: the criterion compares midpoint deviations of ``O(fov)``
    quantities, which cancel to exactly zero below about
    ``sqrt(8 * eps * fov)`` and would read as converged. ``det A`` is formed
    in the Jacobian's dtype, then cast; ``sigma_min A``
    (:func:`~.geometry.sigma_min_2x2`) is formed on the mesh backend from the
    Jacobian's entries cast to float64, which travel with the other columns.
    The values come back on the positions' device, one row per position.
    """
    f64 = backend.float64

    def sample(xy):
        n = xy.shape[0]
        size = backend.padded_size(n, batch_size)
        padded = mesh_backend.pad_rows(xy, size) if size > n else xy
        x = backend.as_array(to_user(padded[:, 0]), dtype=f64, device=device)
        y = backend.as_array(to_user(padded[:, 1]), dtype=f64, device=device)
        columns = list(raytrace(x, y))
        if jacobian is not None:
            J = jacobian(x, y)
            columns.append(J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])
            columns.extend((J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]))
        out = backend.stack([backend.to(c, dtype=f64) for c in columns], dim=-1)
        out = mesh_backend.to(to_mesh(out), device=mesh_backend.device(xy))
        if jacobian is not None:
            sigma = sigma_min_2x2(out[:, 3:].reshape(-1, 2, 2))
            out = mesh_backend.concatenate(
                (out[:, :3], mesh_backend.unsqueeze(sigma, -1)), dim=1
            )
        return out[:n] if size > n else out

    return sample


def leaf_grow(origin_status, origin_deviation, leaf_origin):
    """
    How far each leaf's source-plane triangle reaches beyond itself when queried, ``(L,)``.

    A leaf whose origin did not converge reaches its origin's largest
    midpoint deviation: about how far the true image of one of its edges
    bulges from the straight one. A converged leaf reaches no further than
    its triangle.

    *Unit: arcsec*
    """
    return mesh_backend.where(origin_status == LEAF_CONVERGED, 0.0, origin_deviation)[
        leaf_origin
    ]


def _lens_split(h0, min_img_sep):
    """The lens criterion as :func:`~.refine.refine`'s ``split``."""

    def split(ij, values6, cls, level):
        status, _, _ = lens_status(
            values6[..., :2],
            values6[..., 2],
            values6[..., 3],
            cls,
            level,
            h0,
            min_img_sep,
        )
        return status != LEAF_CONVERGED

    return split


def _warn_depth_limited(h0, min_img_sep, max_level):
    request = f"min_img_sep={2 * min_img_sep:g} arcsec (refined internally to {min_img_sep:g})"
    warn_depth_limited("Lens mesh", request, h0, min_img_sep, max_level)


def build_lens_mesh(
    raytrace,
    jacobian,
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

    Parameters
    ----------
    raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]
        ``raytrace(x, y) -> (bx, by)`` on 1-D arrays, e.g. ``lens.raytrace``.
    jacobian: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ``jacobian(x, y) -> (N, 2, 2)``, the Jacobian of ``raytrace``, e.g.
        ``lens.jacobian_lens_equation``. Under jax that one re-traces its
        autodiff at every call. A jitted Jacobian taking the lens's
        parameters as an input is much cheaper over many builds and follows
        the parameters as they change: after ``lens.to_dynamic()``,
        ``jac = jax.jit(lambda x, y, p: lens.jacobian_lens_equation(x, y, params=p))``
        passed as ``lambda x, y: jac(x, y, lens.get_values())``. It must not
        close over parameter values.
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
        it.

        *Unit: arcsec*
    device: Optional
        Device the lens lives on: ``raytrace`` and ``jacobian`` receive
        coordinates there, and the mesh is stored there.
    batch_size: Optional[int]
        Most points per ``raytrace`` and ``jacobian`` call.

    Returns
    -------
    LensMesh
    """
    mesh = _build_lens_mesh(
        raytrace,
        jacobian,
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
    jacobian,
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
    holes = sample_holes(
        make_sampler(raytrace, None, device, batch_size),
        hole_centers,
        hole_radius,
        min_img_sep,
        batch_size,
    )
    sample = make_sampler(raytrace, jacobian, device, batch_size)
    ij, cls = initial_triangles(init_res, lat.level, ROOT_CLASS)
    cache, store, rows = add_roots(
        empty_cache(4), empty_store(), lat, ij, cls, sample, batch_size
    )
    cache, store = refine(
        cache,
        store,
        rows,
        lat,
        sample,
        _lens_split(h0, min_img_sep),
        max_level,
        batch_size,
    )
    return _freeze(lat, cache, store, sample, min_img_sep, holes, batch_size)


def _freeze(lat, cache, store, sample, min_img_sep, holes, batch_size, seed=None):
    """
    Close and index a refinement, finishing its finest level.

    Each origin not carried over in ``seed`` gets its status, deviation and
    smallest singular value from :func:`~.criterion.lens_status` on its six
    samples: below the finest level its midpoints are in the cache,
    ``refine`` having tested and passed it; at the finest level they are
    sampled here and never cached, and they also give the band. ``seed`` is
    ``(status, deviation, sigma_min, band_rows, mid_keys, mid_values)`` from
    :func:`_seed`: the statuses, deviations and smallest singular values of
    the store's first rows, the store rows of the old band, and the old
    band's midpoint samples, read here rather than sampled again.
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
            mesh_backend.zeros((0,), dtype=int64),
            mesh_backend.zeros((0, 4), dtype=f64),
        )
    seed_status, seed_deviation, seed_sigma_min, band_rows, mid_keys, mid_values = seed
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
    row = mesh_backend.arange(n_rows, dtype=int64)
    young = mesh_backend.flatnonzero(store.valid & (row >= n_seed))
    finest = mesh_backend.flatnonzero(store.level[young] == lat.level - 1)
    fresh = young[finest]

    ij = cache.ij[store.v[young]]
    mid = lattice_key(lat, midpoint_ij(ij))
    todo = mesh_backend.unique(mid[finest].reshape(-1))
    todo = todo[~is_member(mid_keys, todo)]
    if todo.shape[0]:
        xy = lattice_xy(lat, lattice_ij_from_key(lat, todo))
        new_values = sample_points(xy, sample, batch_size)
    else:
        new_values = mesh_backend.zeros((0, 4), dtype=f64)
    table_keys = mesh_backend.concatenate((cache.keys, mid_keys, todo), dim=0)
    table_values = mesh_backend.concatenate(
        (cache.values[cache.slots], mid_values, new_values), dim=0
    )
    order = mesh_backend.argsort(table_keys)
    table_keys, table_values = table_keys[order], table_values[order]

    keys6 = mesh_backend.concatenate((lattice_key(lat, ij), mid), dim=1)
    values6 = table_values[mesh_backend.searchsorted(table_keys, keys6)]
    young_status, young_deviation, young_sigma_min = lens_status(
        values6[..., :2],
        values6[..., 2],
        values6[..., 3],
        store.cls[young],
        store.level[young],
        lat.h0,
        min_img_sep,
    )
    status = mesh_backend.fill_at_indices(status, young, young_status)
    deviation = mesh_backend.fill_at_indices(deviation, young, young_deviation)
    sigma_min = mesh_backend.fill_at_indices(sigma_min, young, young_sigma_min)
    band_rows = mesh_backend.concatenate(
        (band_rows, fresh[mesh_backend.flatnonzero(in_band(values6[finest][..., 2]))]),
        dim=0,
    )

    origin_of_row = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n_rows,), dtype=int64) - 1,
        origin,
        mesh_backend.arange(origin.shape[0], dtype=int64),
    )
    band_leaves = origin_of_row[band_rows]
    order = mesh_backend.argsort(band_leaves)
    band_leaves, band_rows = band_leaves[order], band_rows[order]
    bij = cache.ij[store.v[band_rows]]
    band_keys6 = mesh_backend.concatenate(
        (lattice_key(lat, bij), lattice_key(lat, midpoint_ij(bij))), dim=1
    )
    band = build_band(lat, band_leaves, band_keys6, table_keys, table_values)

    origin_status = status[origin]
    vertices_source = cache.values[used][:, :2]
    origin_deviation = deviation[origin]
    finite = (
        origin_status[leaf_origin] & (LEAF_RAYTRACE_NONFINITE | LEAF_JACOBIAN_NONFINITE)
    ) == 0
    indexed = mesh_backend.flatnonzero(finite)
    grow = leaf_grow(origin_status, origin_deviation, leaf_origin)
    mesh = LensMesh(
        lattice=lat,
        vertices_ij=cache.ij[used],
        vertices_lens=lattice_xy(lat, cache.ij[used]),
        vertices_source=vertices_source,
        vertices_det=cache.values[used][:, 2],
        vertices_sigma_min=cache.values[used][:, 3],
        leaves=leaves,
        leaf_origin=leaf_origin,
        origin_leaves=origin_leaves,
        origin_level=store.level[origin],
        origin_cls=store.cls[origin],
        origin_status=origin_status,
        origin_deviation=origin_deviation,
        origin_sigma_min=sigma_min[origin],
        index=build_index(vertices_source, leaves, indexed, grow[indexed]),
        critical_band=band,
        holes=holes,
        min_img_sep=float(min_img_sep),
    )
    return mesh


def _seed(mesh, lat, k):
    """
    :func:`~.refine.refine`'s state, and :func:`_freeze`'s ``seed``, from a mesh on its lattice grown by ``k``.

    The cache and store are :func:`~.refine.resume`'s.
    """
    mesh = to_device(mesh, build_device())
    int64 = mesh_backend.int64
    cache, store = resume(
        lat,
        k,
        mesh.vertices_ij,
        mesh_backend.concatenate(
            (
                mesh.vertices_source,
                mesh_backend.unsqueeze(mesh.vertices_det, -1),
                mesh_backend.unsqueeze(mesh.vertices_sigma_min, -1),
            ),
            dim=1,
        ),
        mesh.origin_leaves,
        mesh.origin_level,
        mesh.origin_cls,
    )
    band = mesh.critical_band
    bij = cache.ij[mesh.origin_leaves[band.leaves]]
    keys6 = mesh_backend.concatenate(
        (lattice_key(lat, bij), lattice_key(lat, midpoint_ij(bij))), dim=1
    )
    # Band samples are numbered in ascending key order, so these keys ascend.
    sample_keys = mesh_backend.fill_at_indices(
        mesh_backend.zeros((band.det.shape[0],), dtype=int64),
        band.samples.reshape(-1),
        keys6.reshape(-1),
    )
    mid = mesh_backend.flatnonzero(cache_lookup(cache, sample_keys) < 0)
    values = mesh_backend.concatenate(
        (
            band.source,
            mesh_backend.unsqueeze(band.det, -1),
            mesh_backend.unsqueeze(band.sigma_min, -1),
        ),
        dim=1,
    )
    seed = (
        mesh.origin_status,
        mesh.origin_deviation,
        mesh.origin_sigma_min,
        band.leaves,
        sample_keys[mid],
        values[mid],
    )
    return cache, store, seed


def extend_lens_mesh(mesh, raytrace, jacobian, fov, *, batch_size=None):
    """
    Grow a lens mesh to a larger fov about the same center, reusing its refinement.

    Whole level-0 cells are added on every side: the fewest that reach
    ``fov``, to within ``1e-9`` of a cell. The ring is refined as a fresh
    build would refine it, and the balance settles the seam, so the result is
    the mesh :func:`build_lens_mesh` gives on the grown lattice -- bit for bit
    when the lens returns the same value at a point whatever batch it arrives
    in, and the grown fov, center and cell size are dyadic. No old vertex or
    band sample is sampled again.

    Parameters
    ----------
    mesh: LensMesh
    raytrace, jacobian:
        The pair ``mesh`` was built from.
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
        on_mesh, raytrace, jacobian, to_mesh(fov), device=device, batch_size=batch_size
    )
    return mesh if out is on_mesh else to_user(out, device)


def _extend_lens_mesh(mesh, raytrace, jacobian, fov, *, device, batch_size):
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
    sample = make_sampler(raytrace, jacobian, device, batch_size)
    cache, store, seed = _seed(mesh, lat, k)
    ij, cls = ring_triangles(init_res, k, lat.level, ROOT_CLASS)
    cache, store, rows = add_roots(cache, store, lat, ij, cls, sample, batch_size)
    split = _lens_split(h0, mesh.min_img_sep)
    cache, store = refine(cache, store, rows, lat, sample, split, max_level, batch_size)
    return _freeze(
        lat,
        cache,
        store,
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
    jacobian,
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
    raytrace, jacobian, fov, init_res, min_img_sep, max_depth, x0, y0, centers, device, batch_size:
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
        jacobian,
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
    jacobian,
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
        jacobian,
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
            jacobian,
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
