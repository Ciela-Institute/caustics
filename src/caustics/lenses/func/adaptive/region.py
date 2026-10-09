"""
Regions of the source plane between two total magnifications.

:func:`magnification_region` refines a throwaway triangulation of the source
plane, the source mesh, wherever the boundary of ``{mu_min <= mu_tot <=
mu_max}`` can run, samples ``mu_tot`` at its vertices from the lens mesh
alone (:func:`~.magnification._total_magnification`), and traces that
boundary as the zero sets of ``mu_tot - mu_min`` and ``mu_max - mu_tot``,
with the tracer of the critical curves. Every size it uses comes from the
lens mesh: its level-0 cells are the lens mesh's ``init_res`` over the image
of the fov boundary, and a triangle stops splitting once it is no longer
than the source-plane images of the leaves at its samples.
:func:`build_closed_magnification_region` grows the lens mesh, and extends
the source mesh, until the region is bounded.
"""

import math
from typing import NamedTuple
from warnings import warn

from ....backend_obj import ArrayLike, backend
from .mesh_backend import mesh_backend, to_mesh, to_user
from .geometry import (
    ROOT_CLASS,
    build_device,
    csr_offsets,
    is_member,
    ring_next,
    to_device,
)
from .lattice import (
    MAX_KEY,
    Lattice,
    check_lattice_keys,
    depth_floor,
    extend_lattice,
    initial_triangles,
    lattice_fov,
    lattice_h0,
    lattice_init_res,
    lattice_key,
    lattice_xy,
    make_lattice,
    ring_triangles,
)
from .refine import (
    LeafStore,
    VertexCache,
    add_roots,
    close,
    empty_cache,
    empty_store,
    refine,
)
from .lens_mesh import (
    _build_closed_lens_mesh,
    _extend_lens_mesh,
    _fov_boundary,
    inside_fov_image,
)
from .curves import (
    _critical_curves_and_caustics,
    chain_segments,
    edge_zeros,
    triangle_segments,
)
from .magnification import _total_magnification, leaf_tables


class SourceMesh(NamedTuple):
    """
    A refinement of the source plane for one ``(mu_min, mu_max)``, and what an extension resumes from.

    Parameters
    ----------
    lattice: Lattice
        Over a square around the image of the fov boundary.
    max_level: int
        The finest level.
    vertices_ij: ArrayLike
        ``(V, 2)`` int64 lattice coordinates, in ascending lattice-key order.
    vertices: ArrayLike
        ``(V, 2)`` float64 positions.

        *Unit: arcsec*
    values: ArrayLike
        ``(V, 2)`` float64 ``mu_tot`` and the lens mesh's local resolution
        at each vertex (:func:`~.magnification._total_magnification`).
    leaves: ArrayLike
        ``(L, 3)`` int64 closed, conforming, positively oriented leaves.
    leaf_origin: ArrayLike
        ``(L,)`` int64 origin of each leaf.
    origin_leaves: ArrayLike
        ``(N, 3)`` int64 pre-closure leaves.
    origin_level: ArrayLike
        ``(N,)`` int64.
    origin_cls: ArrayLike
        ``(N,)`` int64.
    cell_valid: ArrayLike
        ``(R, R)`` bool per level-0 cell (:func:`_cell_validity`).
    """

    lattice: Lattice
    max_level: int
    vertices_ij: ArrayLike
    vertices: ArrayLike
    values: ArrayLike
    leaves: ArrayLike
    leaf_origin: ArrayLike
    origin_leaves: ArrayLike
    origin_level: ArrayLike
    origin_cls: ArrayLike
    cell_valid: ArrayLike


def _corner(lat):
    """The position of lattice point ``(0, 0)``, ``(2,)``."""
    return lattice_xy(lat, mesh_backend.zeros((2,), dtype=mesh_backend.int64))


def _boundary_box(mesh):
    """The lower and upper corners of the image of ``mesh``'s fov boundary, its finite points only."""
    poly = mesh.vertices_source[_fov_boundary(mesh)]
    poly = poly[
        mesh_backend.flatnonzero(mesh_backend.all(mesh_backend.isfinite(poly), dim=1))
    ]
    return mesh_backend.min(poly, dim=0), mesh_backend.max(poly, dim=0)


def _key_level_cap(init_res):
    """The deepest ``max_level`` whose lattice, one level finer, int64 keys hold (:func:`~.lattice.check_lattice_keys`)."""
    level = 0
    while ((int(init_res) << (level + 2)) + 1) ** 2 < MAX_KEY:
        level += 1
    return level


def _source_lattice(mesh, tables):
    """
    The source lattice, and its finest level.

    The lattice covers the square around the image of ``mesh``'s fov
    boundary, centered on it, in the lens mesh's ``init_res`` level-0 cells
    per axis. The finest level is the first whose triangles are no longer
    than the smallest positive leaf image, ``tables.size``, capped where
    int64 keys run out.
    """
    lo, hi = _boundary_box(mesh)
    side = float(mesh_backend.to_numpy(mesh_backend.max(hi - lo)))
    x0, y0 = mesh_backend.to_numpy(0.5 * (lo + hi)).tolist()
    init_res = lattice_init_res(mesh.lattice)
    usable = mesh_backend.isfinite(tables.size) & (tables.size > 0)
    smallest = mesh_backend.min(mesh_backend.where(usable, tables.size, math.inf))
    max_level = min(
        depth_floor(side / init_res, float(mesh_backend.to_numpy(smallest))),
        _key_level_cap(init_res),
    )
    return make_lattice(side, x0, y0, init_res, max_level + 1), max_level


def _sampler(mesh, tables):
    """``(N, 2)`` positions to ``(N, 2)`` float64 ``(mu_tot, size)``, as :func:`~.refine.refine`'s ``sample``."""

    def sample(xy):
        mu, size = _total_magnification(mesh, tables, xy)
        return mesh_backend.stack((mu, size), dim=-1)

    return sample


def _cells_of_boxes(lo, hi, corner, side, n_cells):
    """
    Cells of a grid each box overlaps, an edge it touches included, as flat indices ``i * n_cells + j``.

    The grid has ``n_cells`` square cells of side ``side`` per axis from
    ``corner``. Boxes are ``(K, 2)`` lower and upper corners; parts outside
    the grid are dropped.
    """
    int64 = mesh_backend.int64
    device = mesh_backend.device(lo)
    u = (lo - corner) / side
    a = mesh_backend.long(mesh_backend.floor(u))
    # A box whose lower edge lies on a cell line touches the cell below it too.
    a = a - mesh_backend.long(u == mesh_backend.to(a, dtype=u.dtype))
    b = mesh_backend.long(mesh_backend.floor((hi - corner) / side))
    a = mesh_backend.clamp(a, 0, None)
    b = mesh_backend.clamp(b, None, n_cells - 1)
    nx = mesh_backend.clamp(b[:, 0] - a[:, 0] + 1, 0, None)
    ny = mesh_backend.clamp(b[:, 1] - a[:, 1] + 1, 0, None)
    count = nx * ny
    box = mesh_backend.repeat(
        mesh_backend.arange(lo.shape[0], dtype=int64, device=device), count, axis=0
    )
    k = mesh_backend.arange(
        box.shape[0], dtype=int64, device=device
    ) - mesh_backend.repeat(mesh_backend.cumsum(count, dim=0) - count, count, axis=0)
    return (a[box, 0] + k // ny[box]) * n_cells + a[box, 1] + k % ny[box]


def _cell_validity(mesh, lat):
    """
    Which level-0 cells of ``lat`` lie wholly inside the image of ``mesh``'s fov boundary, ``(R, R)`` bool.

    A cell is valid when its four corners pass
    :func:`~.lens_mesh.inside_fov_image` and no segment of the boundary's
    image has a bounding box touching it. Corners inside are not enough:
    the boundary's image need not be convex, and can dent into a cell
    between them. Every descendant of a cell inherits its validity, so no
    sample is tested again. A segment with a non-finite end is skipped, as
    :func:`~.lens_mesh.inside_fov_image` skips it.
    """
    int64 = mesh_backend.int64
    r = lattice_init_res(lat)
    g = mesh_backend.arange(r + 1, dtype=int64) * (1 << lat.level)
    gi, gj = mesh_backend.meshgrid(g, g, indexing="ij")
    corners = lattice_xy(
        lat, mesh_backend.stack((gi.reshape(-1), gj.reshape(-1)), dim=-1)
    )
    inside = inside_fov_image(mesh, corners).reshape(r + 1, r + 1)
    valid = inside[:-1, :-1] & inside[1:, :-1] & inside[:-1, 1:] & inside[1:, 1:]
    poly = mesh.vertices_source[_fov_boundary(mesh)]
    nxt = mesh_backend.roll(poly, -1, 0)
    lo, hi = mesh_backend.minimum(poly, nxt), mesh_backend.maximum(poly, nxt)
    finite = mesh_backend.flatnonzero(
        mesh_backend.all(mesh_backend.isfinite(lo) & mesh_backend.isfinite(hi), dim=1)
    )
    cut = _cells_of_boxes(lo[finite], hi[finite], _corner(lat), lattice_h0(lat), r)
    return mesh_backend.fill_at_indices(valid.reshape(-1), cut, False).reshape(r, r)


def _caustic_marks(curves, lat, max_level):
    """
    The cells each true caustic marks, as sorted flat cell indices, one array per level ``0 .. max_level - 1``.

    Only the segments between two consecutive crossings of ``det A = 0``
    (both ``hole == -1``) are caustic: ``mu_tot`` jumps across them.
    Pseudo-caustics and hole arcs are skipped, as ``mu_tot`` is continuous
    across them: the image appearing there has ``mu -> 0``. A segment of
    length ``l`` marks, at every level whose triangles are longer than
    ``l``, the cells its bounding box touches, so along a caustic a triangle
    splits until it is no longer than the caustic's own segments.
    """
    int64, f64 = mesh_backend.int64, mesh_backend.float64
    n_points = curves.source.shape[0]
    if n_points == 0:
        return [mesh_backend.zeros((0,), dtype=int64) for _ in range(max_level)]
    curve, nxt = ring_next(curves.offsets, n_points)
    point = mesh_backend.arange(n_points, dtype=int64)
    has_next = (point != curves.offsets[1:][curve] - 1) | curves.closed[curve]
    keep = mesh_backend.flatnonzero(
        has_next & (curves.hole == -1) & (curves.hole[nxt] == -1)
    )
    a = mesh_backend.to(curves.source[keep], dtype=f64)
    b = mesh_backend.to(curves.source[nxt[keep]], dtype=f64)
    length = mesh_backend.norm(b - a, dim=-1)
    lo, hi = mesh_backend.minimum(a, b), mesh_backend.maximum(a, b)
    h0, r, corner = lattice_h0(lat), lattice_init_res(lat), _corner(lat)
    marks = []
    for level in range(max_level):
        sel = mesh_backend.flatnonzero(length < math.sqrt(2.0) * h0 / 2**level)
        cells = _cells_of_boxes(lo[sel], hi[sel], corner, h0 / 2**level, r << level)
        marks.append(mesh_backend.unique(cells))
    return marks


def _straddles(mu6, c):
    """True where the six samples of a row lie on both sides of ``c``, ``>= c`` counting as above."""
    above = mu6 >= c
    return mesh_backend.any(above, dim=1) & mesh_backend.any(~above, dim=1)


def _source_split(lat, valid, marks, mu_min, mu_max):
    """
    The source mesh's criterion as :func:`~.refine.refine`'s ``split``.

    A triangle splits when its level-0 cell is valid and either the
    threshold can cross it -- ``mu_tot - mu_min``, or ``mu_tot - mu_max``,
    changes sign among its six samples -- while it is longer than the
    smallest leaf image at them; or a caustic marks its cell
    (:func:`_caustic_marks`) and not every sample exceeds the upper
    threshold, ``mu_max`` or, without it, ``mu_min``. The second rule
    catches the strip by a fold where ``mu_tot``, diverging like ``1 /
    sqrt(d)``, exceeds the threshold although every sample near it is below.
    """
    r, h0 = lattice_init_res(lat), lattice_h0(lat)
    upper = mu_min if mu_max is None else mu_max

    def split(ij, values6, cls, level):
        mu, size = values6[..., 0], values6[..., 1]
        corner = mesh_backend.min(ij, dim=1)
        cell = corner // (1 << lat.level)
        ok = valid[cell[:, 0], cell[:, 1]]
        edge = (
            math.sqrt(2.0) * h0 / mesh_backend.to(2**level, dtype=mesh_backend.float64)
        )
        crossed = _straddles(mu, mu_min)
        if mu_max is not None:
            crossed = crossed | _straddles(mu, mu_max)
        coarse = edge > mesh_backend.min(size, dim=1)
        marked = mesh_backend.zeros_like(ok)
        for l in mesh_backend.to_numpy(mesh_backend.unique(level)).tolist():
            rows = mesh_backend.flatnonzero(level == l)
            at = corner[rows] // (1 << (lat.level - l))
            marked = mesh_backend.fill_at_indices(
                marked, rows, is_member(marks[l], at[:, 0] * (r << l) + at[:, 1])
            )
        caustic = marked & ~mesh_backend.all(mu > upper, dim=1)
        return ok & ((crossed & coarse) | caustic)

    return split


def _freeze_source(lat, max_level, cache, store, valid):
    """Close a source refinement into a :class:`SourceMesh`."""
    used, leaves, leaf_origin, origin, origin_leaves = close(lat, cache, store)
    ij = cache.ij[used]
    return SourceMesh(
        lattice=lat,
        max_level=max_level,
        vertices_ij=ij,
        vertices=lattice_xy(lat, ij),
        values=cache.values[used],
        leaves=leaves,
        leaf_origin=leaf_origin,
        origin_leaves=origin_leaves,
        origin_level=store.level[origin],
        origin_cls=store.cls[origin],
        cell_valid=valid,
    )


def _source_mesh(mesh, curves, mu_min, mu_max):
    """The :class:`SourceMesh` of one ``(mu_min, mu_max)``; ``mesh`` and ``curves`` on the build device."""
    tables = leaf_tables(mesh)
    lat, max_level = _source_lattice(mesh, tables)
    sample = _sampler(mesh, tables)
    valid = _cell_validity(mesh, lat)
    split = _source_split(
        lat, valid, _caustic_marks(curves, lat, max_level), mu_min, mu_max
    )
    ij, cls = initial_triangles(lattice_init_res(lat), lat.level, ROOT_CLASS)
    cache, store, rows = add_roots(
        empty_cache(2), empty_store(), lat, ij, cls, sample, None
    )
    cache, store = refine(cache, store, rows, lat, sample, split, max_level, None)
    return _freeze_source(lat, max_level, cache, store, valid)


class MagnificationRegion(NamedTuple):
    """
    The region ``{mu_min <= mu_tot <= mu_max}`` of the source plane: its contours and area.

    Contour ``c`` is rows ``offsets[c]:offsets[c + 1]`` of ``source``. Every
    contour keeps the region on its left; the ``mu_min`` contours come
    first, then the ``mu_max`` ones. A contour is the zero set of ``mu_tot -
    mu_min``, or of ``mu_max - mu_tot``, linear on each source-mesh leaf;
    where a caustic's jump in ``mu_tot`` straddles a threshold, it runs
    along the caustic, within one source-mesh leaf of it.

    Parameters
    ----------
    source: ArrayLike
        ``(P, 2)`` float64 contour points.

        *Unit: arcsec*
    offsets: ArrayLike
        ``(C + 1,)`` int64 CSR offsets, ``offsets[0] == 0``.
    closed: ArrayLike
        ``(C,)`` bool, True where the contour is a loop.
    level: ArrayLike
        ``(C,)`` float64, the threshold each contour belongs to.
    area: float
        The region's area: the sum of its contours' signed shoelace areas.
        NaN when the region reaches outside the valid cells of the source
        mesh, where its extent is unknown.

        *Unit: arcsec^2*
    mu_min: float
    mu_max: float
        ``inf`` when not given.
    """

    source: ArrayLike
    offsets: ArrayLike
    closed: ArrayLike
    level: ArrayLike
    area: float
    mu_min: float
    mu_max: float


def _leaf_valid(src):
    """True where a source leaf's level-0 cell is valid, ``(L,)``."""
    cell = mesh_backend.min(src.vertices_ij[src.leaves], dim=1) // (
        1 << src.lattice.level
    )
    return src.cell_valid[cell[:, 0], cell[:, 1]]


def _contours(src, field, leaf_valid):
    """The zero set of ``field`` over the valid leaves, positive side on the left: points, offsets, closed."""
    tri = src.leaves[mesh_backend.flatnonzero(leaf_valid)]
    start, end = triangle_segments(tri, field >= 0)
    edges, offsets, closed = chain_segments(start, end, src.vertices.shape[0])
    (points,) = edge_zeros(edges, field, (src.vertices,))
    return points, offsets, closed


def _shoelace(points, offsets):
    """The sum of the signed areas of closed polylines, counter-clockwise positive."""
    if points.shape[0] == 0:
        return 0.0
    _, nxt = ring_next(offsets, points.shape[0])
    q = points[nxt]
    twice = mesh_backend.sum(points[:, 0] * q[:, 1] - q[:, 0] * points[:, 1])
    return 0.5 * float(mesh_backend.to_numpy(twice))


def _region(src, mu_min, mu_max):
    """
    The :class:`MagnificationRegion` a source mesh holds.

    Contours are traced over the leaves of valid cells only, so one reaching
    an invalid cell ends there, open. The region is bounded when no vertex
    of an invalid leaf lies in it, so that it lies wholly in the valid
    cells; this also covers a region with no contour at all, such as one
    whose ``mu_min`` is at or below the far-field ``mu_tot``. Its area is
    then the sum of its contours' signed shoelace areas (Green's theorem),
    exactly the area of the region of the source mesh's piecewise-linear
    field; otherwise NaN.
    """
    f64 = mesh_backend.float64
    mu = src.values[:, 0]
    upper = math.inf if mu_max is None else float(mu_max)
    leaf_valid = _leaf_valid(src)
    fields = [(mu - mu_min, mu_min)]
    if mu_max is not None:
        fields.append((mu_max - mu, upper))
    parts = [(_contours(src, field, leaf_valid), value) for field, value in fields]
    points = mesh_backend.concatenate([p for (p, _, _), _ in parts], dim=0)
    offsets = csr_offsets(
        mesh_backend.concatenate([o[1:] - o[:-1] for (_, o, _), _ in parts], dim=0)
    )
    closed = mesh_backend.concatenate([c for (_, _, c), _ in parts], dim=0)
    level = mesh_backend.concatenate(
        [
            mesh_backend.zeros((c.shape[0],), dtype=f64) + value
            for (_, _, c), value in parts
        ],
        dim=0,
    )
    invalid = src.leaves[mesh_backend.flatnonzero(~leaf_valid)].reshape(-1)
    in_invalid = mesh_backend.fill_at_indices(
        mesh_backend.zeros((mu.shape[0],), dtype=mesh_backend.bool), invalid, True
    )
    reaches = bool(mesh_backend.any(in_invalid & (mu >= mu_min) & (mu <= upper)))
    return MagnificationRegion(
        source=points,
        offsets=offsets,
        closed=closed,
        level=level,
        area=math.nan if reaches else _shoelace(points, offsets),
        mu_min=float(mu_min),
        mu_max=upper,
    )


def magnification_region(mesh, curves, mu_min, mu_max=None):
    """
    The region of the source plane where ``mu_min <= mu_tot <= mu_max``, its contours and its area.

    A source mesh is refined over the image of ``mesh``'s fov boundary
    wherever a threshold, or a caustic the region hugs, can pass, down to
    the source-plane size of the lens mesh's leaves there, with ``mu_tot``
    from the lens mesh alone (:func:`~.magnification.total_magnification`);
    the region's boundary is traced through it, and its area summed from the
    contours. No lens call is made, and nothing is kept for another
    ``(mu_min, mu_max)``.

    The region is bounded only when ``mu_min`` exceeds the far-field
    ``mu_tot``: ``1 / |(1 - kappa)**2 - gamma**2|`` of any external
    convergence and shear, 1 without them. When it reaches outside the
    valid cells, those wholly inside the image of the fov boundary, its
    area is NaN and the contours reaching there are open; grow the fov
    (:func:`build_closed_magnification_region`). A closed contour smaller
    than a level-0 cell with no caustic inside can be missed, as the lens
    mesh's ``init_res`` must resolve the lens's smallest structure.

    Parameters
    ----------
    mesh: LensMesh
    curves: CriticalCurvesAndCaustics
        ``mesh``'s, from :func:`~.lens_mesh.build_closed_lens_mesh` or
        :func:`~.curves.critical_curves_and_caustics`.
    mu_min: float
    mu_max: Optional[float]

    Returns
    -------
    MagnificationRegion
    """
    device = backend.device(mesh.vertices_lens)
    region = _magnification_region(
        to_mesh(mesh),
        to_mesh(curves),
        float(mu_min),
        None if mu_max is None else float(mu_max),
    )
    return to_user(region, device)


def _magnification_region(mesh, curves, mu_min, mu_max):
    """:func:`magnification_region` on ``mesh_backend`` arrays."""
    mesh = to_device(mesh, build_device())
    curves = to_device(curves, build_device())
    return _region(_source_mesh(mesh, curves, mu_min, mu_max), mu_min, mu_max)


def _seed_source(src, lat, k):
    """
    :func:`~.refine.refine`'s state from a source mesh, on its lattice grown by ``k``.

    The vertices are in lattice-key order, which a uniform shift keeps, so
    they are the cache as they stand, every one a leaf vertex; the origins
    are the store, row for row.
    """
    int64 = mesh_backend.int64
    ij = src.vertices_ij + (int(k) << lat.level)
    n_vertices, n_origins = ij.shape[0], src.origin_leaves.shape[0]
    cache = VertexCache(
        keys=lattice_key(lat, ij),
        slots=mesh_backend.arange(n_vertices, dtype=int64),
        ij=ij,
        values=src.values,
        active=mesh_backend.ones((n_vertices,), dtype=mesh_backend.bool),
    )
    store = LeafStore(
        v=src.origin_leaves,
        level=src.origin_level,
        cls=src.origin_cls,
        valid=mesh_backend.ones((n_origins,), dtype=mesh_backend.bool),
    )
    return cache, store


def _extend_source_mesh(src, mesh, curves, mu_min, mu_max):
    """
    Grow a source mesh over the image of an extended lens mesh's fov boundary.

    Whole level-0 cells are added about the same center, the fewest that
    cover the new image's bounding box; the cell size and finest level stay
    the first build's. Old vertices keep their ``(mu_tot, size)``; the new
    ring's roots, and the rows of every cell that was invalid and is valid
    now, are refined against the extended lens mesh. The lens extension's
    balance can split old lens leaves along its seam, and a few old vertices
    by the old boundary's image keep values read before that split, so the
    result matches a fresh build only to the mesh's accuracy.
    """
    int64 = mesh_backend.int64
    old = src.lattice
    lo, hi = _boundary_box(mesh)
    corner = _corner(old)
    over = mesh_backend.max(
        mesh_backend.concatenate((corner - lo, hi - corner - lattice_fov(old)), dim=0)
    )
    k = max(0, math.ceil(float(mesh_backend.to_numpy(over)) / lattice_h0(old) - 1e-9))
    lat = extend_lattice(old, k)
    check_lattice_keys(
        lattice_init_res(lat),
        src.max_level,
        "Build the lens mesh at a larger fov to begin with.",
    )
    tables = leaf_tables(mesh)
    sample = _sampler(mesh, tables)
    cache, store = _seed_source(src, lat, k)
    valid = _cell_validity(mesh, lat)
    r0 = lattice_init_res(old)
    was = mesh_backend.fill_at_indices(
        mesh_backend.zeros_like(valid),
        (slice(k, k + r0), slice(k, k + r0)),
        src.cell_valid,
    )
    rows = mesh_backend.arange(store.v.shape[0], dtype=int64)
    cell = mesh_backend.min(cache.ij[store.v], dim=1) // (1 << lat.level)
    newly = valid[cell[:, 0], cell[:, 1]] & ~was[cell[:, 0], cell[:, 1]]
    untested = rows[mesh_backend.flatnonzero(newly)]
    if k:
        ij, cls = ring_triangles(lattice_init_res(lat), k, lat.level, ROOT_CLASS)
        cache, store, ring = add_roots(cache, store, lat, ij, cls, sample, None)
        untested = mesh_backend.concatenate((ring, untested), dim=0)
    split = _source_split(
        lat, valid, _caustic_marks(curves, lat, src.max_level), mu_min, mu_max
    )
    cache, store = refine(
        cache, store, untested, lat, sample, split, src.max_level, None
    )
    return _freeze_source(lat, src.max_level, cache, store, valid)


def build_closed_magnification_region(
    raytrace,
    jacobian,
    fov,
    init_res,
    min_img_sep,
    mu_min,
    mu_max=None,
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
    Build a closed lens mesh, then grow its fov until the region ``mu_min <= mu_tot <= mu_max`` is bounded.

    :func:`~.lens_mesh.build_closed_lens_mesh` runs first, unchanged. Then,
    while the region reaches outside the valid cells of its source mesh
    (:func:`magnification_region`), and fewer than ``max_iters`` further
    extensions have run, :func:`~.lens_mesh.extend_lens_mesh` grows the fov
    by ``growth``, and the source mesh is extended over the new image of
    the fov boundary rather than built again. When ``mu_min`` is at or
    below the far-field ``mu_tot``, no fov bounds the region.

    Parameters
    ----------
    raytrace, jacobian, fov, init_res, min_img_sep, max_depth, x0, y0, centers, device, batch_size:
        As for :func:`~.lens_mesh.build_lens_mesh`.
    mu_min, mu_max:
        As for :func:`magnification_region`.
    growth: float
        Factor each extension asks to multiply the fov by.
    max_iters: int
        Most extensions to run in each of the two loops.

    Returns
    -------
    mesh: LensMesh
    curves: CriticalCurvesAndCaustics
        The last mesh's.
    region: MagnificationRegion
    """
    mesh, curves, region = _build_closed_magnification_region(
        raytrace,
        jacobian,
        to_mesh(fov),
        to_mesh(init_res),
        to_mesh(min_img_sep),
        float(mu_min),
        None if mu_max is None else float(mu_max),
        to_mesh(max_depth),
        growth=to_mesh(growth),
        max_iters=max_iters,
        x0=to_mesh(x0),
        y0=to_mesh(y0),
        centers=to_mesh(centers),
        device=device,
        batch_size=batch_size,
    )
    return to_user(mesh, device), to_user(curves, device), to_user(region, device)


def _build_closed_magnification_region(
    raytrace,
    jacobian,
    fov,
    init_res,
    min_img_sep,
    mu_min,
    mu_max,
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
    """:func:`build_closed_magnification_region` on ``mesh_backend`` arrays, on the build device."""
    mesh, curves = _build_closed_lens_mesh(
        raytrace,
        jacobian,
        fov,
        init_res,
        min_img_sep,
        max_depth,
        growth=growth,
        max_iters=max_iters,
        x0=x0,
        y0=y0,
        centers=centers,
        device=device,
        batch_size=batch_size,
    )
    src = _source_mesh(mesh, curves, mu_min, mu_max)
    region = _region(src, mu_min, mu_max)
    for _ in range(max_iters):
        if not math.isnan(region.area):
            break
        mesh = _extend_lens_mesh(
            mesh,
            raytrace,
            jacobian,
            growth * lattice_fov(mesh.lattice),
            device=device,
            batch_size=batch_size,
        )
        curves = _critical_curves_and_caustics(mesh)
        src = _extend_source_mesh(src, mesh, curves, mu_min, mu_max)
        region = _region(src, mu_min, mu_max)
    if math.isnan(region.area):
        warn(
            f"The region still reaches the edge of the valid source domain after "
            f"{max_iters} extension(s), at fov={lattice_fov(mesh.lattice):g}; raise "
            "max_iters or growth to grow it further. No fov bounds a region whose "
            "mu_min is at or below the far-field total magnification.",
            stacklevel=3,
        )
    return mesh, curves, region
