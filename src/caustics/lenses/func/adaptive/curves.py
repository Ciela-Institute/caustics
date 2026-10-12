"""
Critical curves and caustics, traced through a lens mesh's critical band.

:func:`~caustics.lenses.func.adaptive.build_lens_mesh` keeps, as
``mesh.critical_band``, every ``max_level`` leaf ``det A`` changes sign across,
with ``det A`` and the image at its six samples. Those six samples are the
corners of the leaf's four red-split children, so ``det A`` at them defines a
piecewise-linear field on the children, and this module traces its zero set:
the critical curves, and -- interpolating the images the same way -- their
caustics. Nothing here calls the lens.

Every crossing lies on a child edge whose two ends have opposite classes, so
where ``det A`` is continuous the true critical curve crosses the same edge:
each lens-plane point is within one child edge, half a leaf edge, of the
curve.

A mesh built with ``centers`` also holds holes around them
(:class:`~caustics.lenses.func.adaptive.CenterHoles`): the lens map can jump
at a lens center, so the curves are cut at each hole's circle and re-joined
along the stored hole curve (:func:`join_at_holes`), and each hole curve
that is a pseudo-caustic is added whole, as a loop of its own
(:func:`pseudo_caustics`). That too is a function of the mesh alone.

:func:`caustic_area` gives the area the curves bound: the union of the
interiors of their source-plane polylines.
"""

import math
from typing import NamedTuple

from ....backend_obj import ArrayLike, backend
from .mesh_backend import mesh_backend, to_mesh, to_user
from .geometry import _CHILD_VERTEX_INDEX_TABLE, csr_offsets, expand, union_area


class CriticalCurvesAndCaustics(NamedTuple):
    """
    Critical curves, caustics and pseudo-caustics, one ordered polyline per curve, CSR.

    Curve ``c`` is rows ``offsets[c]:offsets[c + 1]`` of ``lens``, ``source``
    and ``hole``. Most points are crossings of ``det A = 0``: a critical-curve
    point in ``lens``, its caustic point in ``source``. On a mesh with holes,
    a curve that reaches a hole follows the hole's circle clockwise to the
    next curve leaving it (:func:`join_at_holes`); those points lie on the
    circle in ``lens`` and on the hole curve in ``source``. After every other
    curve, each hole whose ``pseudo_caustic`` is True adds its whole hole
    curve as a loop of its own, flagged in ``pseudo_caustic``
    (:func:`pseudo_caustics`): its pseudo-caustic, arcs of which a curve
    joined at the hole also holds.

    Travel keeps ``det A > 0`` on the left, except round a pseudo-caustic,
    which follows its hole's samples in ascending angle, counter-clockwise
    in the lens plane: ``det A`` usually has one sign all round a hole. A
    curve the fov or missing data cuts is open; ``closed`` marks the loops,
    whose last point joins back to their first. The order of the curves, and
    where a loop starts, is deterministic but has no physical meaning. Where
    ``det A`` is exactly zero at a sample, as when a curve runs through
    lattice points, consecutive points can coincide: zero-length segments
    are kept.

    Parameters
    ----------
    lens: ArrayLike
        ``(P, 2)`` float64 critical-curve or hole-circle points.

        *Unit: arcsec*
    source: ArrayLike
        ``(P, 2)`` float64 caustic or hole-curve points.

        *Unit: arcsec*
    offsets: ArrayLike
        ``(C + 1,)`` int64 CSR offsets, ``offsets[0] == 0``.
    closed: ArrayLike
        ``(C,)`` bool, True where the curve is a loop.
    hole: ArrayLike
        ``(P,)`` int64: -1 at a crossing of ``det A = 0``, otherwise the
        index into the mesh's ``holes`` of the hole whose circle the point
        lies on.
    pseudo_caustic: ArrayLike
        ``(C,)`` bool, True where the curve is a hole's pseudo-caustic.
    """

    lens: ArrayLike
    source: ArrayLike
    offsets: ArrayLike
    closed: ArrayLike
    hole: ArrayLike
    pseudo_caustic: ArrayLike


def _sorted_pair(a, b):
    """``(min, max)`` of two index arrays, shape ``(K,) -> (K, 2)``."""
    return mesh_backend.stack(
        (mesh_backend.minimum(a, b), mesh_backend.maximum(a, b)), dim=-1
    )


def triangle_segments(tri, positive):
    """
    The oriented crossing segment of each triangle whose corners are not all of one class.

    A triangle with mixed classes has exactly one odd corner ``i``, the lone
    positive or the lone negative, and with ``(i, j, k)`` cyclic the boundary
    between the classes crosses its edges ``(i, j)`` and ``(k, i)``. The
    segment keeps the positive side on its left: from ``(i, j)`` to ``(k, i)``
    when the odd corner is positive, and the reverse when it is negative. On
    positively oriented triangles sharing an edge, the segment ending on it
    is always met by one starting there, which is what makes the segments
    chain (:func:`chain_segments`).

    Parameters
    ----------
    tri: ArrayLike
        ``(K, 3)`` int64 sample indices, positively oriented.
    positive: ArrayLike
        ``(S,)`` bool class of each sample.

    Returns
    -------
    start: ArrayLike
        ``(M,)`` segments' starting edges, shape ``(M, 2)`` int64 sample
        pairs, smaller index first.
    end: ArrayLike
        Their ending edges, shape ``(M, 2)``, in the same form.
    """
    pos = positive[tri]
    n_positive = mesh_backend.sum(mesh_backend.long(pos), dim=1)
    mixed = mesh_backend.flatnonzero((n_positive == 1) | (n_positive == 2))
    tri, pos = tri[mixed], pos[mixed]
    odd_positive = n_positive[mixed] == 1
    # The odd corner is the one True entry of `pos` where one corner is
    # positive, and the one False entry where two are.
    odd_mask = mesh_backend.where(mesh_backend.unsqueeze(odd_positive, -1), pos, ~pos)
    odd = mesh_backend.argmax(mesh_backend.long(odd_mask), 1)
    rows = mesh_backend.arange(tri.shape[0], dtype=mesh_backend.int64)
    p_i = tri[rows, odd]
    p_j = tri[rows, (odd + 1) % 3]
    p_k = tri[rows, (odd + 2) % 3]
    e_ij = _sorted_pair(p_i, p_j)
    e_ki = _sorted_pair(p_k, p_i)
    flip = mesh_backend.unsqueeze(odd_positive, -1)
    return mesh_backend.where(flip, e_ij, e_ki), mesh_backend.where(flip, e_ki, e_ij)


def child_segments(samples, det):
    """
    The oriented zero-crossing segment of ``det A`` on each red-split child.

    :func:`triangle_segments` on the four children of every band leaf. A
    sample's class is ``det >= 0``: an exact zero counts as positive, which
    gives every child a definite crossing.

    Parameters
    ----------
    samples: ArrayLike
        ``(F, 6)`` int64, ``CriticalBand.samples``.
    det: ArrayLike
        ``(S,)`` float64, ``CriticalBand.det``.

    Returns
    -------
    start: ArrayLike
        ``(K,)`` segments' starting edges, shape ``(K, 2)`` int64 sample
        pairs, smaller index first.
    end: ArrayLike
        Their ending edges, shape ``(K, 2)``, in the same form.
    """
    kids = samples[:, _CHILD_VERTEX_INDEX_TABLE].reshape(-1, 3)
    return triangle_segments(kids, det >= 0)


def edge_zeros(edges, field, planes):
    """
    Where a piecewise-linear ``field`` crosses zero on each edge, in every plane.

    The zero of the linear interpolant from ``p = edges[:, 0]`` to
    ``q = edges[:, 1]``: ``t = field[p] / (field[p] - field[q])``, applied to
    each plane's positions. The ends must have opposite classes, so the
    denominator is never zero, and ``t == 0`` exactly when ``field[p] == 0``.
    Computed in float64 and returned at each plane's dtype.

    Parameters
    ----------
    edges: ArrayLike
        ``(N, 2)`` int64 sample pairs.
    field: ArrayLike
        ``(S,)`` the field at each sample.
    planes: Tuple[ArrayLike, ...]
        ``(S, 2)`` positions of the samples, one array per plane.

        *Unit: arcsec*

    Returns
    -------
    Tuple[ArrayLike, ...]
        ``(N, 2)`` zero of each edge, one array per plane.

        *Unit: arcsec*
    """
    p, q = edges[:, 0], edges[:, 1]
    t = mesh_backend.unsqueeze(field[p] / (field[p] - field[q]), -1)
    points = []
    for plane in planes:
        x = mesh_backend.to(plane, dtype=mesh_backend.float64)
        points.append(mesh_backend.to(x[p] + t * (x[q] - x[p]), dtype=plane.dtype))
    return tuple(points)


def chain_order(succ):
    """
    Order nodes along the paths and cycles of a successor array.

    Vectorized list ranking by pointer jumping: ``ceil(log2(K))`` rounds of
    gathers, with no Python loop over nodes.

    1. Every node follows its successor, a tail following itself, while
       taking the minimum node id seen. Once ``2**rounds >= K``, a node on a
       path points at its tail, which has no successor, while a node on a
       cycle still points into the cycle, whose minimum it now holds.
    2. Every path starts at its node without a predecessor. Every cycle is
       cut at its smallest node, which becomes its start.
    3. Wyllie's list ranking over predecessors then gives each node its start
       and its distance from it.

    Successors are unique, so the scatter filling predecessors never repeats
    an index.

    Parameters
    ----------
    succ: ArrayLike
        ``(K,)`` int64 successor of each node, ``-1`` for none. No node may
        have two predecessors.

    Returns
    -------
    order: ArrayLike
        ``(K,)`` int64 permutation listing the nodes curve by curve, each in
        travel order.
    offsets: ArrayLike
        ``(C + 1,)`` int64 CSR offsets into ``order``.
    closed: ArrayLike
        ``(C,)`` bool, True where the curve is a cycle.
    """
    k = succ.shape[0]
    device = mesh_backend.device(succ)
    int64 = mesh_backend.int64
    if k == 0:
        return (
            mesh_backend.zeros((0,), dtype=int64, device=device),
            mesh_backend.zeros((1,), dtype=int64, device=device),
            mesh_backend.zeros((0,), dtype=mesh_backend.bool, device=device),
        )
    node = mesh_backend.arange(k, dtype=int64, device=device)
    has_next = succ >= 0
    pred = mesh_backend.fill_at_indices(
        mesh_backend.zeros((k,), dtype=int64, device=device) - 1,
        succ[has_next],
        node[has_next],
    )
    rounds = math.ceil(math.log2(max(k, 2)))

    nxt = mesh_backend.where(has_next, succ, node)
    low = node
    for _ in range(rounds):
        low = mesh_backend.minimum(low, low[nxt])
        nxt = nxt[nxt]
    on_cycle = has_next[nxt]

    start = (pred < 0) | (on_cycle & (low == node))
    pred = mesh_backend.where(start, -1, pred)

    back = mesh_backend.where(pred >= 0, pred, node)
    dist = mesh_backend.long(pred >= 0)
    for _ in range(rounds):
        dist = dist + dist[back]
        back = back[back]

    # `lexsort`'s last key is primary: group by start node, then travel order.
    order = mesh_backend.lexsort([dist, back])
    starts = mesh_backend.flatnonzero(start)
    counts = mesh_backend.bincount(back, minlength=k)[starts]
    return order, csr_offsets(counts), on_cycle[starts]


def chain_segments(start, end, n_samples):
    """
    Chain oriented segments, each from one sample-pair edge to another, into curves.

    Each distinct edge is one node, keyed by its sample pair; each segment
    links its starting node to its ending one; :func:`chain_order` orders
    the result. Used by :func:`trace_band`.

    Parameters
    ----------
    start, end: ArrayLike
        ``(K, 2)`` int64 sample pairs, smaller index first, as
        :func:`triangle_segments` returns them.
    n_samples: int
        Number of samples, bounding every index, for the node keys.

    Returns
    -------
    edges: ArrayLike
        ``(N, 2)`` int64 node edges, curve by curve in travel order.
    offsets: ArrayLike
        ``(C + 1,)`` int64 CSR offsets into ``edges``.
    closed: ArrayLike
        ``(C,)`` bool, True where the curve is a loop.
    """
    int64 = mesh_backend.int64
    device = mesh_backend.device(start)
    k = start.shape[0]
    if k == 0:
        return (
            mesh_backend.zeros((0, 2), dtype=int64, device=device),
            mesh_backend.zeros((1,), dtype=int64, device=device),
            mesh_backend.zeros((0,), dtype=mesh_backend.bool, device=device),
        )
    s = n_samples
    keys = mesh_backend.concatenate(
        (start[:, 0] * s + start[:, 1], end[:, 0] * s + end[:, 1]), dim=0
    )
    nodes, inverse = mesh_backend.unique(keys, return_inverse=True)
    frm, to = inverse[:k], inverse[k:]
    n = nodes.shape[0]
    succ = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n,), dtype=int64, device=device) - 1, frm, to
    )
    order, offsets, closed = chain_order(succ)
    ordered = nodes[order]
    return mesh_backend.stack((ordered // s, ordered % s), dim=-1), offsets, closed


def _no_curves(band):
    """The :class:`CriticalCurvesAndCaustics` of a band with no crossing."""
    device = mesh_backend.device(band.det)
    return CriticalCurvesAndCaustics(
        lens=mesh_backend.zeros((0, 2), dtype=band.lens.dtype, device=device),
        source=mesh_backend.zeros((0, 2), dtype=band.source.dtype, device=device),
        offsets=mesh_backend.zeros((1,), dtype=mesh_backend.int64, device=device),
        closed=mesh_backend.zeros((0,), dtype=mesh_backend.bool, device=device),
        hole=mesh_backend.zeros((0,), dtype=mesh_backend.int64, device=device),
        pseudo_caustic=mesh_backend.zeros((0,), dtype=mesh_backend.bool, device=device),
    )


def trace_band(band):
    """
    Trace the zero set of ``det A`` through a :class:`CriticalBand`.

    Each child's segment (:func:`child_segments`) is chained into curves by
    :func:`chain_segments`, each distinct crossing edge one node, keyed by
    its sample pair. Each node's point is computed once, so the two children
    sharing an edge agree on it exactly.

    Parameters
    ----------
    band: CriticalBand

    Returns
    -------
    CriticalCurvesAndCaustics
        ``hole`` is -1 at every point and ``pseudo_caustic`` False on every
        curve: tracing knows nothing of holes; :func:`join_at_holes` and
        :func:`pseudo_caustics` apply them.
    """
    start, end = child_segments(band.samples, band.det)
    if start.shape[0] == 0:
        return _no_curves(band)
    edges, offsets, closed = chain_segments(start, end, band.lens.shape[0])
    lens_points, source_points = edge_zeros(edges, band.det, (band.lens, band.source))
    device = mesh_backend.device(band.det)
    return CriticalCurvesAndCaustics(
        lens=lens_points,
        source=source_points,
        offsets=offsets,
        closed=closed,
        hole=mesh_backend.zeros(
            (lens_points.shape[0],), dtype=mesh_backend.int64, device=device
        )
        - 1,
        pseudo_caustic=mesh_backend.zeros_like(closed),
    )


def _turn(angle):
    """``angle`` wrapped into ``[0, 2 pi)``."""
    two_pi = 2.0 * math.pi
    return angle - two_pi * mesh_backend.floor(angle / two_pi)


def join_at_holes(curves, holes):
    """
    Cut traced curves at hole circles and re-join them along the hole curves.

    Inside a hole the lens map can jump, so a traced curve's points there
    mean nothing. They are replaced:

    1. Every point strictly inside a hole's disk is dropped, and so is a lone
       point between two points of the same hole. This cuts each curve into
       segments.
    2. A segment that starts right after a dropped point departs from that
       hole's circle; one that ends right before a dropped point arrives at
       it. A traced step from one hole straight to another is a *bridge*: a
       segment with no point of its own, departing the first hole and
       arriving at the second.
    3. Where the ends around a circle, sorted by angle, alternate between
       arriving and departing, each arriving end is joined to the next end
       clockwise -- the side ``det A > 0`` is on -- through the hole's stored
       samples strictly between them. Otherwise the hole is left unjoined.
    4. The segments are chained through their joins (:func:`chain_order`).
       Curves that never reach a hole keep :func:`trace_band`'s order; a
       curve left without a point is dropped.

    Every loop then bounds a ``det A > 0`` region with the holes cut out, and
    no caustic cuts across a hole curve, except at a hole whose ends do not
    alternate; at a hole the fov cuts, until
    :func:`~caustics.lenses.func.adaptive.extend_lens_mesh` grows the fov
    over it; where ``max_depth`` left leaves larger than a hole; and where a
    traced segment clips a disk with neither end inside it.

    Parameters
    ----------
    curves: CriticalCurvesAndCaustics
        As :func:`trace_band` returns them.
    holes: CenterHoles

    Returns
    -------
    CriticalCurvesAndCaustics
    """
    n_points = curves.lens.shape[0]
    n_holes = holes.centers.shape[0]
    if n_points == 0 or n_holes == 0:
        return curves
    device = mesh_backend.device(curves.lens)
    int64, f64 = mesh_backend.int64, mesh_backend.float64
    lens = mesh_backend.to(curves.lens, dtype=f64)
    centers = mesh_backend.to(holes.centers, dtype=f64)

    # 1. Tag every point strictly inside a hole's disk; disks are disjoint.
    inside = mesh_backend.norm(
        mesh_backend.unsqueeze(lens, 1) - mesh_backend.unsqueeze(centers, 0), dim=-1
    ) < mesh_backend.unsqueeze(holes.radius, 0)
    tag = mesh_backend.where(
        mesh_backend.any(inside, dim=1),
        mesh_backend.argmax(mesh_backend.long(inside), 1),
        -1,
    )

    # Per-point curve bookkeeping.
    counts = curves.offsets[1:] - curves.offsets[:-1]
    curve, _ = expand(counts, n_points)
    first = curves.offsets[:-1][curve]
    last = curves.offsets[1:][curve] - 1
    point = mesh_backend.arange(n_points, dtype=int64, device=device)
    curve_closed = curves.closed[curve]
    prev = mesh_backend.where(point == first, last, point - 1)
    nxt = mesh_backend.where(point == last, first, point + 1)

    # 1b. A lone untagged point between two points of one hole, left by the
    # tracer's zigzag along its circle, counts as inside it.
    tagged = tag >= 0
    has_prev = (point != first) | curve_closed
    has_next = (point != last) | curve_closed
    lone = ~tagged & has_prev & has_next & (tag[prev] >= 0) & (tag[prev] == tag[nxt])
    tag = mesh_backend.where(lone, tag[prev], tag)
    tagged = tag >= 0

    # 2. Rotate each closed curve that enters a hole to start right after a
    # tagged run, so that no run of untagged points wraps round its end.
    after_run = mesh_backend.flatnonzero(~tagged & tagged[prev] & curve_closed)
    start = mesh_backend.copy(curves.offsets[:-1])
    if after_run.shape[0]:
        run_curve = curve[after_run]
        lead = mesh_backend.concatenate(
            (
                mesh_backend.ones((1,), dtype=mesh_backend.bool, device=device),
                run_curve[1:] != run_curve[:-1],
            ),
            dim=0,
        )
        start = mesh_backend.fill_at_indices(start, run_curve[lead], after_run[lead])
    perm = first + (start[curve] - first + point - first) % counts[curve]
    tag = tag[perm]
    tagged = tag >= 0

    # 3. Segments: the maximal runs of untagged points of each curve, already
    # numbered by their first traced point in rotated order.
    is_first = ~tagged & (
        (point == first) | tagged[mesh_backend.clamp(point - 1, 0, None)]
    )
    is_last = ~tagged & (
        (point == last) | tagged[mesh_backend.clamp(point + 1, None, n_points - 1)]
    )
    seg_first = mesh_backend.flatnonzero(is_first)
    seg_last = mesh_backend.flatnonzero(is_last)
    if seg_first.shape[0] == 0:
        return CriticalCurvesAndCaustics(
            lens=curves.lens[:0],
            source=curves.source[:0],
            offsets=mesh_backend.zeros((1,), dtype=int64, device=device),
            closed=curves.closed[:0],
            hole=curves.hole[:0],
            pseudo_caustic=curves.pseudo_caustic[:0],
        )

    # 3b. Bridges: a step from one hole straight to another is a segment from
    # the step's second point back to its first, with no point of its own.
    # They are numbered after every real segment, so a cycle holding a real
    # segment still starts at one.
    tag_next = tag[mesh_backend.clamp(point + 1, None, n_points - 1)]
    has_segment = mesh_backend.bincount(curve[seg_first], minlength=counts.shape[0]) > 0
    bridge = mesh_backend.flatnonzero(
        (point != last)
        & tagged
        & (tag_next >= 0)
        & (tag_next != tag)
        & has_segment[curve]
    )
    seg_first = mesh_backend.concatenate((seg_first, bridge + 1), dim=0)
    seg_last = mesh_backend.concatenate((seg_last, bridge), dim=0)
    n_seg = seg_first.shape[0]
    seg_closed = curves.closed[curve[seg_first]]
    starts_curve = seg_first == first[seg_first]
    ends_curve = seg_last == last[seg_last]
    before = mesh_backend.where(starts_curve, last[seg_first], seg_first - 1)
    after = mesh_backend.where(ends_curve, first[seg_last], seg_last + 1)
    departs = mesh_backend.where(starts_curve & ~seg_closed, -1, tag[before])
    arrives = mesh_backend.where(ends_curve & ~seg_closed, -1, tag[after])
    # A closed curve that never enters a hole is one segment, its own successor.
    whole = seg_closed & starts_curve & ends_curve
    succ = mesh_backend.where(
        whole, mesh_backend.arange(n_seg, dtype=int64, device=device), -1
    )

    # 4. On each circle, join every arriving end to the next end clockwise.
    lens_rot = lens[perm]
    hole_offsets = mesh_backend.to_numpy(holes.offsets).tolist()
    arcs = {}
    for h in range(n_holes):
        arr = mesh_backend.flatnonzero(arrives == h)
        dep = mesh_backend.flatnonzero(departs == h)
        m = arr.shape[0] + dep.shape[0]
        if m == 0:
            continue
        seg = mesh_backend.concatenate((arr, dep), dim=0)
        arriving = mesh_backend.concatenate(
            (
                mesh_backend.ones(
                    (arr.shape[0],), dtype=mesh_backend.bool, device=device
                ),
                mesh_backend.zeros(
                    (dep.shape[0],), dtype=mesh_backend.bool, device=device
                ),
            ),
            dim=0,
        )
        rel = lens_rot[mesh_backend.concatenate((seg_last[arr], seg_first[dep]), dim=0)]
        rel = rel - centers[h]
        phi = _turn(mesh_backend.arctan2(rel[:, 1], rel[:, 0]))
        o = mesh_backend.argsort(phi)
        seg, arriving, phi = seg[o], arriving[o], phi[o]
        if m % 2 or bool(
            mesh_backend.any(arriving == mesh_backend.roll(arriving, 1, 0))
        ):
            continue
        k_arr = mesh_backend.flatnonzero(arriving)
        k_dep = (k_arr - 1) % m
        succ = mesh_backend.fill_at_indices(succ, seg[k_arr], seg[k_dep])
        lo, hi = hole_offsets[h], hole_offsets[h + 1]
        angle = holes.angle[lo:hi]
        for ka, kd in zip(
            mesh_backend.to_numpy(k_arr).tolist(), mesh_backend.to_numpy(k_dep).tolist()
        ):
            span = _turn(phi[ka] - phi[kd])
            delta = _turn(phi[ka] - angle)
            pick = mesh_backend.flatnonzero((delta > 0) & (delta < span))
            pick = pick[mesh_backend.argsort(delta[pick])]
            arcs[int(mesh_backend.to_numpy(seg[ka]))] = lo + pick
    # 5. Chain the segments through their joins into curves.
    order, seg_offsets, closed = chain_order(succ)

    # 6. Assemble: every segment's points, then its join's arc.
    pool_lens = mesh_backend.concatenate(
        (curves.lens[perm], mesh_backend.to(holes.lens, dtype=curves.lens.dtype)), dim=0
    )
    pool_source = mesh_backend.concatenate(
        (curves.source[perm], mesh_backend.to(holes.source, dtype=curves.source.dtype)),
        dim=0,
    )
    hole_of_sample, _ = expand(
        holes.offsets[1:] - holes.offsets[:-1], holes.lens.shape[0]
    )
    pool_hole = mesh_backend.concatenate(
        (
            mesh_backend.zeros((n_points,), dtype=int64, device=device) - 1,
            hole_of_sample,
        ),
        dim=0,
    )
    first_host = mesh_backend.to_numpy(seg_first).tolist()
    last_host = mesh_backend.to_numpy(seg_last).tolist()
    pieces, sizes = [], []
    for s in mesh_backend.to_numpy(order).tolist():
        pieces.append(
            mesh_backend.arange(
                first_host[s], last_host[s] + 1, dtype=int64, device=device
            )
        )
        size = last_host[s] + 1 - first_host[s]
        if s in arcs:
            pieces.append(n_points + arcs[s])
            size += int(arcs[s].shape[0])
        sizes.append(size)
    gather = mesh_backend.concatenate(pieces, dim=0)
    csum = csr_offsets(mesh_backend.as_array(sizes, dtype=int64, device=device))
    offsets = csum[seg_offsets]
    # A curve of bridges alone whose joins inserted no sample has no point.
    kept = mesh_backend.flatnonzero(offsets[1:] > offsets[:-1])
    if kept.shape[0] < closed.shape[0]:
        offsets = mesh_backend.concatenate((offsets[:1], offsets[1:][kept]), dim=0)
        closed = closed[kept]
    return CriticalCurvesAndCaustics(
        lens=pool_lens[gather],
        source=pool_source[gather],
        offsets=offsets,
        closed=closed,
        hole=pool_hole[gather],
        pseudo_caustic=mesh_backend.zeros_like(closed),
    )


def pseudo_caustics(holes):
    """
    The pseudo-caustic of every hole whose ``pseudo_caustic`` is True, each as a closed curve.

    A curve is its hole's samples as stored, in ascending angle: the circle
    in ``lens``, the hole curve in ``source``, and the hole's index in
    ``hole``. The image count changes across the whole hole curve, whether
    or not a critical curve reaches the hole, so the whole of it is a curve,
    even where :func:`join_at_holes` joins arcs of it into another.

    Parameters
    ----------
    holes: CenterHoles

    Returns
    -------
    CriticalCurvesAndCaustics
        ``pseudo_caustic`` True on every curve.
    """
    device = mesh_backend.device(holes.offsets)
    counts = holes.offsets[1:] - holes.offsets[:-1]
    hole, _ = expand(counts, holes.lens.shape[0])
    rows = mesh_backend.flatnonzero(holes.pseudo_caustic[hole])
    flagged = mesh_backend.flatnonzero(holes.pseudo_caustic)
    loops = mesh_backend.ones(
        (flagged.shape[0],), dtype=mesh_backend.bool, device=device
    )
    return CriticalCurvesAndCaustics(
        lens=holes.lens[rows],
        source=holes.source[rows],
        offsets=csr_offsets(counts[flagged]),
        closed=loops,
        hole=hole[rows],
        pseudo_caustic=loops,
    )


def _append(curves, more):
    """``curves`` followed by ``more``, curve by curve."""
    cat = mesh_backend.concatenate
    return CriticalCurvesAndCaustics(
        lens=cat((curves.lens, mesh_backend.to(more.lens, dtype=curves.lens.dtype))),
        source=cat(
            (curves.source, mesh_backend.to(more.source, dtype=curves.source.dtype))
        ),
        offsets=cat((curves.offsets, more.offsets[1:] + curves.offsets[-1])),
        closed=cat((curves.closed, more.closed)),
        hole=cat((curves.hole, more.hole)),
        pseudo_caustic=cat((curves.pseudo_caustic, more.pseudo_caustic)),
    )


def critical_curves_and_caustics(mesh):
    """
    Critical curves, caustics and pseudo-caustics of the lens a mesh was built from.

    The zero set of ``det A`` traced through the critical band's red-split
    children, each crossing interpolated linearly on a child edge whose ends
    straddle it -- within half a finest leaf edge of the true curve -- and its
    caustic point interpolated the same way. On a mesh with holes the curves
    are cut at each hole's circle and re-joined along its hole curve
    (:func:`join_at_holes`), and each hole curve that is a pseudo-caustic
    follows them as a loop of its own (:func:`pseudo_caustics`). A curve
    ends where the band does: at the fov boundary or next to a non-finite
    leaf. Consecutive points can coincide where ``det A`` is exactly zero
    at a sample; such zero-length segments are kept. No lens call is made.

    Parameters
    ----------
    mesh: LensMesh

    Returns
    -------
    CriticalCurvesAndCaustics
    """
    return to_user(
        _critical_curves_and_caustics(to_mesh(mesh)),
        backend.device(mesh.vertices_lens),
    )


def _critical_curves_and_caustics(mesh):
    """:func:`critical_curves_and_caustics` on ``mesh_backend`` arrays."""
    curves = trace_band(mesh.critical_band)
    if mesh.holes.centers.shape[0] == 0:
        return curves
    return _append(join_at_holes(curves, mesh.holes), pseudo_caustics(mesh.holes))


def caustic_area(curves):
    """
    Area of the region the caustics and pseudo-caustics bound.

    The union of the interiors of every curve's ``source`` polyline --
    caustics, pseudo-caustics and curves joined through holes -- each the
    points it winds about (:func:`~.geometry.union_area`). For a closed lens
    mesh this is the multiply imaged region, to within the curves' accuracy.

    Parameters
    ----------
    curves: CriticalCurvesAndCaustics
        As :func:`critical_curves_and_caustics` returns them.

    Returns
    -------
    float
        NaN when any curve is open: the fov or missing data cut it, and the
        region it bounds is unknown.

        *Unit: arcsec^2*
    """
    return _caustic_area(to_mesh(curves))


def _caustic_area(curves):
    """:func:`caustic_area` on ``mesh_backend`` arrays."""
    if not bool(mesh_backend.all(curves.closed)):
        return math.nan
    return union_area(curves.source, curves.offsets)
