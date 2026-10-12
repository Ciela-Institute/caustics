"""
Triangle maths shared by both meshes, and small array helpers.

The red-refinement child ordering and its integer group tables, the smallest
singular value that scales the lens criterion, the minimum angle closure
compares, the containment tests -- closed and by a fill rule -- nearest
boundary points and barycentric coordinates every source-plane lookup
reads, the segment crossing and winding tests, and the area of a union of
closed polylines. Nothing here calls the lens or knows about a mesh.
"""

from .mesh_backend import map_arrays, mesh_backend

# Indices into the stacked six-point array [theta1, theta2, theta3, m1, m2, m3]
# giving the four children in the orientation-preserving order
#   C_1 = (t1, m3, m2)  C_2 = (t2, m1, m3)  C_3 = (t3, m2, m1)  C_4 = (m1, m2, m3)
# with m_i the midpoint opposite theta_i.
CHILD_VERTEX_INDICES = ((0, 5, 4), (1, 3, 5), (2, 4, 3), (3, 4, 5))

# The two triangles splitting the unit cell along its (0,0)-(1,1) diagonal, both
# positively oriented.
ROOT_SHAPES = (
    ((0, 0), (1, 1), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
)

# `CHILD_VERTEX_INDICES` as one int64 array, to gather all four children at once.
_CHILD_VERTEX_INDEX_TABLE = mesh_backend.as_array(
    CHILD_VERTEX_INDICES, dtype=mesh_backend.int64
)


def shape_matrix(tri):
    """
    Edge matrix ``P = [v1 - v0 | v2 - v0]`` of a triangle.

    Parameters
    ----------
    tri: ArrayLike
        Triangle vertices, shape ``(..., 3, 2)``.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        Shape ``(..., 2, 2)``, the two edge vectors as columns.

        *Unit: arcsec*
    """
    return mesh_backend.stack(
        (tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :]), dim=-1
    )


def _derive_child_matrices():
    """Recover ``M_k`` from the child ordering, where ``P_k = (1/2) P M_k``."""
    parent = mesh_backend.as_array(
        [[0.0, 0.0], [3.0, 2.0], [-1.0, 5.0]], dtype=mesh_backend.float64
    )
    t1, t2, t3 = parent
    six = mesh_backend.stack([t1, t2, t3, (t2 + t3) / 2, (t3 + t1) / 2, (t1 + t2) / 2])
    P = shape_matrix(parent)
    rows = []
    for idx in CHILD_VERTEX_INDICES:
        Pk = shape_matrix(six[list(idx), :])
        rows.append(
            mesh_backend.long(
                mesh_backend.round(2.0 * (mesh_backend.linalg.inv(P) @ Pk))
            )
        )
    return mesh_backend.stack(rows)


def child_matrix_tables():
    """
    Fixed tables for red refinement.

    Red refinement gives ``P_k = (1/2) P M_k`` with ``M_k`` a fixed integer matrix
    per child index, so ``P`` at level ``d`` is ``2**-d * h0 * R @ G[c]`` for one
    element ``G[c]`` of the group the four ``M_k`` generate,
    ``{+-I, +-M_2, +-M_3}``. Both root shapes lie in one orbit, so six classes
    cover every triangle at every level.

    Returns
    -------
    M: ArrayLike
        ``(4, 2, 2)`` int64 child matrices; every ``det M_k == +1``.
    G: ArrayLike
        ``(6, 2, 2)`` int64 group elements in canonical order.
    COMPOSE: ArrayLike
        ``(6, 4)`` int64 with ``G[COMPOSE[c, k]] == G[c] @ M[k]``.
    PINV0: ArrayLike
        ``(6, 2, 2)`` float64 with ``PINV0[c] == inv(R @ G[c])``, ``R`` the class-0
        root shape. A triangle of class ``c`` at level ``d`` has
        ``inv(P_child_k) == (2**(d + 1) / h0) * PINV0[COMPOSE[c, k]]``.
    ROOT_CLASS: ArrayLike
        ``(2,)`` int64 class index of each entry of :data:`ROOT_SHAPES`.
    """
    M = _derive_child_matrices()
    eye = mesh_backend.eye(2, dtype=mesh_backend.int64)
    G = mesh_backend.stack([eye, M[1], M[2], -eye, -M[1], -M[2]])

    lookup = {
        tuple(mesh_backend.to_numpy(G[c]).reshape(-1).tolist()): c for c in range(6)
    }
    compose_rows = []
    for c in range(6):
        row = []
        for k in range(4):
            product = G[c] @ M[k]
            row.append(
                lookup[tuple(mesh_backend.to_numpy(product).reshape(-1).tolist())]
            )
        compose_rows.append(row)
    COMPOSE = mesh_backend.as_array(compose_rows, dtype=mesh_backend.int64)

    R = shape_matrix(mesh_backend.as_array(ROOT_SHAPES[0], dtype=mesh_backend.float64))
    PINV0 = mesh_backend.stack(
        [
            mesh_backend.linalg.inv(
                R @ mesh_backend.to(G[c], dtype=mesh_backend.float64)
            )
            for c in range(6)
        ]
    )

    root_class_values = []
    for shape in ROOT_SHAPES:
        target = shape_matrix(mesh_backend.as_array(shape, dtype=mesh_backend.float64))
        root_class_values.append(
            next(
                c
                for c in range(6)
                if bool(
                    mesh_backend.allclose(
                        R @ mesh_backend.to(G[c], dtype=mesh_backend.float64), target
                    )
                )
            )
        )
    ROOT_CLASS = mesh_backend.as_array(root_class_values, dtype=mesh_backend.int64)
    return M, G, COMPOSE, PINV0, ROOT_CLASS


_, _, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()


def area2(tri):
    """
    Twice the signed area of each triangle.

    Parameters
    ----------
    tri: ArrayLike
        ``(K, 3, 2)`` vertices.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(K,)``, positive for a counter-clockwise triangle.

        *Unit: arcsec^2*
    """
    P = shape_matrix(tri)
    return P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]


def build_device():
    """The backend's default device, where every build runs."""
    return mesh_backend.device(mesh_backend.zeros((0,)))


def to_device(value, device):
    """
    ``value`` on ``device``: an array, or a NamedTuple of arrays and NamedTuples, field by field.

    Anything else is returned as it is.
    """
    return map_arrays(value, lambda a: mesh_backend.to(a, device=device))


def sorted_find(sorted_values, values):
    """
    Where each of ``values`` sits in the ascending ``sorted_values``, and whether it is there.

    Returns
    -------
    pos: ArrayLike
        ``values.shape`` int64 ``searchsorted`` position, clamped to the last
        entry; zeros when ``sorted_values`` is empty.
    found: ArrayLike
        ``values.shape`` bool, ``sorted_values[pos] == values``.
    """
    device = mesh_backend.device(values)
    n = sorted_values.shape[0]
    if n == 0:
        pos = mesh_backend.zeros(values.shape, dtype=mesh_backend.int64, device=device)
        return pos, pos != 0
    pos = mesh_backend.clamp(mesh_backend.searchsorted(sorted_values, values), 0, n - 1)
    return pos, sorted_values[pos] == values


def is_member(sorted_values, values):
    """True where each of ``values`` is in the ascending ``sorted_values``."""
    return sorted_find(sorted_values, values)[1]


def csr_offsets(counts):
    """CSR offsets ``(n + 1,)`` of blocks of ``counts`` ``(n,)`` rows each."""
    zero = mesh_backend.zeros(
        (1,), dtype=mesh_backend.int64, device=mesh_backend.device(counts)
    )
    return mesh_backend.concatenate(
        (zero, mesh_backend.cumsum(mesh_backend.long(counts), dim=0)), dim=0
    )


def expand(counts, total=None):
    """
    The block of each entry of consecutive blocks of ``counts`` ``(n,)`` int64 entries, and its position in the block.

    ``total`` is ``counts.sum()`` where the caller already read it on the host.

    Returns
    -------
    block: ArrayLike
        ``(total,)`` int64.
    within: ArrayLike
        ``(total,)`` int64.
    """
    device = mesh_backend.device(counts)
    int64 = mesh_backend.int64
    if total is None:
        total = int(mesh_backend.to_numpy(mesh_backend.sum(counts)))
    block = mesh_backend.repeat(
        mesh_backend.arange(counts.shape[0], dtype=int64, device=device), counts, axis=0
    )
    within = mesh_backend.arange(
        total, dtype=int64, device=device
    ) - mesh_backend.repeat(mesh_backend.cumsum(counts, dim=0) - counts, counts, axis=0)
    return block, within


def sigma_min_2x2(A):
    """
    Smallest singular value of a batch of 2x2 matrices, in closed form.

    A source-plane displacement pulls back to at most ``||dbeta|| / sigma_min``
    in the lens plane. With ``F = ||A||_F**2`` and ``D = |det A|``,
    ``(sigma_max +- sigma_min)**2 = F +- 2D``; ``F - 2D`` is clipped at zero,
    since rounding takes it an ulp negative for a near-conformal ``A``, and
    ``sigma_min = D / sigma_max`` stays accurate as ``A`` nears singular.

    Parameters
    ----------
    A: ArrayLike
        Shape ``(..., 2, 2)``, float64.

    Returns
    -------
    ArrayLike
        Shape ``(...)``. Exactly ``0.0`` where ``A == 0``; ``NaN`` where ``A`` is
        non-finite, so the criterion fails closed.
    """
    F = mesh_backend.sum(A**2, dim=(-2, -1))
    D = mesh_backend.abs(A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0])
    sigma_max = (
        mesh_backend.sqrt(F + 2 * D)
        + mesh_backend.sqrt(mesh_backend.clamp(F - 2 * D, 0.0, None))
    ) / 2
    zero = sigma_max == 0
    # Double `where` keeps the division away from 0/0 without masking a NaN input:
    # a NaN sigma_max fails the `== 0` test, so NaN reaches the output.
    return mesh_backend.where(zero, 0.0, D / mesh_backend.where(zero, 1.0, sigma_max))


def triangle_weights(tri, beta):
    """
    The three cross products that give containment and barycentric coordinates.

    ``w_i`` is the cross product of the two edges of the sub-triangle
    opposite vertex ``i``, so ``w1 + w2 + w3`` is twice the signed area ``d``
    and ``w / d`` are the barycentric coordinates. Two triangles sharing an
    edge form its weight from the same operands in opposite order, so the
    two weights are exactly negated and no point falls through the seam.

    Parameters
    ----------
    tri: ArrayLike
        Source-plane triangle vertices, shape ``(T, 3, 2)``.

        *Unit: arcsec*

    beta: ArrayLike
        Query points, one per candidate, shape ``(T, 2)``.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        Shape ``(T, 3)``.
    """
    q1 = tri[:, 0, :] - beta
    q2 = tri[:, 1, :] - beta
    q3 = tri[:, 2, :] - beta
    return mesh_backend.stack(
        (
            q2[:, 0] * q3[:, 1] - q2[:, 1] * q3[:, 0],
            q3[:, 0] * q1[:, 1] - q3[:, 1] * q1[:, 0],
            q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0],
        ),
        dim=-1,
    )


def contains(w):
    """
    Containment test: all three weights share a sign, zeros counting as inside.

    Source-plane triangles flip orientation across critical curves, so the
    test asks for one shared sign rather than a positive one. A point on a
    shared edge is inside both triangles.

    Parameters
    ----------
    w: ArrayLike
        Weights from :func:`triangle_weights`, shape ``(T, 3)``.

    Returns
    -------
    ArrayLike
        Shape ``(T,)`` bool.
    """
    nonneg = (w[..., 0] >= 0) & (w[..., 1] >= 0) & (w[..., 2] >= 0)
    nonpos = (w[..., 0] <= 0) & (w[..., 1] <= 0) & (w[..., 2] <= 0)
    return nonneg | nonpos


def contains_half_open(tri, w):
    """
    Containment by a top-left fill rule: a point on an edge two same-orientation triangles share is in exactly one.

    :func:`contains` puts a point on a shared edge in both triangles. Here a
    point strictly inside is inside, and a point on an edge only when the
    triangle owns the edge: taken counter-clockwise, a left edge, pointing
    down, or a top edge, horizontal and pointing left. Two triangles of one
    orientation run a shared edge in opposite directions, so exactly one owns
    it, and of a fan about a vertex exactly one owns both edges at it.
    Source-plane triangles flip orientation across critical curves, so each
    is first turned counter-clockwise; one of zero area contains nothing.
    :func:`triangle_weights` negates shared-edge weights exactly, so both
    triangles agree on which points lie on the edge.

    Parameters
    ----------
    tri: ArrayLike
        ``(K, 3, 2)`` triangle vertices.

        *Unit: arcsec*
    w: ArrayLike
        ``(K, 3)`` their :func:`triangle_weights` at one point each.

    Returns
    -------
    ArrayLike
        ``(K,)`` bool.
    """
    sigma = mesh_backend.sign(w[:, 0] + w[:, 1] + w[:, 2])
    w = w * mesh_backend.unsqueeze(sigma, -1)
    # Edge i, opposite vertex i, runs from vertex i + 1 to vertex i + 2.
    edge = (tri[:, [2, 0, 1]] - tri[:, [1, 2, 0]]) * sigma[:, None, None]
    owned = (edge[..., 1] < 0) | ((edge[..., 1] == 0) & (edge[..., 0] < 0))
    inside = (w > 0) | ((w == 0) & owned)
    return (sigma != 0) & mesh_backend.all(inside, dim=1)


def edge_nearest(tri, beta):
    """
    The point of each triangle's boundary nearest each point: its distance and barycentric coordinates.

    Each edge is projected on, the projection clamped to the edge, and the
    nearest of the three wins, ties taking the first edge in the order
    ``(1, 2), (2, 3), (3, 1)``; a zero-length edge is its own endpoint.
    Outside a triangle the boundary point is the triangle's point nearest
    the point. A NaN point gives a NaN distance.

    Parameters
    ----------
    tri: ArrayLike
        ``(K, 3, 2)`` triangle vertices.

        *Unit: arcsec*
    beta: ArrayLike
        ``(K, 2)`` points, one per triangle.

        *Unit: arcsec*

    Returns
    -------
    dist: ArrayLike
        ``(K,)`` distance to the boundary.

        *Unit: arcsec*
    bary: ArrayLike
        ``(K, 3)`` barycentric coordinates of the nearest boundary point, in
        the simplex.
    """
    zero = mesh_backend.zeros_like(beta[:, 0])
    dist, bary = None, None
    for i in range(3):
        j = (i + 1) % 3
        a, ab = tri[:, i], tri[:, j] - tri[:, i]
        length2 = mesh_backend.sum(ab * ab, dim=-1)
        t = mesh_backend.sum((beta - a) * ab, dim=-1) / mesh_backend.where(
            length2 > 0, length2, 1.0
        )
        t = mesh_backend.clamp(t, 0.0, 1.0)
        d = mesh_backend.norm(beta - (a + mesh_backend.unsqueeze(t, -1) * ab), dim=-1)
        columns = [zero, zero, zero]
        columns[i], columns[j] = 1.0 - t, t
        edge_bary = mesh_backend.stack(columns, dim=-1)
        if dist is None:
            dist, bary = d, edge_bary
            continue
        nearer = d < dist
        dist = mesh_backend.where(nearer, d, dist)
        bary = mesh_backend.where(mesh_backend.unsqueeze(nearer, -1), edge_bary, bary)
    return dist, bary


def _side(p, q, r):
    """Sign of the cross product ``(q - p) x (r - p)``: which side of ``p -> q`` each ``r`` is on."""
    d, e = q - p, r - p
    return mesh_backend.sign(d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0])


def segments_cross(a0, a1, b0, b1):
    """
    Whether segment ``a0 a1`` meets segment ``b0 b1``, touches counting.

    Each segment's ends lie on opposite sides of the other's line, or on it,
    and the two bounding boxes overlap -- which rejects collinear segments
    that do not. Signs are multiplied, never the cross products, so nothing
    underflows.

    Parameters
    ----------
    a0, a1, b0, b1: ArrayLike
        ``(..., 2)`` segment ends, broadcast against each other: ``(P, 1, 2)``
        ends against ``(1, Q, 2)`` give every pair.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        bool, the broadcast shape without its last axis.
    """
    straddle = (_side(a0, a1, b0) * _side(a0, a1, b1) <= 0) & (
        _side(b0, b1, a0) * _side(b0, b1, a1) <= 0
    )
    lo_a, hi_a = mesh_backend.minimum(a0, a1), mesh_backend.maximum(a0, a1)
    lo_b, hi_b = mesh_backend.minimum(b0, b1), mesh_backend.maximum(b0, b1)
    overlap = mesh_backend.all((lo_a <= hi_b) & (lo_b <= hi_a), dim=-1)
    return straddle & overlap


def winding_number(polygon, points):
    """
    Winding number of a closed polyline about each point.

    Segment ``p_k p_{k+1}``, the last closing back to the first, adds +1 where
    it crosses the horizontal ray to the right of a point going up with the
    point on its left, and -1 going down with the point on its right. A
    segment holds its lower end and not its upper one, so a ray through a
    vertex counts once. A point on the polyline gets the count of one side or
    the other. A segment with a NaN end, and a NaN point, cross nothing.

    Parameters
    ----------
    polygon: ArrayLike
        ``(M, 2)`` vertices, in order.

        *Unit: arcsec*
    points: ArrayLike
        ``(B, 2)``.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(B,)`` int64.
    """
    a = mesh_backend.unsqueeze(polygon, 0)
    b = mesh_backend.unsqueeze(mesh_backend.roll(polygon, -1, 0), 0)
    p = mesh_backend.unsqueeze(points, 1)
    side = _side(a, b, p)
    up = (a[..., 1] <= p[..., 1]) & (b[..., 1] > p[..., 1]) & (side > 0)
    down = (a[..., 1] > p[..., 1]) & (b[..., 1] <= p[..., 1]) & (side < 0)
    return mesh_backend.sum(mesh_backend.long(up) - mesh_backend.long(down), dim=1)


def ring_next(offsets, n_points):
    """
    Each point's polyline, and the point after it; a polyline's last point is followed by its first.

    Parameters
    ----------
    offsets: ArrayLike
        ``(C + 1,)`` int64 CSR offsets of ``n_points`` points in ``C`` polylines.
    n_points: int

    Returns
    -------
    curve: ArrayLike
        ``(n_points,)`` int64 polyline of each point.
    nxt: ArrayLike
        ``(n_points,)`` int64 the point after it.
    """
    int64 = mesh_backend.int64
    device = mesh_backend.device(offsets)
    curve, _ = expand(offsets[1:] - offsets[:-1], n_points)
    point = mesh_backend.arange(n_points, dtype=int64, device=device)
    last = offsets[1:][curve] - 1
    return curve, mesh_backend.where(point == last, offsets[:-1][curve], point + 1)


def union_area(points, offsets):
    """
    Area of the union of the nonzero-winding interiors of closed polylines.

    A point is covered when some polyline winds about it
    (:func:`winding_number` nonzero), whichever its orientation, so a part a
    polyline winds about twice counts once, and so does a part two
    polylines both cover. The plane is cut into vertical strips at every
    vertex's x. At each strip's mid-line the segments spanning it are sorted
    by height; walking up, each polyline's winding changes by one at each
    of its crossings, and the covered length is the sum of the gaps between
    crossings where some polyline's winding is nonzero. The area is the sum
    of each strip's width times that length.

    Exact in every strip no two segments cross inside: the covered length
    is linear across it. In a strip where two cross, the error is at most
    the strip's width times the change in covered length across it, of
    order the width squared, so polylines whose segments are short, as
    traced curves are, make it negligible. Vertical and zero-length
    segments have no width and are dropped.

    Parameters
    ----------
    points: ArrayLike
        ``(P, 2)`` vertices, polyline by polyline, each closing back to its
        first point.

        *Unit: arcsec*
    offsets: ArrayLike
        ``(C + 1,)`` int64 CSR offsets.

    Returns
    -------
    float

        *Unit: arcsec^2*
    """
    int64, f64 = mesh_backend.int64, mesh_backend.float64
    device = mesh_backend.device(points)
    if points.shape[0] == 0:
        return 0.0
    points = mesh_backend.to(points, dtype=f64)
    curve, nxt = ring_next(offsets, points.shape[0])
    a, b = points, points[nxt]
    keep = mesh_backend.flatnonzero(a[:, 0] != b[:, 0])
    a, b, curve = a[keep], b[keep], curve[keep]
    if a.shape[0] == 0:
        return 0.0
    x0 = mesh_backend.minimum(a[:, 0], b[:, 0])
    x1 = mesh_backend.maximum(a[:, 0], b[:, 0])
    rightward = mesh_backend.where(b[:, 0] > a[:, 0], 1, -1)
    edges = mesh_backend.unique(mesh_backend.concatenate((x0, x1), dim=0))
    first = mesh_backend.searchsorted(edges, x0)
    span = mesh_backend.searchsorted(edges, x1) - first
    total = int(mesh_backend.to_numpy(mesh_backend.sum(span)))
    seg, within = expand(span, total)
    strip = first[seg] + within
    mid = 0.5 * (edges[strip] + edges[strip + 1])
    y = a[seg, 1] + (mid - a[seg, 0]) / (b[seg, 0] - a[seg, 0]) * (
        b[seg, 1] - a[seg, 1]
    )
    order = mesh_backend.lexsort([y, strip])
    strip, seg, y = strip[order], seg[order], y[order]
    # A closed polyline crosses a strip's mid-line with a net winding of
    # zero, so one running sum over every strip in turn gives each
    # polyline's winding just above each crossing.
    rows = mesh_backend.arange(total, dtype=int64, device=device)
    step = mesh_backend.zeros((total, offsets.shape[0] - 1), dtype=int64, device=device)
    step = mesh_backend.fill_at_indices(step, (rows, curve[seg]), rightward[seg])
    covered = mesh_backend.any(mesh_backend.cumsum(step, dim=0) != 0, dim=1)
    gap = (strip[1:] == strip[:-1]) & covered[:-1]
    width = edges[strip[:-1] + 1] - edges[strip[:-1]]
    area = mesh_backend.sum(mesh_backend.where(gap, (y[1:] - y[:-1]) * width, 0.0))
    return float(mesh_backend.to_numpy(area))


def sanitize_bary(w, d):
    """
    Normalize weights to barycentric coordinates, clamping then falling back.

    A hit passed containment, so its coordinates lie in ``[0, 1]`` up to
    rounding, which the clip removes. Where ``d`` is so near zero that the
    clipped coordinates are non-finite or all zero, the centroid stands in.

    Parameters
    ----------
    w: ArrayLike
        Weights of the hits, shape ``(K, 3)``.
    d: ArrayLike
        Twice the signed source-plane area of each hit leaf, shape ``(K,)``.

    Returns
    -------
    ArrayLike
        Shape ``(K, 3)``, guaranteed to lie in the simplex.
    """
    bary = mesh_backend.clamp(w / mesh_backend.unsqueeze(d, -1), 0.0, 1.0)
    total = bary[..., 0] + bary[..., 1] + bary[..., 2]
    ok = (
        (total > 0)
        & mesh_backend.isfinite(bary[..., 0])
        & mesh_backend.isfinite(bary[..., 1])
        & mesh_backend.isfinite(bary[..., 2])
    )
    safe = mesh_backend.where(ok, total, mesh_backend.ones_like(total))
    normed = bary / mesh_backend.unsqueeze(safe, -1)
    third = mesh_backend.ones_like(bary) / 3
    return mesh_backend.where(mesh_backend.unsqueeze(ok, -1), normed, third)


def min_angle(tri):
    """
    Smallest interior angle of each triangle, in radians.

    Parameters
    ----------
    tri: ArrayLike
        Shape ``(n, 3, 2)``.

    Returns
    -------
    ArrayLike
        Shape ``(n,)``. Degenerate triangles give ``0.0`` rather than ``NaN``.
    """
    a = mesh_backend.norm(tri[:, 2] - tri[:, 1], dim=-1)
    b = mesh_backend.norm(tri[:, 0] - tri[:, 2], dim=-1)
    c = mesh_backend.norm(tri[:, 1] - tri[:, 0], dim=-1)
    cosines = mesh_backend.stack(
        (
            (b * b + c * c - a * a) / (2 * b * c),
            (c * c + a * a - b * b) / (2 * c * a),
            (a * a + b * b - c * c) / (2 * a * b),
        ),
        dim=1,
    )
    angles = mesh_backend.arccos(mesh_backend.clamp(cosines, -1.0, 1.0))
    return mesh_backend.nan_to_num(mesh_backend.min(angles, dim=1))
