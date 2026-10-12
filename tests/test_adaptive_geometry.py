"""Triangle maths, the red-split tables, the lattice, and the spatial index."""

import math

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend
from caustics.lenses.func.adaptive.criterion import (
    child_shape_matrices,
    converged_from_deviation,
    midpoint_deviation,
    parity_from_children,
)
from caustics.lenses.func.adaptive.geometry import (
    CHILD_VERTEX_INDICES,
    COMPOSE,
    PINV0,
    ROOT_CLASS,
    ROOT_SHAPES,
    area2,
    child_matrix_tables,
    component_labels,
    contains,
    contains_half_open,
    csr_offsets,
    edge_nearest,
    expand,
    is_member,
    map_chunks,
    min_angle,
    ring_next,
    sanitize_bary,
    segments_cross,
    shape_matrix,
    sigma_min_2x2,
    sorted_find,
    to_device,
    triangle_weights,
    union_area,
    winding_number,
)
from caustics.lenses.func.adaptive.index import (
    CELLS_PER_BOX,
    KEY_BITS,
    as_points,
    build_index,
    index_cells,
    index_hits,
)
from caustics.lenses.func.adaptive.lattice import (
    check_lattice_keys,
    depth_floor,
    extend_lattice,
    initial_triangles,
    lattice_fov,
    lattice_h0,
    lattice_ij_from_key,
    lattice_init_res,
    lattice_key,
    lattice_xy,
    make_lattice,
    midpoint_ij,
    ring_triangles,
    warn_depth_limited,
)

from adaptive_maps import (
    assert_hits_equal,
    assert_same,
    brute_hits,
    f64,
    i64,
    index_cell,
    index_cell_ranges,
    index_point_cells,
    to_np,
)

RNG = np.random.default_rng(20260918)


def _arr(x):
    return f64(x)


def _i64(x):
    return i64(x)


def signed_area(tri):
    """Twice the signed area of a (..., 3, 2) numpy triangle."""
    e1 = tri[..., 1, :] - tri[..., 0, :]
    e2 = tri[..., 2, :] - tri[..., 0, :]
    return e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]


def np_red_split(tri):
    """Split a (..., 3, 2) numpy triangle into (..., 4, 3, 2) children, canonical order."""
    t1, t2, t3 = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    six = np.stack([t1, t2, t3, (t2 + t3) / 2, (t3 + t1) / 2, (t1 + t2) / 2], axis=-2)
    idx = np.asarray(CHILD_VERTEX_INDICES)
    return six[..., idx, :].reshape(*tri.shape[:-2], 4, 3, 2)


def _six_points(tri):
    t1, t2, t3 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    mid = np.stack([(t2 + t3) / 2, (t3 + t1) / 2, (t1 + t2) / 2], axis=1)
    return tri, mid


def _grid(n):
    """Every point of an ``(n + 1) x (n + 1)`` lattice, shape ``((n + 1)**2, 2)``."""
    axis = np.arange(n + 1)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)


def _same_lattice(a, b):
    return (a.level, a.n, a.scale, a.origin) == (
        b.level,
        b.n,
        b.scale,
        b.origin,
    ) and np.array_equal(to_np(a.lo), to_np(b.lo))


def test_sigma_min_of_zero_matrix_is_exactly_zero():
    A = np.zeros((4, 2, 2))
    got = to_np(sigma_min_2x2(_arr(A)))
    assert (got == 0.0).all()
    assert not np.isnan(got).any()


def test_sigma_min_propagates_nan():
    A = np.zeros((1, 2, 2))
    A[0, 0, 0] = np.nan
    assert np.isnan(to_np(sigma_min_2x2(_arr(A)))).all()


def test_sigma_min_matches_svd():
    A = RNG.normal(size=(2000, 2, 2))
    expected = np.linalg.svd(A, compute_uv=False)[:, -1]
    assert np.allclose(to_np(sigma_min_2x2(_arr(A))), expected, rtol=1e-9, atol=1e-12)


def test_sigma_min_near_and_exactly_singular():
    a = RNG.normal(size=(500, 2))
    perp = np.stack([-a[:, 1], a[:, 0]], axis=1)
    # Rows offset perpendicularly, so det == eps * |a|^2 is genuinely small.
    # Scaling one row instead would give a rank-1 matrix with det exactly zero,
    # which tests the degenerate branch, not the near-degenerate one.
    for eps in (1e-4, 1e-7):
        A = np.stack([a, a + eps * perp], axis=1)
        got = to_np(sigma_min_2x2(_arr(A)))
        expected = np.linalg.svd(A, compute_uv=False)[:, -1]
        assert np.isfinite(got).all()
        assert np.allclose(got, expected, rtol=1e-5, atol=0.0)
    exact = np.stack([a, 2.0 * a], axis=1)  # rank 1: det is exactly 0 in IEEE
    assert (to_np(sigma_min_2x2(_arr(exact))) == 0.0).all()


def test_sigma_min_handles_conformal_ulp_negativity():
    """F - 2D is exactly zero in R but goes negative by an ulp in float.

    Without the clip this returns NaN for a near-circularly-symmetric lens core,
    which fail-closed then refines to max_level. See spec section 2.2.
    """
    ab = RNG.normal(size=(200000, 2)).astype(np.float32)
    A = np.empty((ab.shape[0], 2, 2), dtype=np.float32)
    A[:, 0, 0] = ab[:, 0]
    A[:, 0, 1] = ab[:, 1]
    A[:, 1, 0] = -ab[:, 1]
    A[:, 1, 1] = ab[:, 0]
    F = (A.astype(np.float32) ** 2).sum(axis=(-2, -1))
    D = np.abs(A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0])
    assert (F - 2 * D < 0).any(), "test fixture no longer exercises the guard"
    got = to_np(sigma_min_2x2(_arr(A.astype(np.float64))))
    assert np.isfinite(got).all()


def test_group_tables_close_and_have_order_six():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    assert M.dtype == mesh_backend.int64
    assert G.dtype == mesh_backend.int64
    assert COMPOSE.dtype == mesh_backend.int64
    assert PINV0.dtype == mesh_backend.float64
    assert ROOT_CLASS.dtype == mesh_backend.int64
    M, G, COMPOSE, ROOT_CLASS = (
        to_np(M),
        to_np(G),
        to_np(COMPOSE),
        to_np(ROOT_CLASS),
    )
    assert M.shape == (4, 2, 2) and G.shape == (6, 2, 2)
    dets = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    assert (dets == 1).all(), "every child matrix must have det +1"
    # G is closed under right multiplication by every M, and COMPOSE indexes it
    for c in range(6):
        for k in range(4):
            assert (G[c] @ M[k] == G[COMPOSE[c, k]]).all()
    # exactly six distinct elements
    assert len({G[c].tobytes() for c in range(6)}) == 6
    # both root shapes lie in one orbit
    R0 = to_np(shape_matrix(_arr(ROOT_SHAPES[0])))
    R1 = to_np(shape_matrix(_arr(ROOT_SHAPES[1])))
    assert np.allclose(R0 @ G[ROOT_CLASS[1]], R1)


def test_root_shapes_are_positively_oriented():
    for shape in ROOT_SHAPES:
        assert signed_area(np.asarray(shape, dtype=np.float64)) > 0


@pytest.mark.parametrize("flip", [False, True])
def test_children_preserve_orientation_and_quarter_the_area(flip):
    tri = RNG.normal(size=(64, 3, 2))
    if flip:
        tri = tri[:, ::-1, :]
    parent = signed_area(tri)
    kids = signed_area(np_red_split(tri))
    assert np.allclose(kids, parent[:, None] / 4, rtol=0, atol=1e-12)


def test_pinv_table_scales_by_two_to_the_level():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    COMPOSE, PINV0, ROOT_CLASS = (
        to_np(COMPOSE),
        to_np(PINV0),
        to_np(ROOT_CLASS),
    )
    h0 = 0.05
    for shape_idx in (0, 1):
        root = np.asarray(ROOT_SHAPES[shape_idx], dtype=np.float64) * h0
        cls = ROOT_CLASS[shape_idx]
        tri, level = root, 0
        for _ in range(5):
            kids = np_red_split(tri)
            for k in range(4):
                direct = to_np(mesh_backend.linalg.inv(shape_matrix(_arr(kids[k]))))
                table = (2.0 ** (level + 1) / h0) * PINV0[COMPOSE[cls, k]]
                assert np.allclose(direct, table, rtol=1e-12, atol=1e-12)
            tri, cls, level = kids[0], COMPOSE[cls, 0], level + 1


def test_the_inverse_table_recovers_the_affine_map():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    COMPOSE, PINV0, ROOT_CLASS = (
        to_np(COMPOSE),
        to_np(PINV0),
        to_np(ROOT_CLASS),
    )
    h0 = 0.05
    tri = np.asarray(ROOT_SHAPES[1], dtype=np.float64) * h0
    cls, level = ROOT_CLASS[1], 0
    kids = np_red_split(tri)
    lens_map = RNG.normal(size=(2, 2))
    for k in range(4):
        q = kids[k] @ lens_map.T
        Q = to_np(shape_matrix(_arr(q)))
        table = Q @ ((2.0 ** (level + 1) / h0) * PINV0[COMPOSE[cls, k]])
        assert np.allclose(table, lens_map, rtol=1e-10, atol=1e-12)


def test_converged_from_deviation_fails_closed():
    r = np.zeros((3, 3))
    assert to_np(converged_from_deviation(_arr(r), _arr([1.0, 1.0, 1.0]), 1.0)).all()
    # NaN must take the split branch, not the converged branch
    assert not to_np(
        converged_from_deviation(_arr(r), _arr(np.full(3, np.nan)), 1.0)
    ).any()
    # s == 0 with r == 0 gives 0 < 0, False, split -- the conservative direction
    assert not to_np(converged_from_deviation(_arr(r), _arr(np.zeros(3)), 1.0)).any()


def test_child_shape_matrices_match_shape_matrix_of_each_child():
    """Q_k must be exactly the edge matrix of child k, bit for bit.

    Both forms subtract the same operands in the same order, so this is an
    equality check and not an allclose: a reassociation that changed rounding
    would change the sign of a near-degenerate det and silently move the parity
    verdict, which is the whole thing this kernel decides.
    """
    rng = np.random.default_rng(0)
    beta_v = rng.normal(size=(5, 3, 2))
    beta_m = rng.normal(size=(5, 3, 2))
    Q = to_np(child_shape_matrices(_arr(beta_v), _arr(beta_m)))
    assert Q.shape == (5, 4, 2, 2)
    six = np.concatenate([beta_v, beta_m], axis=1)
    for k, idx in enumerate(CHILD_VERTEX_INDICES):
        expected = to_np(shape_matrix(_arr(six[:, list(idx)])))
        assert np.array_equal(Q[:, k], expected)


def test_parity_from_children_accepts_a_constant_sign():
    Q = np.tile(np.eye(2), (2, 4, 1, 1))
    assert to_np(parity_from_children(_arr(Q))).tolist() == [
        True,
        True,
    ]


def test_parity_from_children_rejects_a_sign_change():
    Q = np.tile(np.eye(2), (1, 4, 1, 1))
    Q[0, 2] = np.array([[0.0, 1.0], [1.0, 0.0]])  # det -1 among three det +1
    assert to_np(parity_from_children(_arr(Q))).tolist() == [False]


def test_parity_from_children_condemns_a_lone_degenerate_child():
    Q = np.tile(np.eye(2), (1, 4, 1, 1))
    Q[0, 3] = 0.0  # sign 0 differs from +1
    assert to_np(parity_from_children(_arr(Q))).tolist() == [False]


def test_parity_from_children_passes_an_entirely_degenerate_triangle():
    """All four dets exactly zero is a constant sign, so there is no parity
    *change* and the triangle passes. Spec section 4.6: condemning this case
    would be the deviation test in disguise, which is out of scope. A kappa == 1
    sheet is the fixture that reaches it.
    """
    Q = np.zeros((1, 4, 2, 2))
    assert to_np(parity_from_children(_arr(Q))).tolist() == [True]


@pytest.mark.parametrize(
    "determinants",
    [
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0.0, 0.0, 0.0],
    ],
)
def test_parity_from_children_rejects_nan_determinants(determinants):
    Q = np.zeros((1, 4, 2, 2))
    Q[0, :, 0, 0] = determinants
    Q[0, :, 1, 1] = 1.0
    assert to_np(parity_from_children(_arr(Q))).tolist() == [False]


def test_midpoint_deviation_pairs_midpoint_with_opposite_edge():
    v = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    m = np.array([[[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]]])  # exact affine images
    assert np.allclose(to_np(midpoint_deviation(_arr(v), _arr(m))), 0.0)
    shifted = m.copy()
    shifted[0, 1] += np.array([0.0, 0.25])
    got = to_np(midpoint_deviation(_arr(v), _arr(shifted)))
    assert np.allclose(got, [[0.0, 0.25, 0.0]])


def test_weights_sum_to_twice_the_signed_area():
    tri = RNG.normal(size=(500, 3, 2))
    beta = RNG.normal(size=(500, 2))
    w = to_np(triangle_weights(_arr(tri), _arr(beta)))
    P = to_np(shape_matrix(_arr(tri)))
    d = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    assert np.allclose(w.sum(axis=1), d, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("flip", [False, True])
def test_containment_agrees_with_barycentric_truth_on_both_parities(flip):
    tri = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    if flip:
        tri = tri[:, ::-1, :]
    pts = RNG.uniform(-0.5, 1.5, size=(4000, 2))
    tiled = np.repeat(tri, len(pts), axis=0)
    hit = to_np(contains(triangle_weights(_arr(tiled), _arr(pts))))
    truth = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] + pts[:, 1] <= 1)
    assert (hit == truth).all()


def test_shared_edge_weights_are_exactly_negated():
    """Two leaves meeting on an edge must not both reject a point on that edge."""
    verts = np.array([[0.0, 0.0], [1.0, 0.3], [0.4, 1.0], [1.3, 1.2]])
    left = verts[[0, 1, 2]][None]
    right = verts[[1, 3, 2]][None]  # shares the edge (1, 2), opposite traversal
    beta = np.array([[0.62, 0.71]])
    wl = to_np(triangle_weights(_arr(left), _arr(beta)))
    wr = to_np(triangle_weights(_arr(right), _arr(beta)))
    # left's weight opposite vertex 0 uses the (1, 2) edge; right's opposite
    # vertex 3 uses (2, 1). They must be bit-exact negatives.
    assert wl[0, 0] == -wr[0, 1]


# Pairs of segments, ((a0, a1), (b0, b1)), and whether they meet.
SEGMENT_PAIRS = [
    pytest.param(((0, 0), (2, 2)), ((0, 2), (2, 0)), True, id="proper-crossing"),
    pytest.param(((0, 0), (1, 0)), ((2, -1), (2, 1)), False, id="short-of-the-line"),
    pytest.param(((0, 0), (1, 0)), ((1, 0), (1, 1)), True, id="shared-end"),
    pytest.param(((0, 0), (2, 0)), ((1, 0), (1, 1)), True, id="end-on-interior"),
    pytest.param(((0, 0), (1, 0)), ((0, 1), (1, 1)), False, id="parallel"),
    pytest.param(((0, 0), (2, 0)), ((1, 0), (3, 0)), True, id="collinear-overlap"),
    pytest.param(((0, 0), (1, 0)), ((2, 0), (3, 0)), False, id="collinear-apart"),
]


@pytest.mark.parametrize("a, b, want", SEGMENT_PAIRS)
def test_segments_cross_counts_a_touch_and_nothing_apart(a, b, want):
    """Either segment, either way round, gives the same answer."""
    cases = [(a, b), (a[::-1], b), (b, a), (b[::-1], a[::-1])]
    a0, a1, b0, b1 = (_arr([c[k][e] for c in cases]) for k in (0, 1) for e in (0, 1))
    assert to_np(segments_cross(a0, a1, b0, b1)).tolist() == [want] * 4


def test_segments_cross_broadcasts_to_a_table_of_every_pair():
    a = np.array([[[0, 0], [2, 2]], [[0, 0], [1, 0]]], dtype=np.float64)
    b = np.array(
        [[[0, 2], [2, 0]], [[1, 0], [1, 1]], [[5, 5], [6, 6]]], dtype=np.float64
    )
    got = segments_cross(
        _arr(a[:, None, 0]),
        _arr(a[:, None, 1]),
        _arr(b[None, :, 0]),
        _arr(b[None, :, 1]),
    )
    assert to_np(got).tolist() == [[True, True, False], [False, True, False]]


def test_sanitize_bary_is_always_in_the_simplex():
    """The clip path keeps wide-but-finite w/d ratios inside the simplex.

    The 1e-300 scaling stresses the clamp with extreme ratios; it does not reach
    the centroid fallback, because 1e-300 is a normal double and w/d is invariant
    under a uniform scaling. The fallback is covered by the two tests below.
    """
    w = RNG.normal(size=(1000, 3)) * 1e-300
    d = w.sum(axis=1)
    bary = to_np(sanitize_bary(_arr(w), _arr(d)))
    assert np.isfinite(bary).all()
    assert (bary >= 0).all() and (bary <= 1).all()
    assert np.allclose(bary.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_sanitize_bary_falls_back_to_the_centroid_on_total_degeneracy():
    w = np.zeros((4, 3))
    d = np.zeros(4)
    bary = to_np(sanitize_bary(_arr(w), _arr(d)))
    assert np.array_equal(bary, np.full((4, 3), 1.0 / 3.0))


def test_sanitize_bary_selects_per_row_between_normalized_and_centroid():
    """Degenerate and ordinary rows in one call must be resolved independently.

    Both other sanitize_bary tests are all-or-nothing -- every row ordinary, or
    every row degenerate -- so neither would catch an implementation that decided
    the fallback once for the whole batch instead of per row.
    """
    w = _arr([[3.0, 1.5, 1.5], [0.0, 0.0, 0.0], [2.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    d = _arr([6.0, 0.0, 4.0, 0.0])
    bary = to_np(sanitize_bary(w, d))
    assert np.allclose(bary[[0, 2]], [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    assert np.allclose(bary[[1, 3]], 1.0 / 3.0)
    assert np.allclose(bary.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_sanitize_bary_recovers_ordinary_coordinates():
    tri = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]])
    beta = np.array([[0.5, 0.75]])
    w = triangle_weights(_arr(tri), _arr(beta))
    d = _arr([2.0 * 3.0])
    bary = to_np(sanitize_bary(w, d))
    assert np.allclose(bary @ tri[0], beta, rtol=1e-12, atol=1e-14)


def test_depth_floor_matches_the_size_criterion():
    assert depth_floor(5.0 / 100, 10.0) == 0
    d = depth_floor(5.0 / 100, 1e-3)
    l0 = np.sqrt(2) * 5.0 / 100
    assert l0 / 2**d <= 1e-3 < l0 / 2 ** (d - 1)


def test_lattice_key_roundtrip_and_geometry():
    lat = make_lattice(4.0, 0.0, 0.0, 4, 3)
    assert lat.n == 32
    ij_np = np.array([[0, 0], [32, 32], [7, 19]], dtype=np.int64)
    ij = mesh_backend.as_array(ij_np, dtype=mesh_backend.int64)
    key = lattice_key(lat, ij)
    assert to_np(lattice_ij_from_key(lat, key)).tolist() == ij_np.tolist()
    assert np.allclose(to_np(lattice_xy(lat, ij[0])), [-2.0, -2.0])
    assert np.allclose(to_np(lattice_xy(lat, ij[1])), [2.0, 2.0])


def test_widening_the_lattice_does_not_move_any_vertex():
    """Bit-identical coordinates, not merely close ones.

    `scale' = fov / (2n)` equals `fl(fov / n) / 2` exactly, because binary
    floating point is scale-invariant under powers of two, and `(2 * ij) *
    scale'` then rounds the same exact real as `ij * scale`. If this ever fails,
    the widened lattice has perturbed the frozen mesh's geometry and every
    downstream bit-exactness argument in the module is void.
    """
    fov, init_res, max_level = 4.0, 4, 3
    narrow = make_lattice(fov, 0.0, 0.0, init_res, max_level)
    wide = make_lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    assert wide.n == 2 * narrow.n
    assert wide.level == max_level + 1

    ij_np = np.stack(
        np.meshgrid(np.arange(narrow.n + 1), np.arange(narrow.n + 1), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    ij = mesh_backend.as_array(ij_np, dtype=mesh_backend.int64)
    assert np.array_equal(
        to_np(lattice_xy(narrow, ij)),
        to_np(lattice_xy(wide, 2 * ij)),
    )


def test_initial_triangles_tile_the_square_and_are_positively_oriented():
    """Ported from `test_adaptive_mesh.py`: geometric properties that the
    oracle-comparison test above does not check -- full square coverage and
    a consistent positive orientation -- rather than element-wise equality
    with the oracle.
    """
    _, _, _, _, root_class = child_matrix_tables()
    init_res, max_level = 4, 2
    ij, cls = initial_triangles(init_res, max_level, root_class)
    assert ij.shape == (2 * init_res**2, 3, 2)
    P = shape_matrix(mesh_backend.to(ij, dtype=mesh_backend.float64))
    area = P[..., 0, 0] * P[..., 1, 1] - P[..., 0, 1] * P[..., 1, 0]
    assert bool(mesh_backend.all(area > 0))
    step = 1 << max_level
    assert np.isclose(float(mesh_backend.sum(area)) / 2, (init_res * step) ** 2)
    assert set(to_np(cls).tolist()) == set(to_np(root_class).tolist())


def test_midpoints_are_exact_integers_and_opposite_their_vertex():
    ij = np.array([[[0, 0], [4, 0], [0, 4]]], dtype=np.int64)
    m = midpoint_ij(_i64(ij))
    assert to_np(m)[0].tolist() == [[2, 2], [0, 2], [2, 0]]


def test_midpoints_are_exact_at_max_level_on_the_widened_lattice():
    """The reason the lattice is one level finer than max_level.

    With the lattice at max_level a triangle's edges are one unit long and
    `midpoint_ij`'s floor division collapses each "midpoint" onto one of that
    edge's own endpoints. One level finer, every edge vector is even at every
    level up to and including max_level, so the midpoints are genuine lattice
    points -- and they are exactly the points with an odd coordinate, which is
    what guarantees they can never collide with a cached vertex.
    """
    _, _, _, _, root_class = child_matrix_tables()
    max_level = 3
    lat = make_lattice(4.0, 0.0, 0.0, 2, max_level + 1)
    ij, cls = initial_triangles(2, lat.level, root_class)
    # Descend to max_level by taking child C_4 (the middle child) each time.
    for _ in range(max_level):
        ij = midpoint_ij(ij)
    assert bool(mesh_backend.all(ij % 2 == 0)), "max_level vertices are even"

    mid = midpoint_ij(ij)
    # Exact: the floor division threw nothing away.
    assert bool(
        mesh_backend.all((ij[:, [1, 2, 0]] + ij[:, [2, 0, 1]]) % 2 == 0)
    ), "edge endpoint sums must be even for the midpoint to be exact"
    # Every midpoint has an odd coordinate, so it is not a vertex of any level.
    assert bool(mesh_backend.all(mesh_backend.any(mid % 2 == 1, dim=-1)))


def test_check_lattice_keys_rejects_a_lattice_too_fine_to_key():
    with pytest.raises(ValueError, match="lattice too fine"):
        check_lattice_keys(2**20, 21, "Raise min_img_sep.")


def test_a_fresh_lattice_places_points_exactly_as_before():
    lat = make_lattice(4.0, 0.3, -0.1, 3, 4)
    ij = _grid(lat.n)
    assert lat.origin == 0
    want = to_np(lat.lo) + ij.astype(np.float64) * lat.scale
    assert np.array_equal(to_np(lattice_xy(lat, i64(ij))), want)


def test_extend_lattice_keeps_every_old_position_bit_for_bit():
    """A non-dyadic center on purpose: nothing may be recomputed from a new fov."""
    lat = make_lattice(4.0, 0.3, -0.1, 3, 4)
    ext = extend_lattice(lat, 2)
    pad = 2 << lat.level
    ij = _grid(lat.n)
    assert np.array_equal(
        to_np(lattice_xy(ext, i64(ij + pad))),
        to_np(lattice_xy(lat, i64(ij))),
    )


def test_check_lattice_keys_rejects_exactly_the_lattices_int64_cannot_key():
    """``45 * 2**26 + 1`` points per axis still key in int64; ``46 * 2**26 + 1`` do not."""
    check_lattice_keys(45, 25, "unused")
    with pytest.raises(ValueError, match="lattice too fine.*Rebuild coarser"):
        check_lattice_keys(46, 25, "Rebuild coarser.")


def test_ring_triangles_are_the_level0_triangles_outside_the_central_block():
    level = 3
    root_class = child_matrix_tables()[4]
    every_ij, every_cls = initial_triangles(6, level, root_class)
    ring_ij, ring_cls = ring_triangles(6, 1, level, root_class)

    def cells(ij):
        return to_np(ij)[:, 0, :] // (1 << level)

    def inner(ij):
        c = cells(ij)
        return ((c >= 1) & (c < 5)).all(axis=1)

    def as_set(ij, cls):
        return {(tuple(t.reshape(-1)), c) for t, c in zip(to_np(ij), to_np(cls))}

    assert ring_ij.shape[0] == 2 * (6 * 6 - 4 * 4)
    assert not inner(ring_ij).any()
    ring = as_set(ring_ij, ring_cls)
    every = as_set(every_ij, every_cls)
    assert ring <= every
    assert len(every - ring) == int(inner(every_ij).sum())


def test_extend_lattice_grows_the_extent_and_keeps_key_order():
    lat = make_lattice(4.0, 0.3, -0.1, 3, 4)
    ext = extend_lattice(lat, 2)
    pad = 2 << lat.level
    assert (ext.level, ext.scale) == (lat.level, lat.scale)
    assert ext.n == lat.n + 2 * pad
    assert ext.origin == pad
    assert np.array_equal(to_np(ext.lo), to_np(lat.lo))
    ij = _grid(lat.n)
    old = to_np(lattice_key(lat, i64(ij)))
    got = to_np(lattice_key(ext, i64(ij + pad)))
    assert np.array_equal(np.argsort(old), np.argsort(got))


def test_an_extended_dyadic_lattice_is_the_fresh_lattice_of_the_larger_fov():
    """The premise of every equivalence test below.

    With a dyadic fov, center and cell size, a fresh lattice over the larger
    fov places every point exactly where the extended one does.
    """
    lat = make_lattice(4.0, 0.5, -0.25, 8, 3)
    ext = extend_lattice(lat, 2)
    fresh = make_lattice(6.0, 0.5, -0.25, 12, 3)
    assert (ext.n, ext.scale) == (fresh.n, fresh.scale)
    ij = i64(_grid(ext.n))
    assert np.array_equal(to_np(lattice_xy(ext, ij)), to_np(lattice_xy(fresh, ij)))


def test_extend_lattice_chains_and_accepts_zero():
    lat = make_lattice(4.0, 0.0, 0.0, 2, 3)
    assert _same_lattice(extend_lattice(lat, 0), lat)
    assert _same_lattice(
        extend_lattice(extend_lattice(lat, 1), 2), extend_lattice(lat, 3)
    )


def test_min_angle_of_an_equilateral_triangle():
    tri = mesh_backend.as_array(
        np.array([[[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]]]),
        dtype=mesh_backend.float64,
    )
    assert to_np(min_angle(tri))[0] == pytest.approx(np.pi / 3)


def test_min_angle_of_a_degenerate_triangle_is_zero_not_nan():
    tri = mesh_backend.as_array(
        np.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]), dtype=mesh_backend.float64
    )
    assert to_np(min_angle(tri))[0] == 0.0


def test_the_table_constants_are_child_matrix_tables():
    _, _, compose, pinv0, root_class = child_matrix_tables()
    assert np.array_equal(to_np(COMPOSE), to_np(compose))
    assert np.array_equal(to_np(PINV0), to_np(pinv0))
    assert np.array_equal(to_np(ROOT_CLASS), to_np(root_class))


def test_area2_is_twice_the_signed_area():
    tri = f64([[[0, 0], [2, 0], [0, 1]], [[0, 0], [0, 1], [2, 0]]])
    assert to_np(area2(tri)).tolist() == [2.0, -2.0]


def test_csr_offsets_start_at_zero_and_accumulate():
    assert to_np(csr_offsets(i64([2, 0, 3]))).tolist() == [0, 2, 2, 5]
    assert to_np(csr_offsets(i64([]))).tolist() == [0]


def test_expand_gives_each_entry_its_block_and_its_place_in_it():
    for total in (None, 6):
        block, within = expand(i64([2, 0, 3, 1]), total)
        assert to_np(block).tolist() == [0, 0, 2, 2, 2, 3]
        assert to_np(within).tolist() == [0, 1, 0, 1, 2, 0]
    block, within = expand(i64([]))
    assert tuple(block.shape) == tuple(within.shape) == (0,)


def test_component_labels_name_each_component_by_its_smallest_node():
    """A chain 2-1-0, a lone node 3 and a pair 4-5; batched with a graph of no edges."""
    adjacent = np.eye(6, dtype=bool)
    for a, b in ((2, 1), (1, 0), (5, 4)):
        adjacent[a, b] = adjacent[b, a] = True
    got = component_labels(mesh_backend.as_array(adjacent))
    assert to_np(got).tolist() == [0, 0, 0, 3, 4, 4]
    both = component_labels(
        mesh_backend.as_array(np.stack([adjacent, np.eye(6, dtype=bool)]))
    )
    assert to_np(both).tolist() == [[0, 0, 0, 3, 4, 4], list(range(6))]


def test_map_chunks_runs_fn_on_chunks_of_at_most_batch_size_rows_and_concatenates():
    points = f64(np.arange(14.0).reshape(7, 2))
    seen = []

    def fn(chunk):
        seen.append(chunk.shape[0])
        return 2.0 * chunk[:, 0], chunk[:, 1:]

    for batch_size, sizes in ((None, [7]), (3, [3, 3, 1]), (10, [7])):
        seen.clear()
        doubled, rest = map_chunks(fn, points, batch_size)
        assert seen == sizes
        assert to_np(doubled).tolist() == np.arange(0.0, 28.0, 4.0).tolist()
        assert to_np(rest).tolist() == np.arange(1.0, 14.0, 2.0)[:, None].tolist()
    assert tuple(map_chunks(lambda c: c[:, 0], points[:0], 3).shape) == (0,)


def test_sorted_find_gives_each_value_s_position_and_whether_it_is_there():
    pos, found = sorted_find(i64([1, 4, 9]), i64([0, 1, 5, 9, 10]))
    assert to_np(pos).tolist() == [0, 0, 2, 2, 2]
    assert to_np(found).tolist() == [False, True, False, True, False]
    member = is_member(i64([1, 4, 9]), i64([0, 1, 5, 9, 10]))
    assert to_np(member).tolist() == to_np(found).tolist()
    pos, found = sorted_find(i64([]), i64([3]))
    assert to_np(pos).tolist() == [0] and to_np(found).tolist() == [False]


def test_to_device_maps_every_array_field_of_nested_named_tuples():
    lat = make_lattice(4.0, 0.0, 0.0, 2, 3)
    moved = to_device(lat, mesh_backend.device(lat.lo))
    assert moved.n == lat.n and moved.scale == lat.scale
    assert np.array_equal(to_np(moved.lo), to_np(lat.lo))


def test_the_derived_lattice_sizes_match_the_build_arguments():
    lat = make_lattice(4.0, 0.25, -0.5, 8, 5)
    assert lattice_init_res(lat) == 8
    assert lattice_h0(lat) == 0.5
    assert lattice_fov(lat) == 4.0
    grown = extend_lattice(lat, 2)
    assert lattice_init_res(grown) == 12 and lattice_fov(grown) == 6.0
    assert lattice_h0(grown) == 0.5


def test_keys_round_trip_without_a_stride_field():
    lat = make_lattice(4.0, 0.0, 0.0, 3, 2)
    ij = i64([[0, 0], [lat.n, lat.n], [5, 7]])
    assert np.array_equal(
        to_np(lattice_ij_from_key(lat, lattice_key(lat, ij))), to_np(ij)
    )
    assert to_np(lattice_key(lat, ij)).tolist()[2] == 5 * (lat.n + 1) + 7


def test_depth_floor_reads_the_level0_hypotenuse():
    assert depth_floor(1.0, 2.0) == 0
    assert depth_floor(1.0, math.sqrt(2.0) / 4) == 2


def test_warn_depth_limited_names_the_request_and_the_needed_depth():
    with pytest.warns(
        UserWarning,
        match=r"Lens mesh is depth-limited.*min_img_sep=0.1.*max_depth >= 4",
    ):
        warn_depth_limited("Lens mesh", "min_img_sep=0.1 arcsec", 0.5, 0.05, 2)


def test_warn_depth_limited_is_silent_when_the_floor_is_reached(recwarn):
    warn_depth_limited("Lens mesh", "x", 0.5, 0.05, 4)
    assert not recwarn.list


def test_build_index_of_an_empty_leaf_set_has_no_levels_and_finds_nothing():
    vs, leaves = f64(np.zeros((3, 2))), i64(np.zeros((0, 3)))
    idx = build_index(vs, leaves, i64(np.zeros(0)))
    assert tuple(idx.levels.shape) == (0,) and tuple(idx.keys.shape) == (0,)
    assert to_np(idx.offsets).tolist() == [0] and tuple(idx.leaves.shape) == (0,)
    start, count = index_cells(idx, f64([[0.5, 0.5]]))
    assert tuple(start.shape) == tuple(count.shape) == (1, 0)
    hits = index_hits(idx, vs, leaves, f64([[0.5, 0.5]]))
    assert [tuple(a.shape) for a in hits] == [(0,), (0,), (0, 3)]


@pytest.mark.skipif(
    backend.backend != "torch", reason="needs torch's meta default device"
)
@pytest.mark.parametrize("n_rows", [0, 60], ids=["empty", "filled"])
def test_build_index_allocates_on_its_inputs_device_not_the_default_one(n_rows):
    """A mesh on another device is indexed at query time (``band_cover``).

    With the default device set to ``meta``, anything ``build_index``
    allocates without its inputs' device lands there and fails to mix.
    """
    import torch

    rng = np.random.default_rng(13)
    vs = f64(rng.normal(size=(50, 2)))
    leaves = i64(rng.integers(0, 50, (60, 3)))
    rows = i64(np.arange(n_rows))
    with torch.device("meta"):
        idx = build_index(vs, leaves, rows)
    for name in ("lo", "hi", "fine", "levels", "keys", "offsets", "leaves"):
        assert getattr(idx, name).device == vs.device, name


def test_build_index_leaves_are_ascending_within_every_cell():
    rng = np.random.default_rng(13)
    vs = rng.normal(size=(50, 2))
    leaves = rng.integers(0, 50, (60, 3))
    idx = build_index(f64(vs), i64(leaves), i64(np.arange(60)))
    offsets = to_np(idx.offsets)
    cell_leaves = to_np(idx.leaves)
    assert offsets[0] == 0 and offsets[-1] == cell_leaves.size
    exercised = 0
    for a, b in zip(offsets[:-1], offsets[1:]):
        block = cell_leaves[a:b]
        if block.size > 1:
            exercised += 1
            assert (np.diff(block) > 0).all()
    assert exercised > 0, "fixture must exercise a cell containing multiple leaves"


def test_as_points_flattens_any_shape_into_float64_rows():
    got = as_points(2.0, 3.0, None)
    assert got.dtype == mesh_backend.float64 and to_np(got).tolist() == [[2.0, 3.0]]
    got = as_points(f64([[0.0, 1.0]]), f64([[2.0, 3.0]]), None)
    assert to_np(got).tolist() == [[0.0, 2.0], [1.0, 3.0]]
    assert as_points(f64([]), f64([]), None).shape == (0, 2)


def test_edge_nearest_projects_onto_the_nearest_edge():
    tri = f64([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]] * 3)
    beta = f64([[0.25, -0.5], [0.75, 0.75], [-0.5, 0.25]])
    dist, bary = edge_nearest(tri, beta)
    assert np.allclose(to_np(dist), [0.5, np.sqrt(2) / 4, 0.5])
    assert np.allclose(
        to_np(bary), [[0.75, 0.25, 0.0], [0.0, 0.5, 0.5], [0.75, 0.0, 0.25]]
    )


def test_edge_nearest_gives_the_vertex_beyond_a_corner_and_the_end_of_a_zero_length_edge():
    tri = f64(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
        ]
    )
    dist, bary = edge_nearest(tri, f64([[2.0, -1.0], [0.0, -1.0]]))
    assert np.allclose(to_np(dist), [np.sqrt(2), 1.0])
    b = to_np(bary)
    assert np.allclose(b[0], [0.0, 1.0, 0.0])
    assert np.allclose(b[1] @ to_np(tri)[1], [0.0, 0.0])


def test_edge_nearest_is_the_nearest_boundary_point_in_the_simplex():
    rng = np.random.default_rng(3)
    tri = rng.normal(size=(200, 3, 2))
    beta = 3.0 * rng.normal(size=(200, 2))
    dist, bary = edge_nearest(f64(tri), f64(beta))
    d, b = to_np(dist), to_np(bary)
    assert (b >= 0).all() and np.allclose(b.sum(axis=1), 1.0)
    nearest = np.einsum("kj,kjd->kd", b, tri)
    assert np.allclose(np.linalg.norm(nearest - beta, axis=1), d)
    s = np.linspace(0.0, 1.0, 401)[None, :, None]
    sampled = np.concatenate(
        [
            tri[:, i, None] + s * (tri[:, (i + 1) % 3, None] - tri[:, i, None])
            for i in range(3)
        ],
        axis=1,
    )
    brute = np.linalg.norm(sampled - beta[:, None], axis=-1).min(axis=1)
    assert (d <= brute + 1e-12).all()


SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def test_winding_number_of_a_square_is_one_inside_zero_outside_and_minus_one_reversed():
    pts = f64([[0.5, 0.5], [1.5, 0.5], [0.5, -0.5], [-0.5, 0.5]])
    assert to_np(winding_number(f64(SQUARE), pts)).tolist() == [1, 0, 0, 0]
    assert to_np(winding_number(f64(SQUARE[::-1]), pts)).tolist() == [-1, 0, 0, 0]


def test_winding_number_counts_a_ray_through_a_vertex_once():
    diamond = f64([[1.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
    pts = f64([[1.0, 1.0], [-1.0, 1.0], [1.0, 0.5]])
    assert to_np(winding_number(diamond, pts)).tolist() == [1, 0, 1]


def test_winding_number_of_a_figure_eight_is_plus_and_minus_one_in_its_lobes():
    eight = f64(
        [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]]
    )
    pts = f64([[0.6, 0.1], [-0.6, 0.1], [2.0, 0.1]])
    assert to_np(winding_number(eight, pts)).tolist() == [1, -1, 0]


def test_winding_number_of_a_nan_point_or_past_a_nan_vertex_is_zero():
    assert to_np(winding_number(f64(SQUARE), f64([[np.nan, 0.5]]))).tolist() == [0]
    broken = f64([[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [0.0, 1.0]])
    assert to_np(winding_number(broken, f64([[0.5, 0.5]]))).tolist() == [0]


def test_build_index_with_a_zero_grow_is_the_plain_index():
    rng = np.random.default_rng(11)
    vs = f64(rng.uniform(-1, 1, (90, 2)))
    leaves = i64(np.arange(90).reshape(30, 3))
    rows = i64(np.arange(30))
    assert_same(
        build_index(vs, leaves, rows, f64(np.zeros(30))), build_index(vs, leaves, rows)
    )


def test_build_index_registers_a_triangle_in_every_cell_its_grown_box_covers():
    rng = np.random.default_rng(12)
    corner = rng.uniform(-1, 1, (200, 2))
    tri = corner[:, None, :] + 0.01 * np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    grow = np.zeros(200)
    grow[0] = 0.3
    idx = build_index(
        f64(tri.reshape(-1, 2)),
        i64(np.arange(600).reshape(200, 3)),
        i64(np.arange(200)),
        f64(grow),
    )
    level, i0, i1 = index_cell_ranges(idx, tri[:1], grow[:1])
    assert (i1[0] - i0[0] >= 2).all(), "the grown box must span several cells"
    for ix in range(i0[0, 0], i1[0, 0] + 1):
        for iy in range(i0[0, 1], i1[0, 1] + 1):
            assert 0 in index_cell(idx, level[0], ix, iy)


def test_index_hits_with_grow_returns_a_point_within_grow_of_a_triangle_and_not_beyond():
    vs = f64([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0], [5.0, 6.0]])
    leaves = i64([[0, 1, 2], [3, 4, 5]])
    grow = f64([0.2, 0.0])
    idx = build_index(vs, leaves, i64([0, 1]), grow)
    beta = f64([[0.5, -0.1], [0.5, -0.3], [0.25, 0.25], [5.5, 4.9]])
    qidx, tri, w = index_hits(idx, vs, leaves, beta, grow)
    assert list(zip(to_np(qidx).tolist(), to_np(tri).tolist())) == [(0, 0), (2, 0)]
    expected = triangle_weights(vs[leaves[tri]], beta[qidx])
    assert np.array_equal(to_np(w), to_np(expected))
    plain_q, plain_t, _ = index_hits(idx, vs, leaves, beta)
    assert list(zip(to_np(plain_q).tolist(), to_np(plain_t).tolist())) == [(2, 0)]


def test_index_hits_with_an_all_zero_grow_matches_a_plain_index():
    rng = np.random.default_rng(11)
    vs = f64(rng.uniform(-1, 1, (90, 2)))
    leaves = i64(np.arange(90).reshape(30, 3))
    rows = i64(np.arange(30))
    grow = f64(np.zeros(30))
    idx = build_index(vs, leaves, rows, grow)
    beta = f64(rng.uniform(-1, 1, (200, 2)))
    qidx_grow, tri_grow, w_grow = index_hits(idx, vs, leaves, beta, grow)
    qidx_plain, tri_plain, w_plain = index_hits(idx, vs, leaves, beta)
    assert np.array_equal(to_np(qidx_grow), to_np(qidx_plain))
    assert np.array_equal(to_np(tri_grow), to_np(tri_plain))
    assert np.array_equal(to_np(w_grow), to_np(w_plain))


def _scattered_triangles(seed, n=400):
    """``n`` triangles with sides from 1e-9 to 1e2 arcsec, the first five moved out to about 1e6."""
    rng = np.random.default_rng(seed)
    size = 10.0 ** rng.uniform(-9, 2, n)
    corner = rng.uniform(-1, 1, (n, 2))
    corner[:5] *= 1e6
    return corner[:, None, :] + size[:, None, None] * rng.uniform(-1, 1, (n, 3, 2))


def _as_mesh(tri):
    """``tri`` ``(n, 3, 2)`` as vertices, triangles and all their rows."""
    n = tri.shape[0]
    return (
        f64(tri.reshape(-1, 2)),
        i64(np.arange(3 * n).reshape(n, 3)),
        i64(np.arange(n)),
    )


def test_index_hits_equal_brute_force_across_eleven_orders_of_magnitude():
    tri = _scattered_triangles(9)
    vs, leaves, rows = _as_mesh(tri)
    rng = np.random.default_rng(10)
    n = tri.shape[0]
    grow = f64(np.where(rng.uniform(size=n) < 0.2, 10.0 ** rng.uniform(-9, 0, n), 0.0))
    inner = tri[:, 0] + 0.3 * (tri[:, 1] - tri[:, 0]) + 0.3 * (tri[:, 2] - tri[:, 0])
    beta = f64(np.concatenate([inner, tri[:, 1], rng.uniform(-1, 1, (200, 2))]))
    idx = build_index(vs, leaves, rows, grow)
    assert_hits_equal(
        index_hits(idx, vs, leaves, beta, grow),
        brute_hits(vs, leaves, rows, beta, grow),
    )


def test_index_hits_skip_nan_infinite_and_far_points_and_take_empty_input():
    rng = np.random.default_rng(14)
    tri = rng.uniform(-1, 1, (40, 1, 2)) + 0.2 * rng.uniform(-1, 1, (40, 3, 2))
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows)
    beta = f64(
        [[np.nan, 0.0], [np.inf, 0.0], [0.0, -np.inf], [1e9, 1e9], tri[3].mean(axis=0)]
    )
    qidx, hit, w = index_hits(idx, vs, leaves, beta)
    assert set(to_np(qidx).tolist()) == {4}
    want_q, want_t, want_w = brute_hits(vs, leaves, rows, beta[4:])
    assert want_t.size > 0
    assert_hits_equal((qidx, hit, w), (want_q + 4, want_t, want_w))
    empty = index_hits(idx, vs, leaves, f64(np.zeros((0, 2))))
    assert [tuple(a.shape) for a in empty] == [(0,), (0,), (0, 3)]


def test_build_index_registers_each_row_at_one_level_in_exactly_the_cells_of_its_box():
    tri = _scattered_triangles(21)
    n = tri.shape[0]
    rng = np.random.default_rng(22)
    grow = np.where(rng.uniform(size=n) < 0.2, 10.0 ** rng.uniform(-9, 0, n), 0.0)
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows, f64(grow))
    level, i0, i1 = index_cell_ranges(idx, tri, grow)
    assert np.unique(level).size > 5, "fixture must spread over many levels"
    assert ((i1 - i0 + 1) <= CELLS_PER_BOX + 2).all()
    keys, offsets = to_np(idx.keys), to_np(idx.offsets)
    key_of_entry = np.repeat(keys, np.diff(offsets))
    row_of_entry = to_np(idx.leaves)
    for r in range(n):
        ix, iy = np.meshgrid(
            np.arange(i0[r, 0], i1[r, 0] + 1),
            np.arange(i0[r, 1], i1[r, 1] + 1),
            indexing="ij",
        )
        want = (level[r] << (2 * KEY_BITS)) | (ix.ravel() << KEY_BITS) | iy.ravel()
        got = key_of_entry[row_of_entry == r]
        assert np.array_equal(np.sort(got), np.sort(want)), r


def test_build_index_keys_ascend_and_give_its_levels_and_fine_side():
    vs, leaves, rows = _as_mesh(_scattered_triangles(23))
    idx = build_index(vs, leaves, rows)
    keys, offsets = to_np(idx.keys), to_np(idx.offsets)
    assert (np.diff(keys) > 0).all()
    assert offsets[0] == 0 and offsets[-1] == to_np(idx.leaves).size
    assert (np.diff(offsets) > 0).all()
    assert to_np(idx.levels).tolist() == np.unique(keys >> (2 * KEY_BITS)).tolist()
    extent = (to_np(idx.hi) - to_np(idx.lo)).max()
    assert float(to_np(idx.fine)) == extent / 2**KEY_BITS


def test_index_cells_gives_the_cell_each_point_falls_in_at_every_level():
    tri = _scattered_triangles(27)
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows)
    rng = np.random.default_rng(28)
    beta = np.concatenate(
        [tri[:, 0], rng.uniform(-1, 1, (100, 2)), [[np.nan, 0.0], [1e9, 1e9]]]
    )
    start, count = index_cells(idx, f64(beta))
    start, count = to_np(start), to_np(count)
    levels, entries = to_np(idx.levels), to_np(idx.leaves)
    assert start.shape == count.shape == (beta.shape[0], levels.size)
    lo, hi = to_np(idx.lo), to_np(idx.hi)
    for b, point in enumerate(beta):
        inside = bool(np.all((point >= lo) & (point <= hi)))
        for j, level in enumerate(levels):
            want = np.zeros(0, dtype=np.int64)
            if inside:
                ix, iy = index_point_cells(idx, [level], point[None])[0]
                want = index_cell(idx, level, ix, iy)
            got = entries[start[b, j] : start[b, j] + count[b, j]]
            assert np.array_equal(got, want), (b, level)


def test_index_hits_find_points_on_the_cell_lines_of_every_level():
    """Points snapped onto each level's cell lines, and ``hi`` itself, are found as brute force finds them."""
    tri = _scattered_triangles(25)[5:]
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows)
    lo, fine = to_np(idx.lo), float(to_np(idx.fine))
    points = [to_np(idx.hi)[None]]
    for level in to_np(idx.levels):
        side = fine * 2.0**level
        snapped = lo + np.floor((tri[::20, 0] - lo) / side) * side
        points += [snapped, np.stack([snapped[:, 0], tri[::20, 0, 1]], axis=-1)]
    beta = f64(np.concatenate(points))
    assert_hits_equal(
        index_hits(idx, vs, leaves, beta), brute_hits(vs, leaves, rows, beta)
    )


# ---------------------------------------------------------------------------
# contains_half_open: the fill rule
# ---------------------------------------------------------------------------

# Two counter-clockwise triangles sharing the diagonal (0,0)-(1,1) of the unit square.
SQUARE_HALVES = [
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
    [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
]


def _half_open(triangles, point):
    tri = f64(triangles)
    beta = f64([point] * len(triangles))
    return to_np(contains_half_open(tri, triangle_weights(tri, beta))).tolist()


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("point", [[0.5, 0.5], [0.25, 0.25], [0.875, 0.875]])
def test_a_point_on_a_shared_edge_is_in_exactly_one_triangle(point, flip):
    triangles = [t[::-1] for t in SQUARE_HALVES] if flip else SQUARE_HALVES
    assert sum(_half_open(triangles, point)) == 1


def test_a_point_strictly_inside_is_in_its_triangle_alone():
    assert _half_open(SQUARE_HALVES, [0.6, 0.2]) == [True, False]
    assert _half_open(SQUARE_HALVES, [0.2, 0.6]) == [False, True]
    assert _half_open(SQUARE_HALVES, [1.5, 0.5]) == [False, False]


@pytest.mark.parametrize("flip", [False, True])
def test_the_center_of_a_fan_is_in_exactly_one_of_its_triangles(flip):
    angle = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    ring = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    fan = [[[0.0, 0.0], ring[k], ring[(k + 1) % 6]] for k in range(6)]
    if flip:
        fan = [t[::-1] for t in fan]
    assert sum(_half_open(fan, [0.0, 0.0])) == 1


def test_a_triangle_of_zero_area_contains_nothing():
    tri = f64([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
    w = triangle_weights(tri, f64([[1.0, 1.0]]))
    assert to_np(contains(w)).tolist() == [True]
    assert to_np(contains_half_open(tri, w)).tolist() == [False]


def test_away_from_edges_the_fill_rule_is_plain_containment():
    rng = np.random.default_rng(5)
    tri = f64(rng.uniform(-1.0, 1.0, (500, 3, 2)))
    w = triangle_weights(tri, f64(rng.uniform(-1.0, 1.0, (500, 2))))
    np.testing.assert_array_equal(to_np(contains_half_open(tri, w)), to_np(contains(w)))


# ---------------------------------------------------------------------------
# ring_next and union_area
# ---------------------------------------------------------------------------


def _loops(*polylines):
    """Closed polylines as mesh-backend ``(points, offsets)``."""
    points = np.concatenate([np.asarray(p, dtype=np.float64) for p in polylines])
    offsets = np.concatenate(([0], np.cumsum([len(p) for p in polylines])))
    return f64(points), i64(offsets)


def _densify(polygon, n):
    """``polygon`` with each edge cut into ``n`` equal pieces."""
    p = np.asarray(polygon, dtype=np.float64)
    t = np.arange(n)[None, :, None] / n
    return (p[:, None] + t * (np.roll(p, -1, 0) - p)[:, None]).reshape(-1, 2)


SQUARE_2X2 = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]


def test_ring_next_wraps_each_polyline_to_its_first_point():
    curve, nxt = ring_next(i64([0, 3, 5]), 5)
    assert to_np(curve).tolist() == [0, 0, 0, 1, 1]
    assert to_np(nxt).tolist() == [1, 2, 0, 4, 3]


@pytest.mark.parametrize("reverse", [False, True])
def test_the_union_of_one_square_is_its_area_either_way_round(reverse):
    assert union_area(*_loops(SQUARE_2X2[::-1] if reverse else SQUARE_2X2)) == 4.0


def test_two_overlapping_squares_count_their_overlap_once():
    shifted = (np.asarray(SQUARE_2X2) + 1.0).tolist()
    assert union_area(*_loops(SQUARE_2X2, shifted)) == 7.0


def test_a_loop_inside_another_adds_nothing_whichever_its_orientation():
    inner = [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]]
    assert union_area(*_loops(SQUARE_2X2, inner)) == 4.0
    assert union_area(*_loops(SQUARE_2X2, inner[::-1])) == 4.0


def test_a_part_a_loop_winds_about_twice_counts_once():
    assert union_area(*_loops(SQUARE_2X2 + SQUARE_2X2)) == 4.0


def test_both_lobes_of_a_figure_eight_count():
    bowtie = _densify([[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]], 200)
    assert union_area(*_loops(bowtie)) == pytest.approx(2.0, rel=1e-12)


def test_vertical_and_zero_length_segments_add_nothing():
    unit = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    assert union_area(*_loops(unit)) == 1.0


def test_a_regular_polygon_has_its_exact_area():
    n = 2000
    angle = 2.0 * np.pi * np.arange(n) / n
    ring = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    assert union_area(*_loops(ring)) == pytest.approx(
        0.5 * n * math.sin(2.0 * math.pi / n), rel=1e-12
    )


def test_no_polyline_has_no_area():
    assert union_area(f64(np.zeros((0, 2))), i64([0])) == 0.0


@pytest.mark.parametrize("seed", range(4))
def test_self_crossing_polygons_cover_what_a_winding_count_covers(seed):
    rng = np.random.default_rng(seed)
    first, second = rng.uniform(0.0, 3.0, (12, 2)), rng.uniform(1.0, 4.0, (9, 2))
    area = union_area(*_loops(_densify(first, 100), _densify(second, 100)))
    n = 600
    axis = (np.arange(n) + 0.5) * 4.0 / n
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    grid = f64(np.stack((gx.ravel(), gy.ravel()), axis=-1))
    covered = (to_np(winding_number(f64(first), grid)) != 0) | (
        to_np(winding_number(f64(second), grid)) != 0
    )
    assert area == pytest.approx(covered.mean() * 16.0, abs=0.02)
