"""Triangle maths, the red-split tables and the array helpers."""

import math
import numpy as np
import pytest

from caustics.lenses.func.adaptive.mesh_backend import mesh_backend
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
from caustics.lenses.func.adaptive.lattice import make_lattice

from adaptive_maps import f64, i64, to_np

RNG = np.random.default_rng(20260918)


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


def test_sigma_min_of_zero_matrix_is_exactly_zero():
    A = np.zeros((4, 2, 2))
    got = to_np(sigma_min_2x2(f64(A)))
    assert (got == 0.0).all()
    assert not np.isnan(got).any()


def test_sigma_min_propagates_nan():
    A = np.zeros((1, 2, 2))
    A[0, 0, 0] = np.nan
    assert np.isnan(to_np(sigma_min_2x2(f64(A)))).all()


def test_sigma_min_matches_svd():
    A = RNG.normal(size=(2000, 2, 2))
    expected = np.linalg.svd(A, compute_uv=False)[:, -1]
    assert np.allclose(to_np(sigma_min_2x2(f64(A))), expected, rtol=1e-9, atol=1e-12)


def test_sigma_min_near_and_exactly_singular():
    a = RNG.normal(size=(500, 2))
    perp = np.stack([-a[:, 1], a[:, 0]], axis=1)
    # Rows offset perpendicularly, so det == eps * |a|^2 is genuinely small.
    # Scaling one row instead would give a rank-1 matrix with det exactly zero,
    # which tests the degenerate branch, not the near-degenerate one.
    for eps in (1e-4, 1e-7):
        A = np.stack([a, a + eps * perp], axis=1)
        got = to_np(sigma_min_2x2(f64(A)))
        expected = np.linalg.svd(A, compute_uv=False)[:, -1]
        assert np.isfinite(got).all()
        assert np.allclose(got, expected, rtol=1e-5, atol=0.0)
    exact = np.stack([a, 2.0 * a], axis=1)  # rank 1: det is exactly 0 in IEEE
    assert (to_np(sigma_min_2x2(f64(exact))) == 0.0).all()


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
    got = to_np(sigma_min_2x2(f64(A.astype(np.float64))))
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
    R0 = to_np(shape_matrix(f64(ROOT_SHAPES[0])))
    R1 = to_np(shape_matrix(f64(ROOT_SHAPES[1])))
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
                direct = to_np(mesh_backend.linalg.inv(shape_matrix(f64(kids[k]))))
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
        Q = to_np(shape_matrix(f64(q)))
        table = Q @ ((2.0 ** (level + 1) / h0) * PINV0[COMPOSE[cls, k]])
        assert np.allclose(table, lens_map, rtol=1e-10, atol=1e-12)


def test_weights_sum_to_twice_the_signed_area():
    tri = RNG.normal(size=(500, 3, 2))
    beta = RNG.normal(size=(500, 2))
    w = to_np(triangle_weights(f64(tri), f64(beta)))
    P = to_np(shape_matrix(f64(tri)))
    d = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    assert np.allclose(w.sum(axis=1), d, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("flip", [False, True])
def test_containment_agrees_with_barycentric_truth_on_both_parities(flip):
    tri = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    if flip:
        tri = tri[:, ::-1, :]
    pts = RNG.uniform(-0.5, 1.5, size=(4000, 2))
    tiled = np.repeat(tri, len(pts), axis=0)
    hit = to_np(contains(triangle_weights(f64(tiled), f64(pts))))
    truth = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] + pts[:, 1] <= 1)
    assert (hit == truth).all()


def test_shared_edge_weights_are_exactly_negated():
    """Two leaves meeting on an edge must not both reject a point on that edge."""
    verts = np.array([[0.0, 0.0], [1.0, 0.3], [0.4, 1.0], [1.3, 1.2]])
    left = verts[[0, 1, 2]][None]
    right = verts[[1, 3, 2]][None]  # shares the edge (1, 2), opposite traversal
    beta = np.array([[0.62, 0.71]])
    wl = to_np(triangle_weights(f64(left), f64(beta)))
    wr = to_np(triangle_weights(f64(right), f64(beta)))
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
    a0, a1, b0, b1 = (f64([c[k][e] for c in cases]) for k in (0, 1) for e in (0, 1))
    assert to_np(segments_cross(a0, a1, b0, b1)).tolist() == [want] * 4


def test_segments_cross_broadcasts_to_a_table_of_every_pair():
    a = np.array([[[0, 0], [2, 2]], [[0, 0], [1, 0]]], dtype=np.float64)
    b = np.array(
        [[[0, 2], [2, 0]], [[1, 0], [1, 1]], [[5, 5], [6, 6]]], dtype=np.float64
    )
    got = segments_cross(
        f64(a[:, None, 0]),
        f64(a[:, None, 1]),
        f64(b[None, :, 0]),
        f64(b[None, :, 1]),
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
    bary = to_np(sanitize_bary(f64(w), f64(d)))
    assert np.isfinite(bary).all()
    assert (bary >= 0).all() and (bary <= 1).all()
    assert np.allclose(bary.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_sanitize_bary_falls_back_to_the_centroid_on_total_degeneracy():
    w = np.zeros((4, 3))
    d = np.zeros(4)
    bary = to_np(sanitize_bary(f64(w), f64(d)))
    assert np.array_equal(bary, np.full((4, 3), 1.0 / 3.0))


def test_sanitize_bary_selects_per_row_between_normalized_and_centroid():
    """Degenerate and ordinary rows in one call must be resolved independently.

    Both other sanitize_bary tests are all-or-nothing -- every row ordinary, or
    every row degenerate -- so neither would catch an implementation that decided
    the fallback once for the whole batch instead of per row.
    """
    w = f64([[3.0, 1.5, 1.5], [0.0, 0.0, 0.0], [2.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    d = f64([6.0, 0.0, 4.0, 0.0])
    bary = to_np(sanitize_bary(w, d))
    assert np.allclose(bary[[0, 2]], [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    assert np.allclose(bary[[1, 3]], 1.0 / 3.0)
    assert np.allclose(bary.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_sanitize_bary_recovers_ordinary_coordinates():
    tri = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]])
    beta = np.array([[0.5, 0.75]])
    w = triangle_weights(f64(tri), f64(beta))
    d = f64([2.0 * 3.0])
    bary = to_np(sanitize_bary(w, d))
    assert np.allclose(bary @ tri[0], beta, rtol=1e-12, atol=1e-14)


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
