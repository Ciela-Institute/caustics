"""The lens mesh's refinement criterion and its ``LEAF_*`` flags."""

import importlib

import numpy as np
import pytest

from caustics.lenses.func.adaptive.mesh_backend import mesh_backend
from caustics.lenses.func.adaptive.criterion import (
    _status,
    child_maps,
    child_shape_matrices,
    children_sigma_min,
    midpoint_deviation,
    parity_from_children,
    LEAF_APPROX_PARITY_UNRESOLVED,
    LEAF_CONVERGED,
    LEAF_CONVERGENCE_FAILED,
    LEAF_JACOBIAN_PARITY_UNRESOLVED,
    LEAF_RAYTRACE_NONFINITE,
    affine_error,
    fitted_jacobian,
    lens_status,
)
from caustics.lenses.func.adaptive.geometry import (
    CHILD_VERTEX_INDICES,
    COMPOSE,
    PINV0,
    ROOT_CLASS,
    ROOT_SHAPES,
    shape_matrix,
    sigma_min_2x2,
)
from caustics.lenses.func.adaptive.lattice import (
    initial_triangles,
    lattice_xy,
    make_lattice,
    midpoint_ij,
)

from adaptive_maps import f64, i64, to_np, AFFINE, affine


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
    Q = to_np(child_shape_matrices(f64(beta_v), f64(beta_m)))
    assert Q.shape == (5, 4, 2, 2)
    six = np.concatenate([beta_v, beta_m], axis=1)
    for k, idx in enumerate(CHILD_VERTEX_INDICES):
        expected = to_np(shape_matrix(f64(six[:, list(idx)])))
        assert np.array_equal(Q[:, k], expected)


@pytest.mark.parametrize(
    "dets, want",
    [
        ([1.0, 1.0, 1.0, 1.0], True),
        ([1.0, 1.0, -1.0, 1.0], False),
        ([1.0, 1.0, 1.0, 0.0], False),
        ([0.0, 0.0, 0.0, 0.0], True),
        ([np.nan] * 4, False),
        ([np.nan, 0.0, 0.0, 0.0], False),
    ],
    ids=[
        "one sign",
        "a sign change",
        "a lone degenerate child",
        "every child degenerate",
        "nan",
        "nan and degenerate",
    ],
)
def test_parity_from_children_asks_one_sign_of_det_q_over_the_four_children(dets, want):
    """An exact zero is a sign of its own: one degenerate child condemns the
    triangle, four do not, there being no change of sign. A NaN fails."""
    Q = np.zeros((1, 4, 2, 2))
    Q[0, :, 0, 0] = dets
    Q[0, :, 1, 1] = 1.0
    assert to_np(parity_from_children(f64(Q))).tolist() == [want]


def test_midpoint_deviation_pairs_midpoint_with_opposite_edge():
    v = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    m = np.array([[[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]]])  # exact affine images
    assert np.allclose(to_np(midpoint_deviation(f64(v), f64(m))), 0.0)
    shifted = m.copy()
    shifted[0, 1] += np.array([0.0, 0.25])
    got = to_np(midpoint_deviation(f64(v), f64(shifted)))
    assert np.allclose(got, [[0.0, 0.25, 0.0]])


H0 = 0.05


def six(tri):
    """Vertices ``(n, 3, 2)`` to the six samples ``(n, 6, 2)``: vertices, then ``m_i`` opposite vertex ``i``."""
    m = np.stack(
        (
            (tri[:, 1] + tri[:, 2]) / 2,
            (tri[:, 2] + tri[:, 0]) / 2,
            (tri[:, 0] + tri[:, 1]) / 2,
        ),
        axis=1,
    )
    return np.concatenate((tri, m), axis=1)


def roots(shift=(0.0, 0.0), h0=H0):
    """The six samples of both level-0 root triangles of cell size ``h0``."""
    return six(h0 * np.asarray(ROOT_SHAPES, dtype=np.float64) + np.asarray(shift))


def explicit_status(beta6, det6, sigma6, cls, level, h0, min_img_sep):
    """The criterion, :func:`_status`, on six given ``det A`` and ``sigma_min A``, with ``Q`` and :func:`child_maps` built from ``beta6``."""
    Q = child_shape_matrices(beta6[:, :3], beta6[:, 3:])
    return _status(beta6, Q, child_maps(Q, cls), det6, sigma6, level, h0, min_img_sep)


def status(theta6, beta, det, level=0, h0=H0, min_img_sep=0.01, sigma=None):
    """The criterion's status of the two root triangles, ``beta``, ``det`` and ``sigma`` numpy maps of ``(..., 2)``.

    ``sigma`` defaults to ``+inf`` at every sample, which no deviation fails,
    so the tests of the other flags read the children's check alone.
    """
    n = theta6.shape[0]
    sigma6 = np.full(theta6.shape[:-1], np.inf) if sigma is None else sigma(theta6)
    return to_np(
        explicit_status(
            f64(beta(theta6)),
            f64(det(theta6)),
            f64(sigma6),
            ROOT_CLASS,
            i64([level] * n),
            h0,
            min_img_sep,
        )[0]
    )


A = AFFINE


def affine_det(p):
    return np.full(p.shape[:-1], np.linalg.det(A))


@pytest.mark.parametrize(
    "beta, det",
    [
        (affine, affine_det),
        (lambda p: p * np.array([1.0, -1.0]), lambda p: -np.ones(p.shape[:-1])),
    ],
    ids=["affine", "reflecting"],
)
def test_an_affine_map_converges_either_way_round(beta, det):
    assert status(roots(), beta, det).tolist() == [LEAF_CONVERGED] * 2


def test_a_fold_sets_both_parity_flags():
    def fold(p):
        return np.stack((p[..., 0], p[..., 1] ** 2), axis=-1)

    got = status(roots((0.0, -0.4 * H0)), fold, lambda p: 2.0 * p[..., 1])
    assert all(s & LEAF_APPROX_PARITY_UNRESOLVED for s in got)
    assert all(s & LEAF_JACOBIAN_PARITY_UNRESOLVED for s in got)


def test_a_nonfinite_image_is_the_row_s_only_flag():
    def beta(p):
        out = affine(p)
        out[0, 4] = np.nan
        return out

    def det(p):

        return np.where(np.arange(6) == 1, np.nan, 1.0) * np.ones(p.shape[:-1])

    got = status(roots(), beta, det)
    assert got.tolist() == [LEAF_RAYTRACE_NONFINITE, LEAF_JACOBIAN_PARITY_UNRESOLVED]


@pytest.mark.parametrize("sample", range(6))
def test_one_flipped_det_sign_flags_jacobian_parity_on_its_row_alone(sample):
    def det(p):
        out = np.full(p.shape[:-1], np.linalg.det(A))
        out[1, sample] *= -1
        return out

    assert status(roots(), affine, det).tolist() == [
        LEAF_CONVERGED,
        LEAF_JACOBIAN_PARITY_UNRESOLVED,
    ]


@pytest.mark.parametrize(
    "bad, want",
    [
        (np.nan, LEAF_JACOBIAN_PARITY_UNRESOLVED),
        (np.inf, LEAF_CONVERGED),
        (-np.inf, LEAF_JACOBIAN_PARITY_UNRESOLVED),
    ],
    ids=["nan", "inf", "-inf"],
)
def test_a_nonfinite_det_is_read_by_its_sign_and_a_nan_has_none(bad, want):
    """``det A > 0`` elsewhere: an infinity of that sign passes, of the other fails, and a NaN fails both sign tests."""

    def det(p):
        out = np.full(p.shape[:-1], np.linalg.det(A))
        out[0, 3] = bad
        return out

    assert np.linalg.det(A) > 0
    assert status(roots(), affine, det).tolist() == [want, 0]


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_an_exactly_zero_det_is_no_sign_so_it_flags_jacobian_parity(sign):
    """A zero sample puts a critical curve on the triangle, whatever the others' sign."""

    def det(p):
        out = np.full(p.shape[:-1], sign)
        out[0, 3] = 0.0
        return out

    assert status(roots(), affine, det).tolist() == [
        LEAF_JACOBIAN_PARITY_UNRESOLVED,
        0,
    ]


def test_the_sign_comes_from_det_a_itself_however_small():
    def det(p):
        out = np.full(p.shape[:-1], 1e-300)
        out[0, 2] = -1e-300
        return out

    assert status(roots(), affine, det).tolist() == [LEAF_JACOBIAN_PARITY_UNRESOLVED, 0]


def test_the_jacobian_flags_add_to_a_deviation_failure_and_never_clear_it():
    def curved(p):
        return np.stack((p[..., 0] + 50.0 * p[..., 0] ** 2, p[..., 1]), axis=-1)

    def det(p):
        out = 1.0 + 100.0 * p[..., 0]
        out[1, 2] *= -1
        return out

    assert status(roots(), curved, det).tolist() == [
        LEAF_CONVERGENCE_FAILED,
        LEAF_CONVERGENCE_FAILED | LEAF_JACOBIAN_PARITY_UNRESOLVED,
    ]


def test_the_level_scales_the_deviation_threshold_with_the_triangle():
    def curved(p):
        return np.stack((p[..., 0] + 5.0 * p[..., 0] ** 2, p[..., 1]), axis=-1)

    def det(p):
        return 1.0 + 10.0 * p[..., 0]

    coarse = status(roots(h0=H0), curved, det, level=0, h0=H0)
    fine = status(roots(h0=H0), curved, det, level=1, h0=2 * H0)
    assert np.array_equal(coarse, fine)


def test_the_leaf_flags_are_distinct_single_bits():
    flags = [
        LEAF_CONVERGENCE_FAILED,
        LEAF_APPROX_PARITY_UNRESOLVED,
        LEAF_JACOBIAN_PARITY_UNRESOLVED,
        LEAF_RAYTRACE_NONFINITE,
    ]
    assert LEAF_CONVERGED == 0 and len(set(flags)) == len(flags)
    assert all(f > 0 and f & (f - 1) == 0 for f in flags)


def test_a_small_sigma_min_at_one_sample_fails_the_deviation_on_its_row_alone():
    """The children pass this mild curve; ``sigma_min(A) = 0.1`` at one sample of row 1 does not."""

    def curved(p):
        return np.stack((p[..., 0] + 2.0 * p[..., 0] ** 2, p[..., 1]), axis=-1)

    def det(p):
        return 1.0 + 4.0 * p[..., 0]

    def honest(p):
        return np.ones(p.shape[:-1])

    def small(p):
        out = np.ones(p.shape[:-1])
        out[1, 4] = 0.1
        return out

    assert status(roots(), curved, det, sigma=honest).tolist() == [0, 0]
    assert status(roots(), curved, det, sigma=small).tolist() == [
        LEAF_CONVERGED,
        LEAF_CONVERGENCE_FAILED,
    ]


def test_a_nan_or_zero_sigma_min_fails_the_deviation_check():
    """``deviation < sigma_min * min_img_sep`` fails on a NaN, and on ``0 < 0``."""

    def sigma(value):
        def fn(p):
            out = np.ones(p.shape[:-1])
            out[1, 2] = value
            return out

        return fn

    for value in (np.nan, 0.0):
        assert status(roots(), affine, affine_det, sigma=sigma(value)).tolist() == [
            LEAF_CONVERGED,
            LEAF_CONVERGENCE_FAILED,
        ]


def test_lens_status_s_deviation_and_sigma_min_on_an_affine_map_and_on_a_collapse():
    theta6 = roots()
    n = theta6.shape[0]
    level, det6 = i64([0] * n), f64(np.ones((n, 6)))
    s = np.linalg.svd(A, compute_uv=False).min()
    _, dev, sig = explicit_status(
        f64(affine(theta6)), det6, f64(np.full((n, 6), s)), ROOT_CLASS, level, H0, 0.01
    )
    assert np.allclose(to_np(dev), 0.0, atol=1e-15)
    assert np.allclose(to_np(sig), s)
    _, dev, sig = explicit_status(
        f64(np.zeros_like(theta6)),
        det6,
        f64(np.zeros((n, 6))),
        ROOT_CLASS,
        level,
        H0,
        0.01,
    )
    assert to_np(sig).tolist() == [0.0, 0.0]
    assert to_np(affine_error(dev, sig)).tolist() == [np.inf, np.inf]


def test_affine_error_is_below_min_img_sep_exactly_where_the_deviation_checks_pass():
    rng = np.random.default_rng(13)
    n = 512
    theta6 = np.repeat(roots(), n // 2, axis=0)
    beta6 = affine(theta6) + rng.normal(scale=1e-3, size=theta6.shape)
    sigma6 = rng.uniform(0.05, 1.0, (n, 6))
    cls = mesh_backend.repeat(ROOT_CLASS, n // 2, axis=0)
    level = i64([0] * n)
    got, deviation, sigma_min = explicit_status(
        f64(beta6), f64(np.ones((n, 6))), f64(sigma6), cls, level, H0, 0.01
    )
    passed = (to_np(got) & LEAF_CONVERGENCE_FAILED) == 0
    error = to_np(affine_error(deviation, sigma_min))
    assert 0 < passed.sum() < n
    assert np.array_equal(error < 0.01, passed)


def descended(levels=3, fov=4.0, init_res=2):
    """Every triangle of levels ``0..levels`` of a lattice: its six lens-plane samples, class and level, and ``h0``."""
    lat = make_lattice(fov, 0.0, 0.0, init_res, levels + 1)
    ij, cls = initial_triangles(init_res, lat.level, ROOT_CLASS)
    table = mesh_backend.as_array(CHILD_VERTEX_INDICES, dtype=mesh_backend.int64)
    theta6, classes, level = [], [], []
    for d in range(levels + 1):
        six_ij = mesh_backend.concatenate((ij, midpoint_ij(ij)), dim=1)
        theta6.append(to_np(lattice_xy(lat, six_ij)))
        classes.append(to_np(cls))
        level.append(np.full(ij.shape[0], d))
        ij, cls = six_ij[:, table].reshape(-1, 3, 2), COMPOSE[cls].reshape(-1)
    return (
        np.concatenate(theta6),
        np.concatenate(classes),
        np.concatenate(level),
        lat.h0,
    )


def quadratic(p):
    x, y = p[..., 0], p[..., 1]
    return np.stack(
        (
            0.3 + 0.7 * x - 0.2 * y + 0.5 * x * x - 0.4 * x * y + 0.1 * y * y,
            -0.1 + 0.1 * x + 0.9 * y - 0.3 * x * x + 0.6 * x * y + 0.2 * y * y,
        ),
        axis=-1,
    )


def quadratic_jacobian(p):
    x, y = p[..., 0], p[..., 1]
    J = np.empty(p.shape[:-1] + (2, 2))
    J[..., 0, 0] = 0.7 + 1.0 * x - 0.4 * y
    J[..., 0, 1] = -0.2 - 0.4 * x + 0.2 * y
    J[..., 1, 0] = 0.1 - 0.6 * x + 0.6 * y
    J[..., 1, 1] = 0.9 + 0.6 * x + 0.4 * y
    return J


def test_the_fit_is_exact_for_a_quadratic_map_at_all_six_samples():
    theta6, cls, level, h0 = descended()
    assert set(cls.tolist()) == set(range(6))
    got = to_np(fitted_jacobian(f64(quadratic(theta6)), i64(cls), i64(level), h0))
    assert got.shape == (theta6.shape[0], 6, 2, 2)
    np.testing.assert_allclose(got, quadratic_jacobian(theta6), rtol=0, atol=1e-11)


def test_a_constant_map_fits_to_exactly_zero():
    theta6, cls, level, h0 = descended()
    beta6 = np.broadcast_to(np.array([0.3, -1.7]), theta6.shape).copy()
    got = to_np(fitted_jacobian(f64(beta6), i64(cls), i64(level), h0))
    assert np.array_equal(got, np.zeros_like(got))


def test_children_sigma_min_is_that_of_the_child_maps():
    theta6, cls, level, h0 = descended(levels=1)
    beta6 = f64(quadratic(theta6))
    Q = child_shape_matrices(beta6[:, :3], beta6[:, 3:])
    maps = child_maps(Q, i64(cls))
    scale = (2.0 ** (level + 1) / h0)[:, None]
    want = (to_np(sigma_min_2x2(maps)) * scale).min(axis=1)
    got = to_np(children_sigma_min(Q, i64(cls), i64(level), h0))
    np.testing.assert_allclose(got, want, rtol=1e-14)


def test_the_fit_is_the_edge_derivative_formula_and_its_inverses_are_the_tables():
    """A_fit(theta_i) = [D_ij, D_ik] @ inv([theta_j - theta_i, theta_k - theta_i]) with D_ij = 4 beta(m_k) - 3 beta_i - beta_j; A_fit(m_k) is the mean of its edge's two."""
    theta6, cls, level, h0 = descended()
    beta6 = np.random.default_rng(1).normal(size=theta6.shape)
    tv, bv, bm = theta6[:, :3], beta6[:, :3], beta6[:, 3:]
    A = np.empty((theta6.shape[0], 3, 2, 2))
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        D = np.stack(
            (
                4 * bm[:, k] - 3 * bv[:, i] - bv[:, j],
                4 * bm[:, j] - 3 * bv[:, i] - bv[:, k],
            ),
            axis=-1,
        )
        E_inv = np.linalg.inv(
            np.stack((tv[:, j] - tv[:, i], tv[:, k] - tv[:, i]), axis=-1)
        )
        table = (2.0**level / h0)[:, None, None] * to_np(PINV0)[to_np(COMPOSE)[cls, i]]
        np.testing.assert_allclose(table, E_inv, rtol=1e-12, atol=1e-12)
        A[:, i] = D @ E_inv
    want = np.stack(
        [0.5 * (A[:, (k + 1) % 3] + A[:, (k + 2) % 3]) for k in range(3)], axis=1
    )
    got = to_np(fitted_jacobian(f64(beta6), i64(cls), i64(level), h0))
    np.testing.assert_allclose(got[:, :3], A, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got[:, 3:], want, rtol=1e-10, atol=1e-10)


def test_lens_status_is_the_criterion_on_the_fit_and_returns_its_corners():
    theta6, cls, level, h0 = descended()
    beta6 = f64(np.random.default_rng(2).normal(size=theta6.shape))
    A = fitted_jacobian(beta6, i64(cls), i64(level), h0)
    det6 = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    Q = child_shape_matrices(beta6[:, :3], beta6[:, 3:])
    want = _status(
        beta6, Q, child_maps(Q, i64(cls)), det6, sigma_min_2x2(A), i64(level), h0, 0.1
    )
    status, deviation, sigma_min, corners = lens_status(
        beta6, i64(cls), i64(level), h0, 0.1
    )
    for got, w in zip((status, deviation, sigma_min), want):
        assert np.array_equal(to_np(got), to_np(w))
    assert np.array_equal(to_np(corners), to_np(A[:, :3]))


def test_a_nan_det_reads_as_parity_unresolved_and_nothing_else():
    theta6, cls, level, h0 = descended(levels=1)
    beta6 = f64(theta6[:1])  # the identity map: finite images, no deviation
    Q = child_shape_matrices(beta6[:, :3], beta6[:, 3:])
    status, _, _ = _status(
        beta6,
        Q,
        child_maps(Q, i64(cls[:1])),
        f64([[np.nan, 1, 1, 1, 1, 1]]),
        f64(np.ones((1, 6))),
        i64(level[:1]),
        h0,
        0.1,
    )
    assert int(to_np(status)[0]) == LEAF_JACOBIAN_PARITY_UNRESOLVED


@pytest.mark.parametrize(
    "module", ["caustics.func", "caustics.lenses.func", "caustics.lenses.func.adaptive"]
)
def test_the_jacobian_nonfinite_flag_is_gone(module):
    assert not hasattr(importlib.import_module(module), "LEAF_JACOBIAN_NONFINITE")
