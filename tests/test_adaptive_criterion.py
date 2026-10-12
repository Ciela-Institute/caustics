"""The lens mesh's refinement criterion and its ``LEAF_*`` flags."""

import numpy as np
import pytest

from caustics.lenses.func.adaptive.mesh_backend import mesh_backend
from caustics.lenses.func.adaptive.criterion import (
    child_shape_matrices,
    midpoint_deviation,
    parity_from_children,
    LEAF_APPROX_PARITY_UNRESOLVED,
    LEAF_CONVERGED,
    LEAF_CONVERGENCE_FAILED,
    LEAF_JACOBIAN_NONFINITE,
    LEAF_JACOBIAN_PARITY_UNRESOLVED,
    LEAF_RAYTRACE_NONFINITE,
    affine_error,
    lens_status,
)
from caustics.lenses.func.adaptive.geometry import (
    CHILD_VERTEX_INDICES,
    ROOT_CLASS,
    ROOT_SHAPES,
    shape_matrix,
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


def test_parity_from_children_accepts_a_constant_sign():
    Q = np.tile(np.eye(2), (2, 4, 1, 1))
    assert to_np(parity_from_children(f64(Q))).tolist() == [
        True,
        True,
    ]


def test_parity_from_children_rejects_a_sign_change():
    Q = np.tile(np.eye(2), (1, 4, 1, 1))
    Q[0, 2] = np.array([[0.0, 1.0], [1.0, 0.0]])  # det -1 among three det +1
    assert to_np(parity_from_children(f64(Q))).tolist() == [False]


def test_parity_from_children_condemns_a_lone_degenerate_child():
    Q = np.tile(np.eye(2), (1, 4, 1, 1))
    Q[0, 3] = 0.0  # sign 0 differs from +1
    assert to_np(parity_from_children(f64(Q))).tolist() == [False]


def test_parity_from_children_passes_an_entirely_degenerate_triangle():
    """All four dets exactly zero is a constant sign, so there is no parity
    *change* and the triangle passes. Spec section 4.6: condemning this case
    would be the deviation test in disguise, which is out of scope. A kappa == 1
    sheet is the fixture that reaches it.
    """
    Q = np.zeros((1, 4, 2, 2))
    assert to_np(parity_from_children(f64(Q))).tolist() == [True]


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
    assert to_np(parity_from_children(f64(Q))).tolist() == [False]


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


def status(theta6, beta, det, level=0, h0=H0, min_img_sep=0.01, sigma=None):
    """``lens_status`` of the two root triangles, ``beta``, ``det`` and ``sigma`` numpy maps of ``(..., 2)``.

    ``sigma`` defaults to ``+inf`` at every sample, which no deviation fails,
    so the tests of the other flags read the children's check alone.
    """
    n = theta6.shape[0]
    sigma6 = np.full(theta6.shape[:-1], np.inf) if sigma is None else sigma(theta6)
    return to_np(
        lens_status(
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


def test_an_affine_map_converges():
    assert status(roots(), affine, affine_det).tolist() == [LEAF_CONVERGED] * 2


def test_a_reflecting_affine_map_converges_too():
    def flip(p):
        return p * np.array([1.0, -1.0])

    assert status(roots(), flip, lambda p: -np.ones(p.shape[:-1])).tolist() == [0, 0]


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
    assert got.tolist() == [LEAF_RAYTRACE_NONFINITE, LEAF_JACOBIAN_NONFINITE]


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


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_nonfinite_det_flags_jacobian_nonfinite_not_parity(bad):
    def det(p):
        out = np.full(p.shape[:-1], np.linalg.det(A))
        out[0, 3] = bad
        return out

    assert status(roots(), affine, det).tolist() == [LEAF_JACOBIAN_NONFINITE, 0]


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_an_exactly_zero_det_is_no_sign_so_it_flags_jacobian_parity_not_nonfinite(
    sign,
):
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


def test_the_approximate_parity_bit_is_child_parity():
    rng = np.random.default_rng(5)
    beta6 = f64(rng.normal(size=(64, 6, 2)))
    det6 = f64(np.ones((64, 6)))
    n = 64
    got = to_np(
        lens_status(
            beta6, det6, f64(np.ones((64, 6))), i64([0] * n), i64([0] * n), H0, 0.01
        )[0]
        & LEAF_APPROX_PARITY_UNRESOLVED
    )
    child_ok = to_np(
        parity_from_children(child_shape_matrices(beta6[:, :3], beta6[:, 3:]))
    )
    assert np.array_equal(got != 0, ~child_ok)


def test_the_leaf_flags_are_distinct_single_bits():
    flags = [
        LEAF_CONVERGENCE_FAILED,
        LEAF_APPROX_PARITY_UNRESOLVED,
        LEAF_JACOBIAN_PARITY_UNRESOLVED,
        LEAF_RAYTRACE_NONFINITE,
        LEAF_JACOBIAN_NONFINITE,
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
    _, dev, sig = lens_status(
        f64(affine(theta6)), det6, f64(np.full((n, 6), s)), ROOT_CLASS, level, H0, 0.01
    )
    assert np.allclose(to_np(dev), 0.0, atol=1e-15)
    assert np.allclose(to_np(sig), s)
    _, dev, sig = lens_status(
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
    got, deviation, sigma_min = lens_status(
        f64(beta6), f64(np.ones((n, 6))), f64(sigma6), cls, level, H0, 0.01
    )
    passed = (to_np(got) & LEAF_CONVERGENCE_FAILED) == 0
    error = to_np(affine_error(deviation, sigma_min))
    assert 0 < passed.sum() < n
    assert np.array_equal(error < 0.01, passed)
