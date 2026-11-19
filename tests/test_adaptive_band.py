"""The critical band: which leaves it holds, and its samples."""

import numpy as np

from caustics.lenses.func.adaptive.band import (
    RESOLVED_SPREAD,
    build_band,
    in_band,
    jacobian_det6,
    resolved,
)
from caustics.lenses.func.adaptive.lattice import lattice_xy, make_lattice

from adaptive_maps import f64, i64, to_np


def test_a_band_leaf_has_finite_dets_of_both_classes_zero_counting_positive():
    det6 = f64(
        [
            [1, 2, 3, -1, 1, 1],
            [1, 2, 3, 4, 5, 6],
            [0, 0, 0, 1, 1, 1],
            [0, -1, 1, 1, 1, 1],
            [np.nan, -1, 1, 1, 1, 1],
        ]
    )
    assert to_np(in_band(det6)).tolist() == [True, False, False, True, False]


def test_build_band_numbers_samples_by_key_and_reads_their_values():
    lat = make_lattice(4.0, 0.0, 0.0, 4, 2)
    keys6 = i64([[30, 10, 20, 11, 21, 31], [10, 40, 20, 41, 21, 11]])
    keys = np.array([10, 11, 20, 21, 30, 31, 40, 41])
    table = f64([[k, -k, k / 100.0, k / 1000.0] for k in keys])
    band = build_band(lat, i64([3, 7]), keys6, i64(keys), table)
    assert to_np(band.leaves).tolist() == [3, 7]
    assert np.array_equal(keys[to_np(band.samples)], to_np(keys6))
    assert np.array_equal(to_np(band.source)[:, 0], keys)
    assert np.array_equal(to_np(band.det), keys / 100.0)
    assert np.array_equal(to_np(band.sigma_min), keys / 1000.0)
    ij = np.stack((keys // (lat.n + 1), keys % (lat.n + 1)), axis=-1)
    assert np.array_equal(to_np(band.lens), to_np(lattice_xy(lat, i64(ij))))


def test_jacobian_det6_is_det_at_the_vertices_and_at_the_midpoint_means():
    J = np.random.default_rng(0).normal(size=(7, 3, 2, 2))
    got = to_np(jacobian_det6(f64(J)))
    assert np.allclose(got[:, :3], np.linalg.det(J))
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        assert np.allclose(got[:, 3 + i], np.linalg.det(0.5 * (J[:, j] + J[:, k])))


def test_a_midpoint_det_has_the_same_bits_from_either_side_of_its_edge():
    J = f64(np.random.default_rng(1).normal(size=(500, 3, 2, 2)))
    a = to_np(jacobian_det6(J))
    b = to_np(jacobian_det6(J[:, [0, 2, 1]]))
    assert np.array_equal(a[:, 3], b[:, 3])


def test_resolved_is_a_spread_below_the_threshold_and_a_nan_fails():
    eye = np.eye(2)
    J = np.stack(
        [
            np.stack([eye, eye, eye + d * np.array([[1.0, 0], [0, 0]])])
            for d in (3.9, 4.1)
        ]
    )
    J = np.concatenate([J, np.full((1, 3, 2, 2), np.nan)])
    assert to_np(resolved(f64(J))).tolist() == [True, False, False]
    assert RESOLVED_SPREAD == 4.0
