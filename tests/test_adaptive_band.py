"""The critical band: which leaves it holds, and its samples."""

import numpy as np

from caustics.lenses.func.adaptive.band import build_band, in_band
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
