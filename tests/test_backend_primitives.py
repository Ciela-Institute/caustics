import numpy as np
import pytest

from caustics.backend_obj import backend
from jax_compiles import compile_count


def test_sort_is_ascending():
    x = backend.as_array(np.array([3, 1, 2, 1], dtype=np.int64))
    assert backend.to_numpy(backend.sort(x)).tolist() == [1, 1, 2, 3]


def test_argsort_is_stable():
    x = backend.as_array(np.array([3, 1, 2, 1], dtype=np.int64))
    assert backend.to_numpy(backend.argsort(x)).tolist() == [1, 3, 2, 0]


def test_unique_returns_sorted_values_and_inverse():
    x = backend.as_array(np.array([3, 1, 2, 1], dtype=np.int64))
    values, inverse = backend.unique(x, return_inverse=True)
    assert backend.to_numpy(values).tolist() == [1, 2, 3]
    assert backend.to_numpy(inverse).reshape(-1).tolist() == [2, 0, 1, 0]


def test_unique_without_inverse_returns_values_only():
    x = backend.as_array(np.array([3, 1, 2, 1], dtype=np.int64))
    assert backend.to_numpy(backend.unique(x)).tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "minlength, expected",
    [
        (None, [1, 0, 2]),
        (1, [1, 0, 2]),
        (3, [1, 0, 2]),
        (5, [1, 0, 2, 0, 0]),
    ],
)
def test_bincount_honours_the_minimum_length(minlength, expected):
    x = backend.as_array(np.array([0, 2, 2], dtype=np.int64))
    got = backend.bincount(x) if minlength is None else backend.bincount(x, minlength)
    assert backend.to_numpy(got).tolist() == expected


def test_bincount_of_an_empty_array_is_all_zeros():
    x = backend.as_array(np.zeros(0, dtype=np.int64), dtype=backend.int64)
    assert backend.to_numpy(backend.bincount(x, minlength=3)).tolist() == [0, 0, 0]


def test_flatnonzero_returns_ascending_indices():
    x = backend.as_array(np.array([3, 1, 2, 1], dtype=np.int64))
    assert backend.to_numpy(backend.flatnonzero(x > 1)).tolist() == [0, 2]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([[0, 1], [1, 0]], [1, 2]),
        (1, [0]),
        (0, []),
    ],
)
def test_flatnonzero_flattens_inputs_before_finding_indices(values, expected):
    x = backend.as_array(np.asarray(values, dtype=np.int64), dtype=backend.int64)
    assert backend.to_numpy(backend.flatnonzero(x)).tolist() == expected


def test_sign_matches_numpy():
    x = backend.as_array(np.array([-2.0, 0.0, 3.0]))
    assert backend.to_numpy(backend.sign(x)).tolist() == [-1.0, 0.0, 1.0]


def test_sign_preserves_nan_like_numpy():
    x = backend.as_array(np.array([np.nan, -2.0, 0.0, 3.0]))
    assert np.array_equal(
        backend.to_numpy(backend.sign(x)),
        np.sign(np.array([np.nan, -2.0, 0.0, 3.0])),
        equal_nan=True,
    )


def test_sign_preserves_integer_dtype_and_values():
    x = backend.as_array(np.array([-2, 0, 3], dtype=np.int64), dtype=backend.int64)
    got = backend.sign(x)
    assert got.dtype == backend.int64
    assert backend.to_numpy(got).tolist() == [-1, 0, 1]


def test_lexsort_matches_numpy_with_the_last_key_primary():
    rng = np.random.default_rng(0)
    a, b, c = (rng.integers(0, 3, 40) for _ in range(3))
    keys = [backend.as_array(k, dtype=backend.int64) for k in (a, b, c)]
    got = backend.to_numpy(backend.lexsort(keys)).tolist()
    assert got == np.lexsort((a, b, c)).tolist()


def test_finfo_exposes_eps_for_both_float_widths():
    assert backend.finfo(backend.float64).eps == pytest.approx(2.220446049250313e-16)
    assert backend.finfo(backend.float32).eps > backend.finfo(backend.float64).eps


def test_integer_dtype_properties_round_trip():
    x = backend.zeros((3,), dtype=backend.int64)
    assert backend.to_numpy(x).dtype == np.int64
    y = backend.zeros((3,), dtype=backend.int8)
    assert backend.to_numpy(y).dtype == np.int8


def test_all_and_any_reduce_over_a_single_axis():
    x = backend.as_array(np.array([[True, True], [True, False]]))
    assert backend.to_numpy(backend.all(x, dim=1)).tolist() == [True, False]
    assert backend.to_numpy(backend.any(x, dim=1)).tolist() == [True, True]


def test_all_and_any_reduce_over_a_tuple_of_axes():
    x = backend.as_array(np.ones((2, 3, 4), dtype=bool))
    assert backend.to_numpy(backend.all(x, dim=(1, 2))).tolist() == [True, True]
    assert backend.to_numpy(backend.any(x, dim=(1, 2))).tolist() == [True, True]


def test_all_and_any_without_dim_still_reduce_everything():
    x = backend.as_array(np.array([[True, False]]))
    assert bool(backend.to_numpy(backend.all(x))) is False
    assert bool(backend.to_numpy(backend.any(x))) is True


_JAX_PADDED = {
    (0, None): 0,
    (1, None): 64,
    (63, None): 64,
    (64, None): 64,
    (65, None): 128,
    (1000, None): 1024,
    (65, 100): 100,
    (3, 3): 3,
    (1000, 1000): 1000,
}


@pytest.mark.parametrize("n, cap", list(_JAX_PADDED))
def test_padded_size_is_n_under_torch_and_a_power_of_two_from_64_under_jax(n, cap):
    want = n if backend.backend == "torch" else _JAX_PADDED[(n, cap)]
    assert backend.padded_size(n, cap) == want


@pytest.mark.parametrize("shape", [(3,), (3, 2), (3, 2, 2)])
def test_pad_rows_repeats_the_last_row(shape):
    x = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    got = backend.to_numpy(backend.pad_rows(backend.as_array(x), 5))
    np.testing.assert_array_equal(got, np.concatenate([x, x[-1:], x[-1:]]))


def test_pad_rows_to_its_own_length_changes_nothing():
    x = np.arange(6, dtype=np.int64).reshape(3, 2)
    got = backend.to_numpy(backend.pad_rows(backend.as_array(x), 3))
    np.testing.assert_array_equal(got, x)


def test_compiles_is_true_only_under_jax():
    assert backend.compiles == (backend.backend == "jax")


def test_read_flag_returns_the_flag_flag_to_host_copied(device):
    for value in (True, False):
        flag = backend.to(backend.as_array(value), device=device)
        assert backend.read_flag(backend.flag_to_host(flag)) is value


def test_while_loop_runs_body_until_cond_fails():
    i, x = backend.while_loop(
        lambda s: s[0] < 5,
        lambda s: (s[0] + 1, s[1] * 2),
        (backend.as_array(0), backend.as_array(1.0)),
    )
    assert int(backend.to_numpy(i)) == 5 and float(backend.to_numpy(x)) == 32.0


def test_pad_rows_repeats_the_last_row_and_unpad_rows_drops_it():
    a = backend.as_array(np.arange(6.0).reshape(3, 2))
    padded = backend.pad_rows(a, 5)
    assert backend.to_numpy(padded).tolist() == [[0, 1], [2, 3], [4, 5], [4, 5], [4, 5]]
    assert backend.to_numpy(backend.unpad_rows(padded, 3)).tolist() == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_pad_rows_and_unpad_rows_compile_nothing_at_a_new_shape():
    a = backend.to(np.arange(14.0).reshape(7, 2))
    before = compile_count()
    backend.unpad_rows(backend.pad_rows(a, 64), 7)
    assert compile_count() == before
