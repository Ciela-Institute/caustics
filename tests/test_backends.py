import numpy as np
import pytest
import torch

from caustics.backend_obj import Backend

BACKENDS = ["torch", "jax"]


@pytest.fixture(params=BACKENDS)
def b(request):
    return Backend(request.param)


def to_np(backend, array):
    return backend.to_numpy(array)


class TestBackendFull:

    def test_creation(self, b):
        arr = b.make_array([1.0, 2.0, 3.0])
        assert isinstance(arr, b.array_type)
        assert b.numel(arr) == 3

    def test_dtypes(self, b):
        assert b.float32 is not None
        assert b.float64 is not None
        assert b.int32 is not None
        arr = b.make_array([1], dtype=b.int32)
        l_arr = b.long(arr)
        assert b.to_numpy(l_arr).dtype == np.int64

    @pytest.mark.parametrize(
        "op",
        [
            "sin",
            "cos",
            "cosh",
            "tanh",
            "exp",
            "log",
            "log10",
            "sqrt",
            "abs",
            "arcsin",
            "arccos",
            "arctan",
            "arcsinh",
            "arccosh",
            "atanh",
        ],
    )
    def test_unary_ops(self, b, op):
        val = 0.5
        arr = b.make_array([val])
        res = to_np(b, getattr(b, op)(arr))
        ref = getattr(np, op if not op.startswith("at") else op.replace("at", "arct"))(
            val
        )
        np.testing.assert_allclose(res, [ref], rtol=1e-5, atol=1e-7)

    def test_arctan2(self, b):
        res = to_np(b, b.arctan2(b.make_array(1.0), b.make_array(1.0)))
        np.testing.assert_allclose(res, np.arctan2(1.0, 1.0))

    @pytest.mark.parametrize("op", ["sum", "mean", "std", "max", "min", "prod"])
    def test_reductions(self, b, op):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr = b.make_array(data)
        res = to_np(b, getattr(b, op)(arr))
        np.testing.assert_allclose(res, getattr(np, op)(data), rtol=1e-4)
        res_dim = to_np(b, getattr(b, op)(arr, dim=0))
        np.testing.assert_allclose(res_dim, getattr(np, op)(data, axis=0), rtol=1e-4)

    def test_stack_cat_reshape(self, b):
        a1 = b.make_array([1, 2])
        a2 = b.make_array([3, 4])
        stacked = b.stack([a1, a2], dim=0)
        assert b.to_numpy(stacked).shape == (2, 2)
        catted = b.concatenate([a1, a2], dim=0)
        assert b.to_numpy(catted).shape == (4,)
        viewed = b.view(catted, (2, 2))
        assert b.to_numpy(viewed).shape == (2, 2)

    def test_transpose(self, b):
        data = np.random.rand(2, 3, 4)
        arr = b.make_array(data)
        res = to_np(b, b.transpose(arr, 0, 1))
        np.testing.assert_array_equal(res, data.transpose(1, 0, 2))

    def test_flip_roll(self, b):
        data = np.array([1, 2, 3])
        arr = b.make_array(data)
        flipped = to_np(b, b.flip(arr, (0,)))
        np.testing.assert_array_equal(flipped, [3, 2, 1])
        rolled = to_np(b, b.roll(arr, 1, dims=0))
        np.testing.assert_array_equal(rolled, [3, 1, 2])

    def test_where_all_any(self, b):
        mask = b.make_array([True, False])
        x = b.make_array([1, 1])
        y = b.make_array([0, 0])
        res = to_np(b, b.where(mask, x, y))
        np.testing.assert_array_equal(res, [1, 0])
        assert b.any(mask)
        assert not b.all(mask)

    def test_at_indices(self, b):
        arr = b.zeros((3,))
        idx = b.make_array([0, 2], dtype=b.int32)
        vals = b.make_array([10.0, 20.0])
        arr = b.fill_at_indices(arr, idx, vals)
        np.testing.assert_array_equal(to_np(b, arr), [10.0, 0.0, 20.0])

    def test_fft_logic(self, b):
        n = 16
        data = np.random.randn(n, n).astype(np.float32)
        arr = b.make_array(data)
        tilde = b.fft.rfft2(arr, s=(n, n))
        res = b.fft.irfft2(tilde, s=(n, n))
        np.testing.assert_allclose(to_np(b, res.real), data, atol=1e-4)

    def test_random(self, b):
        shape = (10, 10)
        r1 = to_np(b, b.randn(*shape))
        r2 = to_np(b, b.randn(*shape))
        assert not np.allclose(r1, r2)
        assert r1.shape == shape

    def test_grad_and_vmap(self, b):
        def simple_func(x):
            return b.sum(x**2)

        data = np.array([1.0, 2.0, 3.0])
        arr = b.make_array(data)
        if b.backend == "jax":
            g_fn = b.grad(simple_func)
            grad_val = to_np(b, g_fn(arr))
        else:
            t_arr = torch.tensor(data, requires_grad=True)
            loss = torch.sum(t_arr**2)
            loss.backward()
            grad_val = t_arr.grad.numpy()
        np.testing.assert_allclose(grad_val, 2 * data)

    def test_conv2d(self, b):
        img = b.ones((1, 1, 5, 5))
        kernel = b.ones((1, 1, 3, 3))
        padding = 1 if b.backend == "torch" else "SAME"
        res = b.conv2d(img, kernel, padding=padding)
        assert b.to_numpy(res).shape == (1, 1, 5, 5)

    def test_topk(self, b):
        data = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        arr = b.make_array(data)
        k = 2
        res = b.topk(arr, k)
        vals = to_np(b, res[0]) if b.backend == "torch" else to_np(b, res.values)
        idxs = to_np(b, res[1]) if b.backend == "torch" else to_np(b, res.indices)
        np.testing.assert_allclose(vals, [8.0, 5.0])
        np.testing.assert_array_equal(idxs, [3, 1])

    def test_argmax(self, b):
        data = np.array([[1, 5], [8, 2]])
        arr = b.make_array(data)
        assert to_np(b, b.argmax(arr, dim=0))[1] == 0

    def test_linalg_basics(self, b):
        data = np.array([[2.0, 1.0], [1.0, 2.0]])
        arr = b.make_array(data)
        res_norm = to_np(b, b.norm(arr))
        np.testing.assert_allclose(res_norm, np.linalg.norm(data))
        diag_vec = b.make_array([1.0, 2.0])
        res_diag = to_np(b, b.diag(diag_vec))
        np.testing.assert_array_equal(res_diag, np.diag([1.0, 2.0]))

    @pytest.mark.parametrize(
        "val", [np.array([1.0, 2.0, 3.0, 4.0]), np.array([[1.0, 2.0], [3.0, 4.0]])]
    )
    def test_special_functions(self, b, val):
        arr = b.make_array(val)
        res_gamma = to_np(b, b.gammaln(arr))
        from scipy.special import gammaln

        np.testing.assert_allclose(res_gamma, gammaln(val), rtol=1e-5, atol=1e-12)
        res_j1 = to_np(b, b.bessel_j1(arr))
        from scipy.special import j1

        np.testing.assert_allclose(res_j1, j1(val), rtol=1e-5)

    def test_grids(self, b):
        res_range = to_np(b, b.arange(0, 5, 1))
        np.testing.assert_array_equal(res_range, np.arange(5))
        res_lin = to_np(b, b.linspace(0, 1, 5))
        np.testing.assert_allclose(res_lin, np.linspace(0, 1, 5))
        x = b.linspace(0, 1, 3)
        y = b.linspace(0, 1, 2)
        grid_x, grid_y = b.meshgrid(x, y, indexing="ij")
        assert to_np(b, grid_x).shape == (3, 2)
        assert to_np(b, grid_y).shape == (3, 2)

    def test_upsample_avgpool(self, b):
        data = np.ones((1, 1, 4, 4), dtype=np.float32)
        arr = b.make_array(data)
        pooled = b.avg_pool2d(arr, kernel=2)
        assert to_np(b, pooled).shape == (1, 1, 2, 2)
        np.testing.assert_allclose(to_np(b, pooled), 1.0)
        upsampled = b.upsample2d(arr, scale_factor=2, method="bilinear")
        assert to_np(b, upsampled).shape == (1, 1, 8, 8)

    def test_comparisons(self, b):
        a = b.make_array([1.0, 2.0, np.nan])
        b_arr = b.make_array([1.0, 3.0, 0.0])
        assert to_np(b, b.isnan(a))[2]
        assert not to_np(b, b.isfinite(a))[2]
        res_min = to_np(b, b.minimum(a[:2], b_arr[:2]))
        np.testing.assert_array_equal(res_min, [1.0, 2.0])

    def test_pad_tile(self, b):
        data = np.array([[1, 2], [3, 4]])
        arr = b.make_array(data)
        tiled = to_np(b, b.tile(arr, (2, 2)))
        assert tiled.shape == (4, 4)
        padding = (1, 1, 1, 1)
        padded = to_np(b, b.pad(arr, padding, mode="constant"))
        assert padded.shape == (4, 4)
        assert padded[0, 0] == 0

    def test_jacobian_consistency(self, b):
        def f(x):
            return b.stack([x[0] ** 2, b.sin(x[1])])

        x_val = np.array([2.0, 0.0])
        arr = b.make_array(x_val)
        if b.backend == "jax":
            jac = to_np(b, b.jacobian(f, arr))
        else:
            import torch

            jac = torch.autograd.functional.jacobian(
                lambda v: torch.stack([v[0] ** 2, torch.sin(v[1])]), torch.tensor(x_val)
            ).numpy()
        np.testing.assert_allclose(jac, [[4.0, 0.0], [0.0, 1.0]], atol=1e-5)

    def test_vmap_consistency(self, b):
        def square(x):
            return x**2

        data = np.array([1.0, 2.0, 3.0])
        arr = b.make_array(data)
        v_square = b.vmap(square)
        res = to_np(b, v_square(arr))
        np.testing.assert_array_equal(res, [1.0, 4.0, 9.0])

    def test_linalg_advanced(self, b):
        data = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        arr = b.make_array(data)
        res_inv = to_np(b, b.linalg.inv(arr))
        np.testing.assert_allclose(res_inv, np.linalg.inv(data), rtol=1e-5)
        res_det = to_np(b, b.linalg.det(arr))
        np.testing.assert_allclose(res_det, np.linalg.det(data), rtol=1e-5)

    def test_special_functions_extended(self, b):
        val = np.array([0.5, 1.5, 2.5])
        arr = b.make_array(val)
        from scipy.special import gammaln

        res_lgamma = to_np(b, b.lgamma(arr))
        np.testing.assert_allclose(res_lgamma, gammaln(val), rtol=1e-5)

    def test_device_logic(self, b):
        arr = b.make_array([1.0, 2.0])
        device = b.device(arr)
        moved_arr = b.to(arr, device=device)
        assert b.numel(moved_arr) == 2
        assert isinstance(moved_arr, b.array_type)

    def test_jacfwd_consistency(self, b):
        def simple_vec_func(x):
            return b.stack([x[0] ** 3, x[1] ** 2])

        data = np.array([2.0, 3.0])
        arr = b.make_array(data)
        if b.backend == "jax":
            jac_val = to_np(b, b.jacfwd(simple_vec_func)(arr))
        else:
            jac_val = to_np(b, b.jacfwd(simple_vec_func)(arr))
        np.testing.assert_allclose(jac_val, [[12.0, 0.0], [0.0, 6.0]], atol=1e-5)

    def test_searchsorted(self, b):
        sorted_arr = b.make_array([1.0, 2.0, 3.0, 4.0, 5.0])
        values = b.make_array([2.5, 0.5, 6.0])
        indices = to_np(b, b.searchsorted(sorted_arr, values))
        np.testing.assert_array_equal(indices, [2, 0, 5])

    def test_nan_to_num(self, b):
        data = np.array([np.nan, np.inf, -np.inf, 1.0])
        arr = b.make_array(data)
        res = to_np(b, b.nan_to_num(arr, posinf=1e10, neginf=-1e10))
        assert res[0] == 0.0
        assert res[1] == 1e10
        assert res[2] == -1e10
        assert res[3] == 1.0

    def test_constants(self, b):
        np.testing.assert_allclose(b.pi, np.pi)
        assert b.inf > 1e30
        assert b.nan != b.nan

    def test_chunking(self, b):
        data = np.arange(10).astype(np.float32)
        arr = b.make_array(data)
        chunks = b.chunk(arr, 2, dim=0)
        assert len(chunks) == 2
        np.testing.assert_array_equal(to_np(b, chunks[0]), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(to_np(b, chunks[1]), [5, 6, 7, 8, 9])
