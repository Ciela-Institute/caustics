"""The backend the adaptive bookkeeping runs on, and the conversions at its boundary."""

from typing import NamedTuple

import caskade as ck
import numpy as np
import pytest

from adaptive_maps import (
    affine,
    affine_jacobian,
    assert_same,
    numpy_lens,
    sie_like,
    sie_like_jacobian,
    SIE_LIKE_LENS,
    stack_2x2,
    to_np,
    user_build,
)

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive import (
    build_closed_lens_mesh,
    build_closed_magnification_region,
    build_lens_mesh,
    caustic_area,
    critical_curves_and_caustics,
    extend_lens_mesh,
    forward_raytrace,
    magnification_region,
    total_magnification,
)
from caustics.lenses.func.adaptive.mesh_backend import (
    TorchMeshBackend,
    map_arrays,
    mesh_backend,
    to_mesh,
    to_user,
)


class Pair(NamedTuple):
    a: object
    b: object


def test_the_torch_mesh_backend_computes_in_torch_and_leaves_the_global_backend():
    import torch

    before = ck.backend.backend
    torch_backend = TorchMeshBackend()
    x = torch_backend.as_array(np.array([3, 1, 2]), dtype=torch_backend.int64)
    assert isinstance(x, torch.Tensor)
    assert torch_backend.to_numpy(torch_backend.sort(x)).tolist() == [1, 2, 3]
    assert ck.backend.backend == before and backend.backend == before


def test_map_arrays_reaches_arrays_in_nested_tuples_and_keeps_the_rest():
    value = Pair(a=(np.ones(2), 3.0), b=Pair(a=None, b=np.zeros(1)))
    got = map_arrays(value, lambda a: a + 1)
    assert isinstance(got, Pair) and isinstance(got.b, Pair)
    assert type(got.a) is tuple and got.a[1] == 3.0 and got.b.a is None
    np.testing.assert_array_equal(got.a[0], [2.0, 2.0])
    np.testing.assert_array_equal(got.b.b, [1.0])


def test_map_arrays_returns_the_value_itself_when_nothing_changes():
    value = Pair(a=np.ones(2), b=(1, None))
    assert map_arrays(value, lambda a: a) is value


@pytest.mark.skipif(backend.backend != "torch", reason="the torch path only")
def test_under_torch_the_bookkeeping_is_the_users_backend_and_nothing_converts():
    value = Pair(a=backend.as_array([1.0, 2.0]), b=2.5)
    assert mesh_backend is backend
    assert to_mesh(value) is value
    assert to_user(value) is value


def _arrays(value):
    found = []
    map_arrays(value, lambda a: found.append(a) or a)
    return found


def test_every_public_function_returns_the_users_arrays():
    """0-d and empty results included: an affine lens has no critical curve."""
    mesh = build_lens_mesh(
        SIE_LIKE_LENS.raytrace, SIE_LIKE_LENS.jacobian_lens_equation, 4.0, 8, 0.1
    )
    bx = backend.as_array([0.05, 0.3], dtype=backend.float64)
    by = backend.as_array([0.02, -0.1], dtype=backend.float64)
    affine = build_lens_mesh(
        lambda x, y: (0.7 * x, 0.9 * y),
        lambda x, y: stack_2x2(
            0.7 * backend.ones_like(x), 0.0 * x, 0.0 * x, 0.9 * backend.ones_like(x)
        ),
        4.0,
        4,
        0.25,
    )
    results = [
        mesh,
        critical_curves_and_caustics(mesh),
        critical_curves_and_caustics(affine),
        forward_raytrace(bx, by, SIE_LIKE_LENS.raytrace, mesh),
    ]
    for value in results:
        for a in _arrays(value):
            assert isinstance(a, backend.array_type), type(a)
    assert to_np(critical_curves_and_caustics(affine).lens).shape == (0, 2)


def test_extending_a_mesh_leaves_the_mesh_it_was_given_unchanged():
    mesh = build_lens_mesh(
        SIE_LIKE_LENS.raytrace, SIE_LIKE_LENS.jacobian_lens_equation, 4.0, 4, 0.1
    )
    before = [to_np(a).copy() for a in _arrays(mesh)]
    extend_lens_mesh(
        mesh, SIE_LIKE_LENS.raytrace, SIE_LIKE_LENS.jacobian_lens_equation, 6.0
    )
    for a, b in zip(_arrays(mesh), before):
        np.testing.assert_array_equal(to_np(a), b)


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_a_jax_build_compiles_only_for_its_padded_lens_batches():
    """Before, every bookkeeping op compiled per new shape: thousands per build."""
    import jax

    compiles = []
    jax.monitoring.register_event_duration_secs_listener(
        lambda event, duration, **kw: (
            compiles.append(event)
            if event == "/jax/core/compile/backend_compile_duration"
            else None
        )
    )
    lens_ = numpy_lens(sie_like, sie_like_jacobian)
    jax.clear_caches()
    before = len(compiles)
    build_lens_mesh(lens_.raytrace, lens_.jacobian_lens_equation, 4.0, 4, 0.1)
    assert len(compiles) - before <= 150


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_handing_back_an_array_of_a_new_shape_compiles_nothing():
    """Every public result is handed back array by array, each of a shape new to jax."""
    import jax

    compiles = []
    jax.monitoring.register_event_duration_secs_listener(
        lambda event, duration, **kw: (
            compiles.append(event)
            if event == "/jax/core/compile/backend_compile_duration"
            else None
        )
    )
    a = mesh_backend.as_array(np.arange(3 * 1237).reshape(1237, 3))
    before = len(compiles)
    got = to_user(a)
    assert len(compiles) == before
    np.testing.assert_array_equal(to_np(got), to_np(a))


@pytest.mark.skipif(backend.backend != "jax", reason="under torch nothing is copied")
def test_a_handed_back_array_shares_no_memory_with_the_bookkeeping():
    a = mesh_backend.as_array(np.arange(6.0))
    got = to_user(a)
    a[0] = 99.0  # the bookkeeping writes in place
    assert to_np(got)[0] == 0.0


def test_scalars_of_the_users_backend_build_what_plain_floats_do():
    """A lens parameter's value is a 0-d array of the user's backend: as a
    number, and inside a list of centers."""
    lens_ = numpy_lens(sie_like, sie_like_jacobian)

    def scalar(v):
        return backend.as_array(v, dtype=backend.float64)

    def built(number):
        kw = dict(x0=number(0.0), y0=number(0.0), centers=[(number(0.0), number(0.0))])
        args = (lens_.raytrace, lens_.jacobian_lens_equation, number(4.0), 4)
        mesh = build_lens_mesh(*args, number(0.1), **kw)
        closed, _ = build_closed_lens_mesh(
            *args, number(0.1), growth=number(1.5), max_iters=1, **kw
        )
        return mesh, closed

    for got, want in zip(built(scalar), built(float)):
        assert_same(got, want)


def test_the_magnification_functions_return_the_users_arrays():
    mesh = build_lens_mesh(
        SIE_LIKE_LENS.raytrace, SIE_LIKE_LENS.jacobian_lens_equation, 4.0, 8, 0.1
    )
    curves = critical_curves_and_caustics(mesh)
    bx = backend.as_array([0.05, 0.3], dtype=backend.float64)
    by = backend.as_array([0.02, -0.1], dtype=backend.float64)
    results = [
        total_magnification(bx, by, mesh),
        magnification_region(mesh, curves, 3.0),
        *build_closed_magnification_region(
            SIE_LIKE_LENS.raytrace,
            SIE_LIKE_LENS.jacobian_lens_equation,
            4.0,
            8,
            0.1,
            3.0,
        ),
    ]
    for value in results:
        for a in _arrays(value):
            assert isinstance(a, backend.array_type), type(a)
    assert isinstance(caustic_area(curves), float)
    assert isinstance(results[1].area, float)


def test_a_magnification_region_leaves_the_mesh_it_was_given_unchanged():
    mesh = build_lens_mesh(
        SIE_LIKE_LENS.raytrace, SIE_LIKE_LENS.jacobian_lens_equation, 4.0, 8, 0.1
    )
    curves = critical_curves_and_caustics(mesh)
    before = [to_np(a).copy() for a in _arrays((mesh, curves))]
    magnification_region(mesh, curves, 3.0, 10.0)
    for a, b in zip(_arrays((mesh, curves)), before):
        np.testing.assert_array_equal(to_np(a), b)


def test_total_magnification_takes_numpy_sources():
    lens_ = numpy_lens(sie_like, sie_like_jacobian)
    mesh = build_lens_mesh(lens_.raytrace, lens_.jacobian_lens_equation, 4.0, 8, 0.1)
    mu = total_magnification(np.array([0.1, 0.2]), np.array([0.0, 0.1]), mesh)
    assert isinstance(mu, backend.array_type) and to_np(mu).shape == (2,)


def test_a_user_mesh_s_lattice_gives_its_corner_in_the_user_s_backend():
    """On an extended lattice, whose origin is not lattice point ``(0, 0)``."""
    mesh, lens_ = user_build(affine, affine_jacobian)
    grown = extend_lens_mesh(mesh, lens_.raytrace, lens_.jacobian_lens_equation, 6.0)
    lat = grown.lattice
    assert lat.origin > 0
    corner = lat.corner
    assert isinstance(corner, backend.array_type)
    np.testing.assert_array_equal(to_np(corner), to_np(lat.lo) - lat.origin * lat.scale)
