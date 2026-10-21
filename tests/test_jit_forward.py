import gc
import weakref

import numpy as np
import pytest
from caskade import ActiveContext, Param

from caustics import jit_forward
from caustics.backend_obj import backend
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import NFW, SIE, SIS, ExternalShear, Multiplane, SinglePlane
from jax_compiles import compile_count

only_jax = pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")


def _plane():
    """An SIE and a shear in one plane, every parameter static."""
    cosmology = FlatLambdaCDM(name="cosmo")
    sie = SIE(
        name="sie", cosmology=cosmology, x0=0.1, y0=-0.2, q=0.6, phi=0.4, Rein=1.0
    )
    shear = ExternalShear(
        name="shear", cosmology=cosmology, x0=0.0, y0=0.0, gamma_1=0.05, gamma_2=-0.02
    )
    lens = SinglePlane(
        name="plane", cosmology=cosmology, lenses=[sie, shear], z_l=0.5, z_s=1.5
    )
    return lens, sie, shear


def _f64(a):
    # Both backends; under jax this can compile, so compile-counting tests call it before counting.
    return backend.as_array(np.asarray(a, dtype=np.float64), dtype=backend.float64)


def _points():
    return _f64(np.linspace(-1.5, 1.5, 64)), _f64(np.linspace(1.2, -0.7, 64))


def _close(got, want, atol=1e-14):
    for g, w in zip(
        got if isinstance(got, tuple) else (got,),
        want if isinstance(want, tuple) else (want,),
    ):
        np.testing.assert_allclose(
            backend.to_numpy(g), backend.to_numpy(w), rtol=1e-12, atol=atol
        )


def _state(lens):
    return [p._value for p in lens.topological_ordering() if isinstance(p, Param)]


def test_jit_forward_is_jitted_only_under_jax():
    lens, _, _ = _plane()
    assert jit_forward(lens.raytrace).jitted == (backend.backend == "jax")


def test_jit_forward_matches_the_method_and_follows_dynamic_and_static_values():
    lens, sie, shear = _plane()
    sie.Rein.to_dynamic()
    sie.q.to_dynamic()
    raytrace = jit_forward(lens.raytrace)
    jacobian = jit_forward(lens.jacobian_lens_equation)
    x, y = _points()
    for Rein, q, gamma_1 in ((1.0, 0.6, 0.05), (1.3, 0.45, 0.1)):
        sie.Rein, sie.q, shear.gamma_1 = Rein, q, gamma_1
        for params in (lens.get_values(), lens.get_values("list")):
            _close(raytrace(x, y, params=params), lens.raytrace(x, y, params=params))
            _close(
                jacobian(x, y, params=params),
                lens.jacobian_lens_equation(x, y, params=params),
            )


def test_jit_forward_follows_a_pointer_to_a_traced_value():
    """An NFW in a plane points its redshifts at the plane's; its deflection reads them."""
    # Its distance table is float32 by default, where jitted and eager ops round differently.
    cosmology = FlatLambdaCDM(name="cosmo").to(dtype=backend.float64)
    nfw = NFW(name="nfw", cosmology=cosmology, x0=0.0, y0=0.0, mass=1e13, c=8.0)
    lens = SinglePlane(
        name="plane", cosmology=cosmology, lenses=[nfw], z_l=0.5, z_s=1.5
    )
    lens.z_l.to_dynamic()
    raytrace = jit_forward(lens.raytrace)
    x, y = _points()
    for z_l in (0.5, 0.8):
        params = [_f64(z_l)]
        # Where the deflection nearly cancels the angle, the source position is a small
        # difference of larger terms, so the deflection's rounding shows as an absolute error.
        _close(
            raytrace(x, y, params=params),
            lens.raytrace(x, y, params=params),
            atol=1e-12,
        )


def test_jit_forward_follows_a_switch_between_static_and_dynamic():
    lens, sie, shear = _plane()
    sie.Rein.to_dynamic()
    raytrace = jit_forward(lens.raytrace)
    x, y = _points()
    raytrace(x, y, params=lens.get_values())
    shear.gamma_1.to_dynamic(0.1)
    params = lens.get_values()
    _close(raytrace(x, y, params=params), lens.raytrace(x, y, params=params))


def test_jit_forward_reads_the_values_in_force_inside_an_active_context():
    lens, sie, _ = _plane()
    sie.Rein.to_dynamic()
    raytrace = jit_forward(lens.raytrace)
    x, y = _points()
    params = [_f64(1.3)]
    with ActiveContext(lens):
        lens.fill_params(params)
        got = raytrace(x, y)
        inside = _state(lens)
        raytrace(x, y)
        assert all(a is b for a, b in zip(_state(lens), inside))
    _close(got, lens.raytrace(x, y, params=params))


def test_jit_forward_leaves_the_lens_as_it_found_it():
    lens, sie, _ = _plane()
    sie.Rein.to_dynamic()
    before = _state(lens)
    x, y = _points()
    jit_forward(lens.raytrace)(x, y, params=lens.get_values())
    assert all(a is b for a, b in zip(_state(lens), before)) and not lens.active


def test_jit_forward_restores_the_lens_when_the_method_raises():
    lens, sie, _ = _plane()
    sie.Rein.to_dynamic(None)
    before = _state(lens)
    x, y = _points()
    with pytest.raises(Exception):
        jit_forward(lens.raytrace)(x, y)
    assert all(a is b for a, b in zip(_state(lens), before)) and not lens.active


@only_jax
def test_jit_forward_compiles_nothing_for_new_parameter_values():
    lens, sie, shear = _plane()
    sie.Rein.to_dynamic()
    raytrace = jit_forward(lens.raytrace)
    x, y = _points()
    first, second = _f64(1.0), _f64(1.3)
    raytrace(x, y, params=[first])
    shear.gamma_1 = 0.1
    before = compile_count()
    raytrace(x, y, params=[second])
    assert compile_count() == before


@only_jax
def test_jit_forward_frees_its_programs_with_the_lens():
    x, y = _points()

    def run():
        lens, _, _ = _plane()
        jit_forward(lens.raytrace)(x, y)
        return weakref.ref(lens)

    ref = run()
    gc.collect()
    assert ref() is None


def test_jit_forward_runs_a_multiplane_eagerly():
    cosmology = FlatLambdaCDM(name="cosmo")
    first = SIS(name="first", cosmology=cosmology, x0=0.3, y0=-0.2, Rein=1.0, z_l=0.5)
    second = SIS(name="second", cosmology=cosmology, x0=1.2, y0=0.9, Rein=0.2, z_l=1.0)
    lens = Multiplane(cosmology=cosmology, lenses=[first, second], z_s=2.0)
    raytrace = jit_forward(lens.raytrace)
    assert not raytrace.jitted
    x, y = _points()
    _close(raytrace(x, y), lens.raytrace(x, y))
