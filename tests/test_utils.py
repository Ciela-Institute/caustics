import gc
import weakref

import numpy as np
import pytest
from astropy.wcs import WCS

import caustics.utils as utils
from caustics.utils import (
    batch_lm,
    meshgrid,
    pixel_to_world,
    world_to_pixel,
    plane_to_world_gnomonic,
    world_to_plane_gnomonic,
    pixel_to_plane,
    plane_to_pixel,
)
from caustics.constants import arcsec_to_deg
from caustics.backend_obj import backend
from jax_compiles import compile_count

_sip_powers = (
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 0),
    (2, 1),
    (2, 2),
    (3, 0),
    (3, 1),
    (3, 2),
    (4, 0),
    (4, 1),
    (5, 0),
)
_sip_coefs = (
    (2.26e-6, -9.78e-6),
    (8.90e-11, -3.71e-10),
    (1.40e-13, -1.60e-13),
    (-1.65e-17, -1.64e-17),
    (-7.53e-6, 6.45e-6),
    (-5.04e-10, -2.88e-11),
    (-1.19e-14, 9.53e-15),
    (6.30e-18, -1.76e-17),
    (8.52e-6, -2.96e-6),
    (-1.12e-10, -4.00e-10),
    (9.68e-14, -1.14e-13),
    (-5.05e-18, -1.18e-17),
    (-4.89e-10, 1.17e-10),
    (3.77e-14, -4.20e-14),
    (-9.51e-18, 2.45e-18),
    (1.97e-14, 1.90e-14),
    (2.42e-18, -8.86e-18),
    (4.85e-18, -6.06e-18),
)


@pytest.mark.parametrize("crval", [(0.0, 0.0), (10.0, 90.0), (180.0, -45.0)])
def test_gnomonic_projection(crval):

    px, py = meshgrid(pixelscale=0.1 * arcsec_to_deg, nx=100, dtype=backend.float64)

    ra, dec = plane_to_world_gnomonic(
        px, py, crval=backend.as_array(crval, dtype=backend.float64)
    )
    px2, py2 = world_to_plane_gnomonic(
        ra, dec, crval=backend.as_array(crval, dtype=backend.float64)
    )

    assert backend.allclose(px, px2, atol=1e-7)
    assert backend.allclose(py, py2, atol=1e-7)


@pytest.mark.parametrize("crpix", [(0.0, 0.0), (50.0, 50.0), (-10.0, -100.0)])
@pytest.mark.parametrize("crval", [(0.0, 0.0), (10.0, 89.9), (180.0, -45.0)])
@pytest.mark.parametrize(
    "CD",
    [((1e-4, 0.0), (0.0, 1e-4)), ((-9.72e-6, -1.03e-5), (-9.63e-6, 8.70e-6))],
)
def test_tangent_projection(crpix, crval, CD):
    crpix = backend.as_array(crpix, dtype=backend.float64)
    crval = backend.as_array(crval, dtype=backend.float64)
    CD = backend.as_array(CD, dtype=backend.float64)

    px, py = backend.meshgrid(
        backend.arange(0, 100, dtype=backend.float64),
        backend.arange(0, 100, dtype=backend.float64),
        indexing="ij",
    )

    ra, dec = pixel_to_plane(px, py, crpix=crpix, CD=CD)
    px2, py2 = plane_to_pixel(ra, dec, crpix=crpix, CD=CD)

    assert backend.allclose(px, px2, atol=1e-7)
    assert backend.allclose(py, py2, atol=1e-7)


@pytest.mark.parametrize("crpix", [(0.0, 0.0), (50.0, 50.0), (-10.0, -100.0)])
@pytest.mark.parametrize("crval", [(0.0, 0.0), (10.0, 89.9), (180.0, -45.0)])
@pytest.mark.parametrize(
    "CD",
    [((1e-4, 0.0), (0.0, 1e-4)), ((-9.72e-6, -1.03e-5), (-9.63e-6, 8.70e-6))],
)
def test_sip_wcs(crpix, crval, CD):
    crpix = backend.as_array(crpix, dtype=backend.float64)
    crval = backend.as_array(crval, dtype=backend.float64)
    CD = backend.as_array(CD, dtype=backend.float64)
    # SIP coefficients from a random HST image
    sip_powers = backend.as_array(_sip_powers, dtype=backend.float64)
    sip_coefs = backend.as_array(_sip_coefs, dtype=backend.float64)

    px, py = backend.meshgrid(
        backend.arange(0, 100, dtype=backend.float64),
        backend.arange(0, 100, dtype=backend.float64),
        indexing="ij",
    )

    ra, dec = pixel_to_world(
        px,
        py,
        crpix=crpix,
        crval=crval,
        CD=CD,
        sip_powers=sip_powers,
        sip_coefs=sip_coefs,
    )
    px2, py2 = world_to_pixel(
        ra,
        dec,
        crpix=crpix,
        crval=crval,
        CD=CD,
        sip_powers=sip_powers,
        sip_coefs=-sip_coefs,
    )

    assert backend.allclose(px, px2, atol=1e-3)
    assert backend.allclose(py, py2, atol=1e-3)


@pytest.mark.parametrize("crpix", [(0.0, 0.0), (2048, 1024)])
@pytest.mark.parametrize("crval", [(0.0, 0.0), (217.54699362734, 41.080926847432)])
def test_caustics_vs_astropy_wcs_linear(crpix, crval):

    # taken from a FITS header slight modifications
    wcs_info = {
        "CRPIX1": crpix[0],
        "CRPIX2": crpix[1],
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CD1_1": -9.708567438787e-5,
        "CD1_2": -1.0614686517095e-4,
        "CD2_1": -9.8726557770183e-5,
        "CD2_2": 9.088209859399e-5,
    }
    astropy_wcs = WCS(wcs_info)

    px, py = np.meshgrid(
        np.arange(0, 4096)[::4],
        backend.arange(0, 2048)[::4],
        indexing="ij",
    )

    # Astropy
    astropy_wx, astropy_wy = astropy_wcs.pixel_to_world_values(px, py)

    # Caustics
    crpix = backend.as_array(crpix, dtype=backend.float64) - 1
    crval = backend.as_array(crval, dtype=backend.float64)
    CD = backend.as_array(
        (
            (wcs_info["CD1_1"], wcs_info["CD1_2"]),
            (wcs_info["CD2_1"], wcs_info["CD2_2"]),
        ),
        dtype=backend.float64,
    )
    caustics_wx, caustics_wy = pixel_to_world(
        backend.as_array(px, dtype=backend.float64),
        backend.as_array(py, dtype=backend.float64),
        crpix=crpix,
        crval=crval,
        CD=CD,
    )

    # Compare
    # Use modulo to enforce periodicity
    assert np.allclose(astropy_wx % 360, backend.to_numpy(caustics_wx) % 360, atol=1e-7)
    assert np.allclose(
        (astropy_wy + 90) % 180,
        (backend.to_numpy(caustics_wy) + 90) % 180,
        atol=1e-7,
    )


@pytest.mark.parametrize("crpix", [(0.0, 0.0), (2048, 1024)])
@pytest.mark.parametrize("crval", [(0.0, 0.0), (217.54699362734, 41.080926847432)])
def test_caustics_vs_astropy_wcs_sip(crpix, crval):

    # taken from a FITS header slight modifications
    wcs_info = {
        "CRPIX1": crpix[0],
        "CRPIX2": crpix[1],
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CD1_1": -9.708567438787e-5,
        "CD1_2": -1.0614686517095e-4,
        "CD2_1": -9.8726557770183e-5,
        "CD2_2": 9.088209859399e-5,
        "A_ORDER": 5,
        "B_ORDER": 5,
    }
    for powers, coefs in zip(_sip_powers, _sip_coefs):
        wcs_info[f"A_{powers[0]}_{powers[1]}"] = coefs[0]
        wcs_info[f"B_{powers[0]}_{powers[1]}"] = coefs[1]

    astropy_wcs = WCS(wcs_info)

    px, py = np.meshgrid(
        np.arange(0, 4096)[::4],
        backend.arange(0, 2048)[::4],
        indexing="ij",
    )

    # Astropy
    astropy_wx, astropy_wy = astropy_wcs.pixel_to_world_values(px, py)

    # Caustics
    sip_powers = backend.as_array(_sip_powers, dtype=backend.float64)
    sip_coefs = backend.as_array(_sip_coefs, dtype=backend.float64)
    crpix = backend.as_array(crpix, dtype=backend.float64) - 1
    crval = backend.as_array(crval, dtype=backend.float64)
    CD = backend.as_array(
        (
            (wcs_info["CD1_1"], wcs_info["CD1_2"]),
            (wcs_info["CD2_1"], wcs_info["CD2_2"]),
        ),
        dtype=backend.float64,
    )
    caustics_wx, caustics_wy = pixel_to_world(
        backend.as_array(px, dtype=backend.float64),
        backend.as_array(py, dtype=backend.float64),
        crpix=crpix,
        crval=crval,
        CD=CD,
        sip_powers=sip_powers,
        sip_coefs=sip_coefs,
    )

    # Compare
    # Use modulo to enforce periodicity
    assert np.allclose(astropy_wx % 360, backend.to_numpy(caustics_wx) % 360, atol=1e-7)
    assert np.allclose(
        (astropy_wy + 90) % 180,
        (backend.to_numpy(caustics_wy) + 90) % 180,
        atol=1e-7,
    )


def _rosenbrock(x):
    """Residuals whose least-squares root is ``(1, 1)``.

    A long curved valley, so Levenberg-Marquardt needs many iterations from a
    distant start while a nearby start converges almost immediately -- exactly the
    mix of easy and hard elements a batched solve has to survive.
    """
    return backend.stack((10 * (x[..., 1] - x[..., 0] ** 2), 1 - x[..., 0]), dim=-1)


def _solve(starts, **kw):
    X = backend.as_array(np.asarray(starts, dtype=np.float64))
    Y = backend.as_array(np.zeros((len(starts), 2)))
    root, _, _ = batch_lm(X, Y, _rosenbrock, **kw)
    return backend.to_numpy(root)


def test_batch_lm_converges_on_a_slow_element_alone():
    """Baseline: the distant start does reach the root when solved by itself."""
    assert np.allclose(_solve([[-1.2, 1.0]])[0], [1.0, 1.0], atol=1e-4)


def test_batch_lm_does_not_abandon_a_slow_element_when_others_converge():
    """A batch must not stop iterating while one of its elements is still moving.

    The stopping rule reads "step is small" as convergence, but a *rejected* step
    leaves ``X`` exactly unchanged -- so an element whose trial steps are being
    rejected, which is precisely one that needs more iterations, scores as
    finished. Paired with a majority test on the damping, two easy elements can
    end the loop while a third is still at its starting point.
    """
    roots = _solve([[-1.2, 1.0], [0.99, 0.99], [0.99, 0.99]])
    assert np.allclose(
        roots[0], [1.0, 1.0], atol=1e-4
    ), f"slow element abandoned at {roots[0]}"


def test_batch_lm_is_independent_of_batch_composition():
    """The property the bug above violates, stated directly.

    A batched solve must agree with solving each element on its own; the elements
    are mathematically independent and only the termination is shared.
    """
    alone = _solve([[-1.2, 1.0]])[0]
    for companions in ([[0.99, 0.99]], [[0.99, 0.99], [1.01, 1.02]]):
        together = _solve([[-1.2, 1.0]] + companions)[0]
        assert np.allclose(
            alone, together, atol=1e-6
        ), f"same element gave {alone} alone and {together} with {len(companions)} companions"


# Alone, the first start converges at the 5th iteration and the second at the
# 13th. At its root an element rejects nearly every step, and its damping
# climbs by `L_up` each time: by the 13th the first has left the converged
# state. Without sticky convergence a batch of both runs on until both reach
# `L_max`, which counts as done, at the 33rd.
STAGGERED = [[0.99, 0.99], [0.92, -1.3]]


def _iterations(monkeypatch, starts, **kw):
    """How many iterations ``batch_lm``'s Python loop runs on ``starts``."""
    calls = []
    iterate = utils._iterate
    monkeypatch.setattr(utils, "_iterate", lambda *a: calls.append(None) or iterate(*a))
    _solve(starts, **kw)
    monkeypatch.setattr(utils, "_iterate", iterate)
    return len(calls)


@pytest.mark.parametrize("jit", [False, True])
def test_batch_lm_stops_once_its_slowest_element_has_converged(monkeypatch, jit):
    """A batch runs as many iterations as its slowest element alone, though the others converged iterations earlier."""
    if jit and backend.backend == "jax":
        pytest.skip("under jax a jitted solve loops on the device")
    slowest = _iterations(monkeypatch, STAGGERED[1:], jit=jit)
    assert _iterations(monkeypatch, STAGGERED[:1], jit=jit) < slowest
    assert _iterations(monkeypatch, STAGGERED, jit=jit) == slowest


@pytest.mark.parametrize("jit", [False, True])
def test_batch_lm_returns_each_element_as_it_returns_it_alone(jit):
    """Root, damping and chi^2 alike, though the elements converge iterations apart.

    Under jax with ``jit=True`` the loop runs on the device, so this is what
    shows there that the batch does not keep stepping a converged element.
    """

    def lm(starts):
        X = backend.as_array(np.asarray(starts, dtype=np.float64))
        Y = backend.as_array(np.zeros((len(starts), 2)))
        return [backend.to_numpy(a) for a in batch_lm(X, Y, _rosenbrock, jit=jit)]

    together = lm(STAGGERED)
    for k, start in enumerate(STAGGERED):
        for name, alone, both in zip(("root", "L", "chi2"), lm([start]), together):
            np.testing.assert_allclose(
                both[k], alone[0], rtol=1e-12, atol=1e-30, err_msg=f"{name} of {start}"
            )


def _sis(x):
    """Residuals of a singular isothermal sphere, ``x - x / |x|``, whose caustic is the point ``0``."""
    r = backend.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
    return backend.stack((x[..., 0] - x[..., 0] / r, x[..., 1] - x[..., 1] / r), dim=-1)


@pytest.mark.parametrize("jit", [False, True])
def test_batch_lm_does_not_stop_on_a_rejected_step(jit):
    """A rejected step leaves ``X`` where it was, which is no sign of a root.

    ``beta = 1e-4 u`` has an image at ``(1 + 1e-4) u``, where the map is
    nearly flat along the critical curve ``|x| = 1``. From a start rotated
    ``1e-3`` off it along the curve, the solve rejects a step at a damping
    far below ``1e-2`` and then crawls to the image; reading the unchanged
    ``X`` of that rejection as a zero step stops it ``8e-4`` from the image.
    """
    t, u = 1e-4, np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    image = (1 + t) * u
    c, s = np.cos(1e-3), np.sin(1e-3)
    start = np.array([[c, -s], [s, c]]) @ image
    X, Y = backend.as_array(start[None]), backend.as_array((t * u)[None])
    root, _, _ = batch_lm(X, Y, _sis, stopping=1e-10, jit=jit)
    gap = np.linalg.norm(backend.to_numpy(root)[0] - image)
    assert gap < 1e-8, f"stopped {gap:.3g} from the image"


def test_batch_lm_returns_one_row_per_element_whatever_the_padding():
    """Under jax three elements run padded to 64 rows; three come back."""
    X = backend.as_array(np.asarray([[-1.2, 1.0], [0.99, 0.99], [0.5, 0.5]]))
    Y = backend.as_array(np.zeros((3, 2)))
    root, L, chi2 = batch_lm(X, Y, _rosenbrock, jit=True)
    assert tuple(root.shape) == (3, 2)
    assert tuple(L.shape) == (3,) and tuple(chi2.shape) == (3,)
    assert np.allclose(backend.to_numpy(root), 1.0, atol=1e-4)


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_batch_lm_compiles_a_handful_of_programs_for_a_new_batch_size():
    """The step is jitted over a padded batch, so a batch size it has not seen
    costs a few compiles rather than one for every primitive of the step."""
    import jax

    compiles = []
    jax.monitoring.register_event_duration_secs_listener(
        lambda event, duration, **kw: (
            compiles.append(event)
            if event == "/jax/core/compile/backend_compile_duration"
            else None
        )
    )
    # Another test may have compiled these very shapes already.
    jax.clear_caches()
    _solve([[-1.2, 1.0], [0.99, 0.99]], jit=True)
    before = len(compiles)
    _solve([[-1.2, 1.0], [0.99, 0.99], [0.5, 0.5]], jit=True)
    assert len(compiles) - before <= 8


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_batch_lm_tests_termination_inside_the_jitted_step():
    """A padded size the process has not seen costs about what a seen one does:
    the per-iteration termination test is compiled with the step, not op by op."""
    import jax

    compiles = []
    jax.monitoring.register_event_duration_secs_listener(
        lambda event, duration, **kw: (
            compiles.append(event)
            if event == "/jax/core/compile/backend_compile_duration"
            else None
        )
    )
    jax.clear_caches()
    _solve([[-1.2, 1.0], [0.99, 0.99]], jit=True)
    before = len(compiles)
    _solve([[-1.2, 1.0]] * 100, jit=True)  # padded to 128, a size not seen yet
    assert len(compiles) - before <= 10


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_batch_lm_runs_eagerly_unless_asked_to_jit():
    """A caller can hand ``batch_lm`` a fresh closure every call: a jitted step
    would compile again each time, where eager ops reuse jax's cache."""
    import jax

    compiles = []
    jax.monitoring.register_event_duration_secs_listener(
        lambda event, duration, **kw: (
            compiles.append(event)
            if event == "/jax/core/compile/backend_compile_duration"
            else None
        )
    )
    _solve([[-1.2, 1.0], [0.99, 0.99]])
    before = len(compiles)
    _solve([[-1.2, 1.0], [0.99, 0.99]])
    assert len(compiles) == before


def _offset_rosenbrock(x, a):
    """`_rosenbrock` with its root moved to ``(1, 1) + a``."""
    return _rosenbrock(x - a)


@pytest.mark.parametrize("jit", [False, True])
def test_batch_lm_reads_each_done_flag_one_iteration_late(monkeypatch, jit):
    """The Python loop never waits on the iteration it just launched: it stops
    one iteration after the first whose batch is done."""
    if jit and backend.backend == "jax":
        pytest.skip("under jax a jitted solve loops on the device")
    dones = []
    iterate = utils._iterate

    def spy(*args):
        out = iterate(*args)
        dones.append(bool(backend.to_numpy(out[-1])))
        return out

    monkeypatch.setattr(utils, "_iterate", spy)
    _solve([[-1.2, 1.0], [0.99, 0.99]], jit=jit)
    assert True in dones
    assert len(dones) == min(dones.index(True) + 2, 50)


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_batch_lm_reuses_its_compiled_solve_for_new_f_args():
    starts = np.asarray([[-1.2, 1.0], [0.99, 0.99]])
    X0, X1 = backend.to(starts), backend.to(starts + 0.5)
    Y = backend.to(np.zeros((2, 2)))
    a0, a1 = backend.to(np.zeros(2)), backend.to(np.full(2, 0.5))
    batch_lm(X0, Y, _offset_rosenbrock, f_args=(a0,), jit=True)
    before = compile_count()
    root, _, _ = batch_lm(X1, Y, _offset_rosenbrock, f_args=(a1,), jit=True)
    assert compile_count() == before
    np.testing.assert_allclose(backend.to_numpy(root), 1.5, atol=1e-4)


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_batch_lm_compiles_nothing_for_a_new_batch_size_that_pads_alike():
    def put(starts):
        return backend.to(np.asarray(starts)), backend.to(np.zeros((len(starts), 2)))

    batch_lm(*put([[-1.2, 1.0], [0.99, 0.99]]), _rosenbrock, jit=True)
    X, Y = put([[-1.2, 1.0], [0.99, 0.99], [0.5, 0.5]])
    before = compile_count()
    root, L, chi2 = batch_lm(X, Y, _rosenbrock, jit=True)
    assert compile_count() == before
    assert tuple(root.shape) == (3, 2) and tuple(L.shape) == tuple(chi2.shape) == (3,)


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_batch_lm_frees_its_compiled_solve_with_f():
    def f(x):
        return _rosenbrock(x)

    X, Y = backend.to(np.asarray([[-1.2, 1.0]])), backend.to(np.zeros((1, 2)))
    batch_lm(X, Y, f, jit=True)
    ref = weakref.ref(f)
    del f
    gc.collect()
    assert ref() is None
