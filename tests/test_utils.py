import numpy as np
import pytest
from astropy.wcs import WCS

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
