import torch
import numpy as np
import pytest
from astropy.wcs import WCS

from caustics.utils import (
    meshgrid,
    pixel_to_world,
    world_to_pixel,
    plane_to_world_gnomonic,
    world_to_plane_gnomonic,
    pixel_to_plane,
    plane_to_pixel,
)
from caustics.constants import arcsec_to_deg

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

    px, py = meshgrid(pixelscale=0.1 * arcsec_to_deg, nx=100, dtype=torch.float64)

    ra, dec = plane_to_world_gnomonic(
        px, py, crval=torch.tensor(crval, dtype=torch.float64)
    )
    px2, py2 = world_to_plane_gnomonic(
        ra, dec, crval=torch.tensor(crval, dtype=torch.float64)
    )

    assert torch.allclose(px, px2, atol=1e-7)
    assert torch.allclose(py, py2, atol=1e-7)


@pytest.mark.parametrize("crpix", [(0.0, 0.0), (50.0, 50.0), (-10.0, -100.0)])
@pytest.mark.parametrize("crval", [(0.0, 0.0), (10.0, 89.9), (180.0, -45.0)])
@pytest.mark.parametrize(
    "CD",
    [((1e-4, 0.0), (0.0, 1e-4)), ((-9.72e-6, -1.03e-5), (-9.63e-6, 8.70e-6))],
)
def test_tangent_projection(crpix, crval, CD):
    crpix = torch.tensor(crpix, dtype=torch.float64)
    crval = torch.tensor(crval, dtype=torch.float64)
    CD = torch.tensor(CD, dtype=torch.float64)

    px, py = torch.meshgrid(
        torch.arange(0, 100, dtype=torch.float64),
        torch.arange(0, 100, dtype=torch.float64),
        indexing="ij",
    )

    ra, dec = pixel_to_plane(px, py, crpix=crpix, CD=CD)
    px2, py2 = plane_to_pixel(ra, dec, crpix=crpix, CD=CD)

    assert torch.allclose(px, px2, atol=1e-7)
    assert torch.allclose(py, py2, atol=1e-7)


@pytest.mark.parametrize("crpix", [(0.0, 0.0), (50.0, 50.0), (-10.0, -100.0)])
@pytest.mark.parametrize("crval", [(0.0, 0.0), (10.0, 89.9), (180.0, -45.0)])
@pytest.mark.parametrize(
    "CD",
    [((1e-4, 0.0), (0.0, 1e-4)), ((-9.72e-6, -1.03e-5), (-9.63e-6, 8.70e-6))],
)
def test_sip_wcs(crpix, crval, CD):
    crpix = torch.tensor(crpix, dtype=torch.float64)
    crval = torch.tensor(crval, dtype=torch.float64)
    CD = torch.tensor(CD, dtype=torch.float64)
    # SIP coefficients from a random HST image
    sip_powers = torch.tensor(_sip_powers, dtype=torch.float64)
    sip_coefs = torch.tensor(_sip_coefs, dtype=torch.float64)

    px, py = torch.meshgrid(
        torch.arange(0, 100, dtype=torch.float64),
        torch.arange(0, 100, dtype=torch.float64),
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

    assert torch.allclose(px, px2, atol=1e-3)
    assert torch.allclose(py, py2, atol=1e-3)


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
        torch.arange(0, 2048)[::4],
        indexing="ij",
    )

    # Astropy
    astropy_wx, astropy_wy = astropy_wcs.pixel_to_world_values(px, py)

    # Caustics
    crpix = torch.tensor(crpix, dtype=torch.float64) - 1
    crval = torch.tensor(crval, dtype=torch.float64)
    CD = torch.tensor(
        (
            (wcs_info["CD1_1"], wcs_info["CD1_2"]),
            (wcs_info["CD2_1"], wcs_info["CD2_2"]),
        ),
        dtype=torch.float64,
    )
    caustics_wx, caustics_wy = pixel_to_world(
        torch.tensor(px, dtype=torch.float64),
        torch.tensor(py, dtype=torch.float64),
        crpix=crpix,
        crval=crval,
        CD=CD,
    )

    # Compare
    # Use modulo to enforce periodicity
    assert np.allclose(
        astropy_wx % 360, caustics_wx.detach().cpu().numpy() % 360, atol=1e-7
    )
    assert np.allclose(
        (astropy_wy + 90) % 180,
        (caustics_wy.detach().cpu().numpy() + 90) % 180,
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
        torch.arange(0, 2048)[::4],
        indexing="ij",
    )

    # Astropy
    astropy_wx, astropy_wy = astropy_wcs.pixel_to_world_values(px, py)

    # Caustics
    sip_powers = torch.tensor(_sip_powers, dtype=torch.float64)
    sip_coefs = torch.tensor(_sip_coefs, dtype=torch.float64)
    crpix = torch.tensor(crpix, dtype=torch.float64) - 1
    crval = torch.tensor(crval, dtype=torch.float64)
    CD = torch.tensor(
        (
            (wcs_info["CD1_1"], wcs_info["CD1_2"]),
            (wcs_info["CD2_1"], wcs_info["CD2_2"]),
        ),
        dtype=torch.float64,
    )
    caustics_wx, caustics_wy = pixel_to_world(
        torch.tensor(px, dtype=torch.float64),
        torch.tensor(py, dtype=torch.float64),
        crpix=crpix,
        crval=crval,
        CD=CD,
        sip_powers=sip_powers,
        sip_coefs=sip_coefs,
    )

    # Compare
    # Use modulo to enforce periodicity
    assert np.allclose(
        astropy_wx % 360, caustics_wx.detach().cpu().numpy() % 360, atol=1e-7
    )
    assert np.allclose(
        (astropy_wy + 90) % 180,
        (caustics_wy.detach().cpu().numpy() + 90) % 180,
        atol=1e-7,
    )
